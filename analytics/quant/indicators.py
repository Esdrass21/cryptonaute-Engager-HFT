"""Quant analytics (Layer 2 — §3.1).

Technical indicators implemented in pure numpy/pandas so the platform works without the
optional ``ta-lib`` / ``pandas-ta`` binaries. ``ta-lib`` is used when available for speed.

Produces :class:`EventType.TECH_SIGNAL` events per symbol from closed candles. Each signal
carries a direction (-1/0/+1) and a strength in [0,1] plus the raw indicator values, so the
Layer 4 fusion can weight them.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

log = logging.getLogger("analytics.quant")


@dataclass
class Candle:
    t: float
    o: float
    h: float
    l: float
    c: float
    v: float


def sma(series: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(series).rolling(n).mean().to_numpy()


def ema(series: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(series).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(series: np.ndarray, n: int = 14) -> np.ndarray:
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)).to_numpy()


def atr(c: List[Candle], n: int = 14) -> float:
    tr = np.array([
        max(c[i].h - c[i].l, abs(c[i].h - c[i - 1].c), abs(c[i].l - c[i - 1].c))
        for i in range(1, len(c))
    ])
    return float(np.mean(tr[-n:])) if len(tr) else 0.0


def macd(series: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = ema(series, fast)
    ema_s = ema(series, slow)
    macd_line = ema_f - ema_s
    signal = pd.Series(macd_line).rolling(sig).mean().to_numpy()
    hist = macd_line - signal
    return macd_line, signal, hist


class QuantAnalytics(Component):
    """Consumes CANDLE events, computes indicators, emits TECH_SIGNAL.

    EMA and RSI are maintained *incrementally* (stateful, O(1) per candle) instead of
    recomputing a full-window pandas rolling each time — this is the dominant cost on the
    hot path and was previously O(lookback) per candle / O(lookback^2) per run.
    """

    def __init__(self, bus, cfg, clock, metrics, lookback: int = 200):
        super().__init__(bus, cfg, clock, metrics, name="analytics.quant")
        self.lookback = lookback
        self._candles: Dict[str, list] = {}          # bounded list of Candle (last `lookback`)
        self.symbols = cfg.get("symbols", [])
        self._fast = int(cfg.get("analytics.quant.ma_fast", 12))
        self._slow = int(cfg.get("analytics.quant.ma_slow", 26))
        self._rsi_n = int(cfg.get("analytics.quant.rsi_n", 14))
        # incremental state per symbol
        self._ema_f: Dict[str, float] = {}
        self._ema_s: Dict[str, float] = {}
        self._ema_f_seeded: Dict[str, bool] = {}
        self._ema_s_seeded: Dict[str, bool] = {}
        self._rsi_buf: Dict[str, list] = {}          # last rsi_n deltas (rolling SMA of gain/loss)
        self._rsi_seeded: Dict[str, bool] = {}
        self._recent: Dict[str, list] = {}           # last <=30 closes for vol estimate

    def start(self) -> None:
        self._running = True
        for s in self.symbols:
            self._candles[s] = []
            self._ema_f[s] = 0.0
            self._ema_s[s] = 0.0
            self._ema_f_seeded[s] = False
            self._ema_s_seeded[s] = False
            self._rsi_buf[s] = []
            self._rsi_seeded[s] = False
            self._recent[s] = []
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        log.info("Quant analytics online (symbols=%s)", self.symbols)

    def on_candle(self, ev: NormalizedEvent) -> None:
        sym = ev.payload["symbol"]
        if sym not in self._candles:
            self._candles[sym] = []
            self._ema_f[sym] = self._ema_s[sym] = 0.0
            self._ema_f_seeded[sym] = self._ema_s_seeded[sym] = False
            self._rsi_buf[sym] = []
            self._rsi_seeded[sym] = False
            self._recent[sym] = []
        p = ev.payload
        c = p["close"]
        bars = self._candles[sym]
        bars.append(Candle(p["time"] if "time" in p else self.clock.now(),
                           p["open"], p["high"], p["low"], c, p["volume"]))
        if len(bars) > self.lookback:
            bars.pop(0)
        self._recent[sym].append(c)
        if len(self._recent[sym]) > 30:
            self._recent[sym].pop(0)
        # update incremental indicator state on EVERY candle (not only when emitting),
        # so the EMA/RSI start from the first close and match the pure reference exactly.
        self._update_ema(sym, c)
        self._update_rsi(sym, c)
        if len(bars) >= self._slow + 2:
            self._emit_signal(sym)

    def _update_ema(self, sym: str, price: float) -> None:
        af = 2.0 / (self._fast + 1)
        as_ = 2.0 / (self._slow + 1)
        if not self._ema_f_seeded[sym]:
            self._ema_f[sym] = price
            self._ema_f_seeded[sym] = True
        else:
            self._ema_f[sym] = price * af + self._ema_f[sym] * (1 - af)
        if not self._ema_s_seeded[sym]:
            self._ema_s[sym] = price
            self._ema_s_seeded[sym] = True
        else:
            self._ema_s[sym] = price * as_ + self._ema_s[sym] * (1 - as_)

    def _update_rsi(self, sym: str, price: float) -> float:
        buf = self._rsi_buf[sym]
        n = self._rsi_n
        if not buf:
            buf.append(price)
            return 50.0
        prev = buf[-1]
        delta = price - prev
        if len(buf) <= n:                      # warm-up: simple rolling window
            buf.append(price)
        else:
            buf.append(price)
            buf.pop(0)
        gains = [max(0.0, buf[i] - buf[i - 1]) for i in range(1, len(buf))]
        losses = [max(0.0, buf[i - 1] - buf[i]) for i in range(1, len(buf))]
        ag = sum(gains) / len(gains)
        al = sum(losses) / len(losses)
        rs = ag / (al + 1e-9)
        return float(100 - 100 / (1 + rs))

    def _emit_signal(self, sym: str) -> None:
        bars = self._candles[sym]
        c = bars[-1].c
        # state (EMA/RSI) already updated per-candle in on_candle; read it here.
        ma_f = self._ema_f[sym]
        ma_s = self._ema_s[sym]
        ma_dir = 1.0 if ma_f > ma_s else -1.0
        # Trend conviction = (fast-slow EMA gap) normalized by recent volatility — a
        # "trend z-score". Persistent drift produces a stable, strong conviction in [-1, 1].
        recent = self._recent[sym]
        vol = float(np.std(recent)) + 1e-9 if len(recent) >= 2 else 1e-9
        trend_conv = float(np.clip((ma_f - ma_s) / vol, -1.0, 1.0))
        r = self._rsi_buf_state(sym)
        if np.isnan(r):
            r = 50.0
        if r > 70:
            rsi_dir, rsi_strength = -1.0, float((r - 70) / 30)
        elif r < 30:
            rsi_dir, rsi_strength = 1.0, float((30 - r) / 30)
        else:
            rsi_dir, rsi_strength = 0.0, 0.0
        # Composite = trend conviction (signed), lightly dampened only when RSI is stretched.
        rsi_damp = rsi_dir * rsi_strength * 0.3
        signal = float(np.clip(trend_conv + rsi_damp, -1.0, 1.0))
        strength = float(min(1.0, abs(trend_conv)))
        ev = make_event(
            EventType.TECH_SIGNAL, "analytics.quant",
            payload={"symbol": sym, "direction": float(np.sign(signal)),
                      "strength": strength,
                      "ma_fast": float(ma_f), "ma_slow": float(ma_s),
                      "rsi": float(r), "composite": signal},
            symbol_impact=[sym], timestamp=self.clock.now(),
        )
        self.emit(ev)
        self.metrics.inc("tech_signal_emitted")

    def _rsi_buf_state(self, sym: str) -> float:
        """Final RSI from the rolling delta buffer (no mutation — state set in on_candle)."""
        buf = self._rsi_buf[sym]
        if len(buf) < 2:
            return 50.0
        gains = [max(0.0, buf[i] - buf[i - 1]) for i in range(1, len(buf))]
        losses = [max(0.0, buf[i - 1] - buf[i]) for i in range(1, len(buf))]
        ag = sum(gains) / len(gains)
        al = sum(losses) / len(losses)
        rs = ag / (al + 1e-9)
        return float(100 - 100 / (1 + rs))
