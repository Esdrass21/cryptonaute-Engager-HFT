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
    """Consumes CANDLE events, computes indicators, emits TECH_SIGNAL."""

    def __init__(self, bus, cfg, clock, metrics, lookback: int = 200):
        super().__init__(bus, cfg, clock, metrics, name="analytics.quant")
        self.lookback = lookback
        self._bars: Dict[str, List[Candle]] = {}
        self.symbols = cfg.get("symbols", [])
        self._fast = int(cfg.get("analytics.quant.ma_fast", 12))
        self._slow = int(cfg.get("analytics.quant.ma_slow", 26))
        self._rsi_n = int(cfg.get("analytics.quant.rsi_n", 14))

    def start(self) -> None:
        self._running = True
        for s in self.symbols:
            self._bars[s] = []
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        log.info("Quant analytics online (symbols=%s)", self.symbols)

    def on_candle(self, ev: NormalizedEvent) -> None:
        sym = ev.payload["symbol"]
        if sym not in self._bars:
            self._bars[sym] = []
        p = ev.payload
        self._bars[sym].append(Candle(p["time"] if "time" in p else self.clock.now(),
                                       p["open"], p["high"], p["low"], p["close"], p["volume"]))
        if len(self._bars[sym]) > self.lookback:
            self._bars[sym] = self._bars[sym][-self.lookback:]
        if len(self._bars[sym]) >= self._slow + 2:
            self._emit_signal(sym)

    def _emit_signal(self, sym: str) -> None:
        bars = self._bars[sym]
        closes = np.array([b.c for b in bars])
        ma_f = ema(closes, self._fast)[-1]
        ma_s = ema(closes, self._slow)[-1]
        ma_dir = 1.0 if ma_f > ma_s else -1.0
        # Trend conviction = (fast-slow EMA gap) normalized by recent volatility — a
        # "trend z-score". Persistent drift produces a stable, strong conviction in [-1, 1].
        recent = closes[-30:] if len(closes) >= 30 else closes
        vol = float(np.std(recent)) + 1e-9
        trend_conv = float(np.clip((ma_f - ma_s) / vol, -1.0, 1.0))
        r = rsi(closes, self._rsi_n)[-1]
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
