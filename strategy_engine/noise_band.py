# strategy_engine/noise_band.py
"""
Noise-band breakout strategy.

Thesis: every market has a "noise floor" — the typical per-candle move you must
ignore. We measure it with a rolling sigma of returns over `lookback` candles.
Moves within k*sigma of zero are noise: no trade. When price closes beyond
k*sigma in the SAME direction for `confirm_n` consecutive candles, the move has
outgrown the noise band: treat it as drift, enter with the drift, place SL/TP
scaled to the band width, and exit as soon as price falls back inside the band
(noise resumed — the drift thesis is dead).

Everything goes through the standard path: ORDER_PROPOSAL (execute=False,
passes_ev=True) -> FSM -> signed risk veto -> execution. Nothing bypasses L6.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent, make_event
from core.metrics import Metrics

log = logging.getLogger("strategy.noise_band")


class NoiseBandStrategy(Component):
    def __init__(self, bus: MessageBus, cfg: Config, clock: Clock, metrics: Metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="strategy.noise_band")
        sec = "strategy.noise_band"
        self.lookback = int(cfg.get(f"{sec}.lookback", 60))
        self.k = float(cfg.get(f"{sec}.k", 2.0))
        self.confirm_n = int(cfg.get(f"{sec}.confirm_n", 2))
        self.min_bars = int(cfg.get(f"{sec}.min_bars", 30))
        self.min_band = float(cfg.get(f"{sec}.min_band", 0.00005))
        self.sl_mult = float(cfg.get(f"{sec}.sl_mult", 2.0))
        self.tp_mult = float(cfg.get(f"{sec}.tp_mult", 4.0))
        self.position_pct = float(cfg.get(f"{sec}.position_pct", 0.5))
        self.max_holding_s = float(cfg.get(f"{sec}.max_holding_s", 14400.0))
        self.cooldown_s = float(cfg.get(f"{sec}.cooldown_s", 1800.0))
        self.max_concurrent = int(cfg.get(f"{sec}.max_concurrent", 1))
        self.enabled = bool(cfg.get(f"{sec}.enabled", True))
        self.symbols_allowed = set(cfg.get(f"{sec}.symbols", cfg.get("symbols", ["EURUSD"])))
        self._closes: dict[str, deque] = {}
        self._open: dict[str, dict] = {}
        self._cooldown: dict[str, float] = {}
        self._armed: dict[str, float] = {}  # symbol -> direction of last proposal

    # ---- helpers --------------------------------------------------------
    def _returns(self, sym: str) -> np.ndarray:
        closes = list(self._closes[sym])
        prices = np.asarray(closes, dtype=float)
        return np.diff(prices) / prices[:-1]

    def _band(self, rets: np.ndarray) -> float:
        sigma = float(np.std(rets))
        return max(sigma * self.k, self.min_band)

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        log.info(
            "NoiseBandStrategy online (lookback=%d k=%.2f confirm=%d min_bars=%d)",
            self.lookback, self.k, self.confirm_n, self.min_bars,
        )

    # ---- event handlers -------------------------------------------------
    def on_candle(self, ev: NormalizedEvent) -> None:
        if not self.enabled:
            return
        p = ev.payload
        sym = p.get("symbol")
        if sym is None or sym not in self.symbols_allowed:
            return
        if not p.get("closed", True):
            return
        self._closes.setdefault(sym, deque(maxlen=self.lookback + 1)).append(float(p["close"]))
        if len(self._closes[sym]) < self.min_bars:
            return

        rets = self._returns(sym)
        band = self._band(rets)
        last_ret = float(rets[-1])
        now = self.clock.now()

        # manage open position: exit when noise resumes (re-entry inside band)
        entry = self._open.get(sym)
        if entry is not None:
            if (now - entry["entry_ts"] >= self.max_holding_s
                    or (entry["direction"] > 0 and last_ret < band / self.k * 0.5)
                    or (entry["direction"] < 0 and last_ret > -band / self.k * 0.5)):
                self.emit(make_event(
                    EventType.ORDER_EXIT, self.name,
                    payload={"symbol": sym, "reason": "noise_reentry", "execute": True},
                    symbol_impact=[sym], timestamp=now,
                ))
                self.metrics.inc("noise_band_exit", symbol=sym)
                self._open.pop(sym, None)
                self._cooldown[sym] = now + self.cooldown_s
            return

        # entry: confirm_n consecutive returns beyond the band, same sign
        streak = 0
        side = 0
        for r in reversed(rets):
            if abs(r) < band:
                break
            if side == 0:
                side = 1 if r > 0 else -1
                streak = 1
            elif (r > 0) == (side > 0):
                streak += 1
            else:
                break
        if streak < self.confirm_n:
            return
        if now < self._cooldown.get(sym, 0.0):
            self.metrics.inc("noise_band_cooldown", symbol=sym)
            return
        if len(self._open) >= self.max_concurrent:
            self.metrics.inc("noise_band_max_concurrent")
            return

        self._emit_proposal(sym, side, band, now)

    def _emit_proposal(self, sym: str, direction: int, band: float, now: float) -> None:
        price = list(self._closes[sym])[-1]
        sl_pct = max(band * self.sl_mult / price * 100.0, 0.05)
        tp_pct = max(band * self.tp_mult / price * 100.0, 0.10)
        self._armed[sym] = float(direction)
        self.emit(make_event(
            EventType.ORDER_PROPOSAL, self.name,
            payload={
                "symbol": sym,
                "direction": float(direction),
                "qty": 1.0,
                "position_pct": self.position_pct,
                "sl": 0.0,
                "tp": 0.0,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "passes_ev": True,
                "execute": False,
                "source": "noise_band",
                "band": band,
            },
            symbol_impact=[sym],
            timestamp=now,
        ))
        self.metrics.inc("noise_band_proposal", symbol=sym)

    def on_fill(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        sym = p.get("symbol")
        if sym is None:
            return
        if p.get("closed", False):
            self._open.pop(sym, None)
            return
        # Only claim fills whose direction matches OUR last armed proposal, so a
        # position opened by another strategy (release_drift, fusion) is never
        # "adopted" and force-closed by this one.
        direction = 1.0 if float(p.get("direction", 0.0)) > 0 else -1.0
        armed = self._armed.pop(sym, None)
        if armed is None or abs(armed - direction) > 1e-9:
            return
        self._open[sym] = {
            "entry_ts": self.clock.now(),
            "direction": direction,
        }
