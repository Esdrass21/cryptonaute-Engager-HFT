# strategy_engine/tsmom.py
"""
Live port of the strategy-lab's top-ranked strategy: the equal-weight
time-series-momentum blend (Hurst, Ooi & Pedersen 2017, SSRN 2993026;
Moskowitz, Ooi & Pedersen 2012, SSRN 2089463).

Thesis: go with the sign of the trailing return over several horizons
(1d / 5d / 30d of M5 bars), size by ex-ante volatility so each position
contributes a constant amount of risk, and rebalance once per day. The lab
showed this is the only family that stays net-positive in most symbols AFTER
paying 5-10 bps of spread on this market sample.

Signal is computed only on closed candles; the position can change at most
once per `rebalance_s`. Everything flows through the standard path:
ORDER_PROPOSAL (execute=False, passes_ev=True) -> FSM -> veto -> execution.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent, make_event
from core.metrics import Metrics

log = logging.getLogger("strategy.tsmom")

BARS_PER_YEAR = 288 * 365  # M5 grid, 24/7 markets


class TSMOMStrategy(Component):
    def __init__(self, bus: MessageBus, cfg: Config, clock: Clock, metrics: Metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="strategy.tsmom")
        sec = "strategy.tsmom"
        self.lookbacks: list[int] = list(cfg.get(f"{sec}.lookbacks", [288, 1440, 8640]))
        self.rebalance_s = float(cfg.get(f"{sec}.rebalance_s", 86400.0))
        self.vol_win = int(cfg.get(f"{sec}.vol_win", 288))
        self.vol_target = float(cfg.get(f"{sec}.vol_target", 0.5))
        self.max_leverage = float(cfg.get(f"{sec}.max_leverage", 1.0))
        self.min_bars = int(cfg.get(f"{sec}.min_bars", max(self.lookbacks) + 1))
        self.enabled = bool(cfg.get(f"{sec}.enabled", True))
        self.symbols_allowed = set(cfg.get(f"{sec}.symbols", cfg.get("symbols", ["EURUSD"])))
        self._closes: dict[str, deque] = {}
        self._open: dict[str, dict] = {}
        self._armed: dict[str, float] = {}  # symbol -> direction of last proposal
        self._last_rebalance: dict[str, float] = {}

    # ---- signal ---------------------------------------------------------
    def _target_position(self, sym: str) -> float:
        """Signed, vol-scaled target in [-max_leverage, +max_leverage]."""
        closes = np.asarray(list(self._closes[sym]), dtype=float)
        rets = closes[1:] / closes[:-1] - 1.0
        if len(closes) < self.min_bars:
            return 0.0
        legs = []
        for lb in self.lookbacks:
            if len(closes) > lb:
                mom = closes[-1] / closes[-1 - lb] - 1.0
                legs.append(1.0 if mom > 0 else -1.0)
        if not legs:
            return 0.0
        sign = float(np.mean(legs))
        if sign == 0.0:
            return 0.0
        sigma = float(np.std(rets[-self.vol_win:])) * math.sqrt(BARS_PER_YEAR)
        if not math.isfinite(sigma) or sigma <= 0.0:
            return 0.0
        scale = min(self.vol_target / sigma, self.max_leverage)
        return float(np.sign(sign) * scale)

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        log.info(
            "TSMOMStrategy online (lookbacks=%s rebalance_s=%.0f vol_target=%.2f min_bars=%d)",
            self.lookbacks, self.rebalance_s, self.vol_target, self.min_bars,
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
        maxlen = max(self.min_bars, self.vol_win) + 2
        self._closes.setdefault(sym, deque(maxlen=maxlen)).append(float(p["close"]))
        if len(self._closes[sym]) < self.min_bars:
            return

        now = self.clock.now()
        if now - self._last_rebalance.get(sym, -math.inf) < self.rebalance_s:
            return

        target = self._target_position(sym)
        self._last_rebalance[sym] = now
        self._rebalance(sym, target, now)

    def _rebalance(self, sym: str, target: float, now: float) -> None:
        entry = self._open.get(sym)
        if entry is not None:
            same = (entry["direction"] > 0) == (target > 0) and target != 0.0
            if not same:
                self.emit(make_event(
                    EventType.ORDER_EXIT, self.name,
                    payload={"symbol": sym, "reason": "tsmom_flip", "execute": True},
                    symbol_impact=[sym], timestamp=now,
                ))
                self.metrics.inc("tsmom_exit", symbol=sym)
                self._open.pop(sym, None)
                self._armed.pop(sym, None)
        if target != 0.0 and sym not in self._open:
            self._emit_proposal(sym, target, now)

    def _emit_proposal(self, sym: str, target: float, now: float) -> None:
        direction = 1.0 if target > 0 else -1.0
        self._armed[sym] = direction
        self.emit(make_event(
            EventType.ORDER_PROPOSAL, self.name,
            payload={
                "symbol": sym,
                "direction": direction,
                "qty": 1.0,
                "position_pct": abs(target),
                "sl": 0.0,
                "tp": 0.0,
                "passes_ev": True,
                "execute": False,
                "source": "tsmom",
                "target": target,
            },
            symbol_impact=[sym],
            timestamp=now,
        ))
        self.metrics.inc("tsmom_proposal", symbol=sym)

    def on_fill(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        sym = p.get("symbol")
        if sym is None:
            return
        if p.get("closed", False):
            self._open.pop(sym, None)
            return
        # Only claim fills whose direction matches OUR last armed proposal, so a
        # position opened by another strategy is never adopted or force-closed.
        direction = 1.0 if float(p.get("direction", 0.0)) > 0 else -1.0
        armed = self._armed.pop(sym, None)
        if armed is None or abs(armed - direction) > 1e-9:
            return
        self._open[sym] = {
            "entry_ts": self.clock.now(),
            "direction": direction,
        }
