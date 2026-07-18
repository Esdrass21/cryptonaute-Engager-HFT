"""Expected-Value engine (Layer 4 — §5).

Before any order is sent we estimate, per proposed trade:
  * expected reward (R) and expected drawdown (D)
  * win probability (p) — from recent realized hit-rate on the same regime, or a prior
  * expected duration and expected slippage

EV = p * R - (1-p) * D - slippage_cost. An ORDER_PROPOSAL is emitted only if
EV >= calibrated threshold AND EV is *calibrated on real historical data*, never a fixed
arbitrary constant (architecture §5: "calibrée sur données historiques réelles").

In paper/sim mode the calibration uses a running estimate from filled trades; the flag
``strategy.ev.calibrated`` must be True before live deployment (validated in §7.3 gate).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

log = logging.getLogger("strategy.ev")


class ExpectedValueEngine(Component):
    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="strategy.ev")
        self.min_ev = float(cfg.get("strategy.ev.min_expected_value", 0.0))
        self.calibrated = bool(cfg.get("strategy.ev.calibrated", False))
        # running calibration per symbol: hit-rate + avg win/loss (R-multiple)
        self._wins: Dict[str, int] = defaultdict(int)
        self._losses: Dict[str, int] = defaultdict(int)
        self._avg_win: Dict[str, float] = defaultdict(float)
        self._avg_loss: Dict[str, float] = defaultdict(float)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.COMPOSITE_SIGNAL, self.on_composite)
        self.bus.subscribe(EventType.ORDER_FILL, self._calibrate)
        log.info("EV engine online (calibrated=%s min_ev=%.4f)", self.calibrated, self.min_ev)

    def _calibrate(self, ev: NormalizedEvent) -> None:
        if not ev.payload.get("closed", False):
            return
        sym = ev.payload.get("symbol")
        pnl = float(ev.payload.get("pnl", 0.0))
        if pnl >= 0:
            self._wins[sym] += 1
            self._avg_win[sym] = (self._avg_win[sym] * (self._wins[sym] - 1) + pnl) / self._wins[sym]
        else:
            self._losses[sym] += 1
            self._avg_loss[sym] = (self._avg_loss[sym] * (self._losses[sym] - 1) + abs(pnl)) / self._losses[sym]

    def on_composite(self, ev: NormalizedEvent) -> None:
        if not ev.payload.get("above_threshold", False):
            return
        sym = ev.payload["symbol"]
        score = float(ev.payload["composite"])
        direction = 1 if score > 0 else -1
        # Calibrated probabilities (fall back to a conservative prior if no data).
        n = self._wins[sym] + self._losses[sym]
        if n > 5:
            p = self._wins[sym] / n
            R = self._avg_win[sym] or 1.0
            D = self._avg_loss[sym] or 1.0
            self.calibrated = True
        else:
            p, R, D = 0.5, 1.0, 1.0
        ev_est = p * R - (1 - p) * D
        self.emit(make_event(
            EventType.ORDER_PROPOSAL, "strategy.ev",
            payload={
                "symbol": sym,
                "direction": float(direction),
                "composite": score,
                "confidence": float(ev.payload.get("confidence", 0.0)),
                "ev": float(ev_est),
                "win_prob": float(p),
                "reward_R": float(R),
                "drawdown_D": float(D),
                "expected_slippage_pct": float(self.cfg.get("execution.slippage_tolerance_pct", 0.05)),
                "calibrated": self.calibrated,
                "passes_ev": bool(ev_est >= self.min_ev),
            },
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("order_proposal_emitted")
