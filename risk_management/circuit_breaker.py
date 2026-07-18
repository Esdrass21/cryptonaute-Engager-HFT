"""Circuit breaker (Layer 6 — §7.1).

Hard daily-drawdown cut-out. Trips once it is breached and requires a manual reset — this is
a governance control, not something the AI or any other layer can silently clear.
"""

from __future__ import annotations

import logging

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

log = logging.getLogger("risk.circuit")


class CircuitBreaker(Component):
    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="risk.circuit")
        self.enabled = bool(cfg.get("risk.circuit_breaker.enabled", True))
        self.max_dd_pct = float(cfg.get("risk.max_daily_drawdown_pct", 2.0))
        self.tripped = False

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            return
        self.bus.subscribe(EventType.METRIC, self.on_metric)
        log.info("Circuit breaker online (limit=%.2f%%)", self.max_dd_pct)

    def on_metric(self, ev) -> None:
        if ev.payload.get("name") == "account_drawdown_pct" and not self.tripped:
            val = float(ev.payload.get("value", 0.0))
            if val >= self.max_dd_pct:
                self.tripped = True
                self.emit(make_event(
                    EventType.RISK_BREACH, "risk.circuit",
                    payload={"reason": "daily-drawdown-limit", "value": val},
                    timestamp=self.clock.now(),
                ))
                self.metrics.inc("circuit_breaker_trips")
                log.critical("CIRCUIT BREAKER TRIPPED at %.2f%% drawdown — trading halted", val)

    def reset(self) -> None:
        self.tripped = False
