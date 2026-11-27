# risk_management/circuit_breaker.py
"""
Circuit breaker (Layer 6 -- S7.1).

Hard daily-drawdown cut-out. Trips once it is breached and requires a manual reset -- this is
a governance control, not something the AI or any other layer can silently clear.

Phase 2: reset is gated by a two-person rule (audit-logged).
"""
from __future__ import annotations

import logging

from core.component import Component
from core.events import EventType, make_event
from risk_management.two_person import (
    ACTION_CIRCUIT_BREAKER_RESET,
    Approval,
    TwoPersonRule,
)

log = logging.getLogger("risk.circuit")


class CircuitBreaker(Component):
    def __init__(self, bus, cfg, clock, metrics,
                 two_person: TwoPersonRule | None = None):
        super().__init__(bus, cfg, clock, metrics, name="risk.circuit")
        self.enabled = bool(cfg.get("risk.circuit_breaker.enabled", True))
        self.max_dd_pct = float(cfg.get("risk.max_daily_drawdown_pct", 2.0))
        self.two_person = two_person
        self._tripped = False

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            return
        self.bus.subscribe(EventType.METRIC, self.on_metric)
        log.info("Circuit breaker online (limit=%.2f%%)", self.max_dd_pct)

    def on_metric(self, ev) -> None:
        if ev.payload.get("name") == "account_drawdown_pct" and not self._tripped:
            val = float(ev.payload.get("value", 0.0))
            if val >= self.max_dd_pct:
                self._tripped = True
                self.emit(make_event(
                    EventType.RISK_BREACH, "risk.circuit",
                    payload={"reason": "daily-drawdown-limit", "value": val,
                             "state": "tripped"},
                    timestamp=self.clock.now(),
                ))
                self.metrics.inc("circuit_breaker_trips")
                log.critical("CIRCUIT BREAKER TRIPPED at %.2f%% drawdown -- trading halted", val)

    def reset(self, actor: str = "admin", approvers: tuple[str, ...] = ()) -> None:
        """Human re-arm. Requires two-person approval when configured."""
        if self.two_person is not None:
            self.two_person.check(Approval(
                actor=actor,
                approvers=approvers,
                action=ACTION_CIRCUIT_BREAKER_RESET,
                payload={"state_before": "tripped" if self._tripped else "armed"},
            ))
        self._tripped = False
        self.emit(make_event(
            EventType.RISK_BREACH, "risk.circuit",
            payload={"state": "armed"},
            timestamp=self.clock.now(),
        ))
        log.info("Circuit breaker reset (armed) by %s", actor)

    @property
    def tripped(self) -> bool:
        return self._tripped
