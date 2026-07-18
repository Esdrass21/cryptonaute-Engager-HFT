"""Per-symbol trade state machine (Layer 4 — §5).

Uses the ``transitions`` library. A small deterministic FSM enforces that a position can
only be opened after a composite signal crosses threshold AND an EV proposal passes AND the
risk veto (Layer 6) returns allow. This is the deterministic "what decides" core; the LLM
is never in this path.

States:  idle -> signal_detected -> confirmed -> position_open -> managing -> closed -> idle
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

from risk_management.veto_service import RiskVetoService

log = logging.getLogger("strategy.fsm")

try:
    from transitions import Machine

    _HAVE_TRANSITIONS = True
except Exception:  # noqa: BLE001
    Machine = None  # type: ignore
    _HAVE_TRANSITIONS = False


class SymbolFSM:
    states = ["idle", "signal_detected", "confirmed", "position_open", "managing", "closed"]

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_proposal: Optional[dict] = None
        self.open_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        if _HAVE_TRANSITIONS:
            self.machine = Machine(self, states=self.states, initial="idle", auto_transitions=False)
            self.machine.add_transition("detect", "idle", "signal_detected")
            self.machine.add_transition("confirm", "signal_detected", "confirmed")
            self.machine.add_transition("open", ["confirmed", "idle"], "position_open")
            self.machine.add_transition("manage", "position_open", "managing")
            self.machine.add_transition("close", ["managing", "position_open"], "closed")
            self.machine.add_transition("reset", "*", "idle")
        else:  # pragma: no cover - fallback if transitions missing
            self.state = "idle"

    def _manual_detect(self):
        if not _HAVE_TRANSITIONS:
            self.state = "signal_detected"

    def _manual_confirm(self):
        if not _HAVE_TRANSITIONS:
            self.state = "confirmed"

    def _manual_open(self):
        if not _HAVE_TRANSITIONS:
            self.state = "position_open"

    def _manual_manage(self):
        if not _HAVE_TRANSITIONS:
            self.state = "managing"

    def _manual_close(self):
        if not _HAVE_TRANSITIONS:
            self.state = "closed"

    def _manual_reset(self):
        if not _HAVE_TRANSITIONS:
            self.state = "idle"


class StrategyOrchestrator(Component):
    """Drives per-symbol FSMs from ORDER_PROPOSAL + RISK_VETO + ORDER_FILL events."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="strategy.fsm")
        self._fsms: Dict[str, SymbolFSM] = {s: SymbolFSM(s) for s in cfg.get("symbols", [])}
        self._pending_veto: Dict[str, dict] = {}
        self._risk_secret = cfg.secret("RISK_SIGNING_KEY", "cryptonaute-local-dev-key")

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self.on_proposal)
        self.bus.subscribe(EventType.RISK_VETO, self.on_veto)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        log.info("Strategy FSM orchestrator online (symbols=%s, transitions=%s)",
                 list(self._fsms), _HAVE_TRANSITIONS)

    def on_proposal(self, ev: NormalizedEvent) -> None:
        sym = ev.payload["symbol"]
        fsm = self._fsms.get(sym)
        if fsm is None:
            return
        if not ev.payload.get("passes_ev", False):
            return
        # Only act on a fresh signal when no position is open for this symbol.
        if fsm.state != "idle":
            return
        fsm.detect()
        fsm.confirm()
        self._pending_veto[sym] = ev.payload  # wait for risk veto before opening

    def on_veto(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        if sym not in self._pending_veto:
            return
        # Architecture §5: execution only acts on a *verified, signed* risk permission.
        # Reject any veto whose signature does not validate against the shared signing key.
        if not RiskVetoService.verify(self._risk_secret, ev.payload):
            log.error("RISK VETO for %s FAILED signature verification — DROPPED (tamper/forgery)", sym)
            self.metrics.inc("veto_signature_rejected")
            self._pending_veto.pop(sym, None)
            fsm = self._fsms.get(sym)
            if fsm is not None:
                fsm.reset()
            return
        decision = ev.payload.get("decision")
        proposal = self._pending_veto.pop(sym)
        fsm = self._fsms[sym]
        if decision == "allow":
            fsm.open()
            fsm.manage()
            self.emit(make_event(
                EventType.ORDER_PROPOSAL, "strategy.fsm",
                payload={**proposal, "veto": "allow", "execute": True,
                         "veto_signature": ev.payload.get("signature"),
                         "veto_proposal": ev.payload.get("proposal")},
                symbol_impact=[sym], timestamp=self.clock.now(),
            ))
            self.metrics.inc("order_forwarded_to_execution")
        else:
            fsm.reset()
            self.metrics.inc("order_blocked_by_risk")

    def on_fill(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        fsm = self._fsms.get(sym)
        if fsm is None or not ev.payload.get("closed", False):
            return
        fsm.close()
        fsm.reset()

    def state_of(self, sym: str) -> str:
        fsm = self._fsms.get(sym)
        return fsm.state if fsm else "unknown"
