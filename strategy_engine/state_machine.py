"""Per-symbol trade state machine (Layer 4 — §5).

Uses the ``transitions`` library. A small deterministic FSM enforces that a position can
only be opened after a composite signal crosses threshold AND an EV proposal passes AND the
risk veto (Layer 6) returns allow. This is the deterministic "what decides" core; the LLM
is never in this path.

States:  idle -> signal_detected -> confirmed -> position_open -> managing -> closed -> idle
"""

from __future__ import annotations

import logging

from core.component import Component
from core.events import EventType, NormalizedEvent, make_event
from core.keys import KeyManager
from risk_management.signing import verify_any

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
        self.last_proposal: dict | None = None
        self.open_side: str | None = None
        self.entry_price: float | None = None
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

    # Public API kept stable whether or not the ``transitions`` library is installed.
    def detect(self):
        if _HAVE_TRANSITIONS:
            self.machine.detect()
        else:
            self._manual_detect()

    def confirm(self):
        if _HAVE_TRANSITIONS:
            self.machine.confirm()
        else:
            self._manual_confirm()

    def open(self):  # noqa: A002 - shadows builtin, intentional FSM verb
        if _HAVE_TRANSITIONS:
            self.machine.open()
        else:
            self._manual_open()

    def manage(self):
        if _HAVE_TRANSITIONS:
            self.machine.manage()
        else:
            self._manual_manage()

    def close(self):
        if _HAVE_TRANSITIONS:
            self.machine.close()
        else:
            self._manual_close()

    def reset(self):
        if _HAVE_TRANSITIONS:
            self.machine.reset()
        else:
            self._manual_reset()


class StrategyOrchestrator(Component):
    """Drives per-symbol FSMs from ORDER_PROPOSAL + RISK_VETO + ORDER_FILL events."""

    def __init__(self, bus, cfg, clock, metrics, key_manager: KeyManager | None = None):
        super().__init__(bus, cfg, clock, metrics, name="strategy.fsm")
        self._fsms: dict[str, SymbolFSM] = {s: SymbolFSM(s) for s in cfg.get("symbols", ["EURUSD"])}
        self.threshold = float(cfg.get("strategy.fusion.threshold", 0.55))
        self.exit_flip = float(cfg.get("strategy.exit_flip_conviction", 0.15))
        self._open_side: dict[str, int] = {}
        self._pending_veto: dict[str, dict] = {}
        # Phase 2: key manager -- no dev fallback
        if key_manager is not None:
            self.km = key_manager
        else:
            from core.keys import build_key_provider
            self.km = KeyManager(
                build_key_provider(cfg.get("core.keys", None)),
                mode=cfg.get("mode", "paper"),
            )

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self.on_proposal)
        self.bus.subscribe(EventType.RISK_VETO, self.on_veto)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        self.bus.subscribe(EventType.COMPOSITE_SIGNAL, self.on_composite)
        log.info("Strategy FSM orchestrator online (symbols=%s, transitions=%s)",
                 list(self._fsms), _HAVE_TRANSITIONS)

    def on_composite(self, ev: NormalizedEvent) -> None:
        """Drive exits: if a position is open and the composite flips to the opposite
        side, request a close (architecture §4: signal reversal exits the trade)."""
        sym = ev.payload.get("symbol")
        if sym is None or sym not in self._open_side:
            return
        side = self._open_side[sym]
        comp = float(ev.payload.get("composite", 0.0))
        # opposite sign with at least modest conviction (configurable; rides trends when high)
        if side != 0 and ((side > 0 and comp < -self.exit_flip) or
                          (side < 0 and comp > self.exit_flip)):
            self.emit(make_event(
                EventType.ORDER_EXIT, "strategy.fsm",
                payload={"symbol": sym, "reason": "composite_flip", "execute": True},
                symbol_impact=[sym], timestamp=self.clock.now(),
            ))
            self.metrics.inc("exit_signal_flip")

    def on_proposal(self, ev: NormalizedEvent) -> None:
        sym = ev.payload["symbol"]
        fsm = self._fsms.get(sym)
        if fsm is None:
            return
        # Ignore our own forwarded execution orders (execute=True) — those are handled by
        # the execution layer, not re-fed into the FSM.
        if ev.payload.get("execute"):
            return
        if not ev.payload.get("passes_ev", False):
            return
        # Edge-triggered: only arm the machine from idle, and never while a veto is already
        # pending for this symbol. SignalFusion emits a proposal on EVERY hot-composite tick,
        # so without this guard we'd re-arm / re-submit to the veto every tick (a re-entrancy
        # storm that wedges the FSM in `confirmed` and prevents clean entries/exits).
        if fsm.state != "idle" or sym in self._pending_veto:
            return
        fsm.detect()
        fsm.confirm()
        self._pending_veto[sym] = ev.payload  # wait for risk veto before opening

    def on_veto(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        if sym not in self._pending_veto:
            return
        # Architecture S5: execution only acts on a *verified, signed* risk permission.
        # Reject any veto whose signature does not validate against the signing key.
        if not verify_any(self.km, ev.payload):
            log.error("RISK VETO for %s FAILED signature verification -- DROPPED (tamper/forgery)", sym)
            self.metrics.inc("veto_signature_rejected")
            self._pending_veto.pop(sym, None)
            fsm = self._fsms.get(sym)
            if fsm is not None:
                fsm.reset()
            return
        decision = ev.payload.get("decision", "")
        proposal = self._pending_veto.pop(sym)
        fsm = self._fsms[sym]
        if decision == "allow":
            fsm.open()
            fsm.manage()
            self._open_side[sym] = 1 if proposal.get("direction", 1.0) > 0 else -1
            self.emit(make_event(
                EventType.ORDER_PROPOSAL, "strategy.fsm",
                payload={
                    **proposal,
                    "veto": "allow",
                    "execute": True,
                    # --- signature bundle forwarded verbatim ---
                    "veto_signature":       ev.payload.get("signature"),         # dict (v2)
                    "veto_signed_payload":  ev.payload.get("signed_payload"),    # dict (v2)
                    "veto_signature_v1":    ev.payload.get("signature_v1"),      # str  (legacy)
                    "veto_proposal":        ev.payload.get("proposal"),
                    "veto_decision":        ev.payload.get("decision"),
                },
                symbol_impact=[sym], timestamp=self.clock.now(),
            ))
            self.metrics.inc("order_forwarded_to_execution")
        else:
            fsm.reset()
            self.metrics.inc("order_blocked_by_risk")

    def on_fill(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        fsm = self._fsms.get(sym)
        if fsm is None:
            return
        if ev.payload.get("closed", False):  # exit fill
            self._open_side.pop(sym, None)
            fsm.close()
            fsm.reset()
            self.metrics.inc("exit_filled")
        else:  # entry fill — already tracked in on_veto; nothing to do here
            pass

    def state_of(self, sym: str) -> str:
        fsm = self._fsms.get(sym)
        return fsm.state if fsm else "unknown"
