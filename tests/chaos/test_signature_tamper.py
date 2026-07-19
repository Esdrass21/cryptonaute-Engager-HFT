"""Chaos: a signed veto is tampered with in flight. Both L4 and L5 must drop it."""
import pytest

from core.bus import MessageBus
from core.event_log import InMemoryEventLog
from core.events import make_event
from core.keys import KeyManager, FileKeyProvider
from core.schema_registry import register_default_schemas
from risk_management.signing import sign_veto
from strategy_engine.state_machine import StrategyOrchestrator

pytestmark = pytest.mark.chaos


class _Clock:
    def now(self): return 1_000_000.0


class _Metrics:
    def __init__(self): self.c = {}
    def inc(self, k, n=1, **__): self.c[k] = self.c.get(k, 0) + n
    def observe(self, *_, **__): pass
    def set(self, *_, **__): pass
    def snapshot(self): return dict(self.c)


def test_tampered_qty_is_rejected(cfg, keyring):
    register_default_schemas()
    km = KeyManager(FileKeyProvider(keyring), mode="paper")
    bus = MessageBus(event_log=InMemoryEventLog())
    fsm = StrategyOrchestrator(bus, cfg, _Clock(), _Metrics(), key_manager=km)
    fsm.start()

    # Arm the FSM.
    bus.publish(make_event(
        "order_proposal", "test",
        payload={"symbol": "EURUSD", "direction": 1.0, "qty": 1.0, "passes_ev": True},
        timestamp=1_000_000.0,
    ))

    proposal = {"proposal_id": "p1", "symbol": "EURUSD", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1_000_000.0)

    # Tamper the payload AFTER signing.
    tampered = dict(signed); tampered["qty"] = 99.0

    forwarded = []
    bus.subscribe("order_proposal", lambda e: forwarded.append(e))

    bus.publish(make_event(
        "risk_veto", "risk.veto",
        payload={
            "symbol": "EURUSD", "decision": "allow", "reasons": [],
            "signature": sig, "signed_payload": tampered,
            "proposal": proposal, "proposal_ref": "x",
        },
        timestamp=1_000_000.0,
    ))

    execs = [e for e in forwarded if e.payload.get("execute") is True]
    assert not execs, "tampered veto must not forward to execution"
    assert fsm.metrics.snapshot().get("veto_signature_rejected", 0) == 1
