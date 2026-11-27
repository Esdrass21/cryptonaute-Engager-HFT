"""Chaos: broker stops responding mid-session. System must fail closed, retry,
and never place an unverified order."""
import pytest

from core.bus import MessageBus
from core.event_log import InMemoryEventLog
from core.events import make_event
from core.keys import FileKeyProvider, KeyManager
from core.schema_registry import register_default_schemas
from execution.order_manager import OrderManager
from risk_management.signing import sign_veto

pytestmark = pytest.mark.chaos


class _Clock:
    def now(self): return 1_000_000.0


class _Metrics:
    def __init__(self): self.c = {}
    def inc(self, k, n=1, **__): self.c[k] = self.c.get(k, 0) + n
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def snapshot(self): return dict(self.c)


class _DeadBroker:
    """Every send fails with a plausible reason."""
    def connect(self): return True
    def update_price(self, *_, **__): pass
    def get_account_state(self): return {"balance": 10_000.0, "drawdown_pct": 0.0}
    def get_open_positions(self): return []
    def send_market_order(self, *_, **__): return {"ok": False, "reason": "timeout"}
    def close_position(self, *_, **__): return {"ok": False}


def test_broker_outage_emits_reject_and_stays_alive(cfg, keyring, monkeypatch):
    register_default_schemas()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    km = KeyManager(FileKeyProvider(keyring), mode="paper")
    bus = MessageBus(event_log=InMemoryEventLog())
    om = OrderManager(bus, cfg, _Clock(), _Metrics(), broker=_DeadBroker(), key_manager=km)
    om.start()

    rejects = []
    bus.subscribe("order_reject", lambda e: rejects.append(e))

    proposal = {"proposal_id": "p1", "symbol": "EURUSD", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1_000_000.0)

    bus.publish(make_event(
        "order_proposal", "strategy.fsm",
        payload={
            "symbol": "EURUSD", "direction": 1.0, "qty": 1.0,
            "veto": "allow", "execute": True,
            "veto_signature": sig, "veto_signed_payload": signed,
            "veto_decision": "allow",
        },
        timestamp=1_000_000.0,
    ))

    assert rejects, "broker outage must produce ORDER_REJECT"
    assert om._running is True, "order manager must not crash on broker failure"
