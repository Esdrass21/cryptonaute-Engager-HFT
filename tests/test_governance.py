# tests/test_governance.py
"""Governance tests for the signed risk-veto (architecture S5).

The risk layer signs every order decision; the strategy FSM and the execution layer
(L5) must independently *verify* the signature and refuse to act on anything unverifiable.
These tests prove the fail-closed property with real components, not mocks.

Phase 2: uses KeyManager-backed signing (no raw-secret string).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit import AuditLog
from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent, make_event
from core.keys import EnvKeyProvider, KeyManager
from core.metrics import Metrics
from execution.broker_adapter import PaperBroker
from execution.order_manager import OrderManager
from risk_management.signing import sign_veto
from risk_management.veto_service import RiskVetoService
from strategy_engine.state_machine import StrategyOrchestrator


@pytest.fixture
def km(tmp_path):
    """KeyManager with a test key."""
    import base64
    import os
    os.environ["RISK_SIGNING_KEY"] = base64.b64encode(b"test-key-32bytes-long-value-1234567890!!").decode()
    return KeyManager(EnvKeyProvider(), mode="paper")


def _veto_payload(km, symbol, decision, proposal):
    signed_payload, sig = sign_veto(km, proposal, decision, "", 0.0)
    return make_event(
        EventType.RISK_VETO, "risk.veto",
        payload={"symbol": symbol, "decision": decision, "reasons": [],
                 "signature": sig, "signed_payload": signed_payload,
                 "proposal_ref": "p1", "proposal": dict(proposal)},
        symbol_impact=[symbol], timestamp=0.0,
    )


def test_valid_signed_veto_passes_through_execution(km):
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics, key_manager=km)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics, key_manager=km)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker, key_manager=km)
    veto.start()
    fsm.start()
    om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))
    rej = []
    bus.subscribe(EventType.ORDER_REJECT, lambda e: rej.append(e))

    # Set a price so the book-aware broker can fill
    broker.update_price("EURUSD", 1.10)

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    # strategy proposes -> veto (signed allow) -> fsm forwards -> execution verifies + executes
    fsm._pending_veto["EURUSD"] = proposal
    bus.publish(_veto_payload(km, "EURUSD", "allow", proposal))
    bus.run_replay()

    assert fills or broker.get_account_state().get("open_positions"), \
        "a valid signed veto should let the order execute"
    assert rej == [], "valid signed veto must not be rejected"


def test_tampered_veto_is_rejected_fail_closed(km):
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics, key_manager=km)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics, key_manager=km)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker, key_manager=km)
    veto.start()
    fsm.start()
    om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    good = _veto_payload(km, "EURUSD", "allow", proposal)
    # FORGERY: flip the decision after signing
    forged = make_event(
        EventType.RISK_VETO, "attacker",
        payload={**good.payload, "decision": "deny", "symbol": "EURUSD"},
        symbol_impact=["EURUSD"], timestamp=0.0,
    )
    # Attacker cannot reproduce the signature without the key:
    forged.payload["signature"] = {"key_id": "env:default", "digest_b64": "deadbeefdeadbeef", "v": 1}

    fsm._pending_veto["EURUSD"] = proposal
    bus.publish(forged)
    bus.run_replay()

    # FSM drops the forged veto (cannot verify signature) before any order is placed.
    assert metrics.snapshot().get("veto_signature_rejected", 0) > 0, \
        "forged veto must be rejected at the FSM"
    assert not broker.get_account_state().get("open_positions"), \
        "forged veto must NOT open a position (fail-closed)"


def test_missing_signature_veto_blocked(km):
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics, key_manager=km)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics, key_manager=km)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker, key_manager=km)
    veto.start()
    fsm.start()
    om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    unsigned = make_event(
        EventType.RISK_VETO, "risk.veto",
        payload={"symbol": "EURUSD", "decision": "allow", "reasons": [],
                 "signature": None, "proposal_ref": "p1", "proposal": dict(proposal)},
        symbol_impact=["EURUSD"], timestamp=0.0,
    )

    fsm._pending_veto["EURUSD"] = proposal
    bus.publish(unsigned)
    bus.run_replay()

    # FSM drops the unverifiable veto (fail-closed) before it ever reaches execution.
    assert metrics.snapshot().get("veto_signature_rejected", 0) > 0, \
        "unsigned veto must be rejected at the FSM"
    assert not broker.get_account_state().get("open_positions"), \
        "unsigned veto must NOT open a position"


def test_veto_verify_helper_detects_tamper():
    """Verify that sign_veto / verify_veto detect tampering."""
    import base64
    import os
    os.environ["RISK_SIGNING_KEY"] = base64.b64encode(b"test-key-32bytes-long-value-1234567890!!").decode()
    km = KeyManager(EnvKeyProvider(), mode="paper")
    proposal = {"symbol": "XAUUSD", "position_pct": 1.0, "proposal_id": "p1",
                "side": "buy", "qty": 1.0}
    signed_payload, sig = sign_veto(km, proposal, "allow", "", 0.0)
    from risk_management.signing import verify_veto
    assert verify_veto(km, proposal, "allow", sig, "", 0.0)
    # Tamper with decision
    assert not verify_veto(km, proposal, "deny", sig, "", 0.0)
