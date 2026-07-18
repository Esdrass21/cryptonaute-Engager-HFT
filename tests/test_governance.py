"""Governance tests for the signed risk-veto (architecture §5).

The risk layer signs every order decision; the strategy FSM and the execution layer
(L5) must independently *verify* the signature and refuse to act on anything unverifiable.
These tests prove the fail-closed property with real components, not mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent, make_event
from core.metrics import Metrics

from risk_management.veto_service import RiskVetoService
from strategy_engine.state_machine import StrategyOrchestrator
from execution.order_manager import OrderManager
from execution.broker_adapter import PaperBroker


def _veto_payload(veto: RiskVetoService, symbol, decision, proposal):
    sig = veto._sign(proposal, decision)
    return make_event(
        EventType.RISK_VETO, "risk.veto",
        payload={"symbol": symbol, "decision": decision, "reasons": [],
                 "signature": sig, "proposal": dict(proposal)},
        symbol_impact=[symbol], timestamp=0.0,
    )


def test_valid_signed_veto_passes_through_execution():
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker)
    veto.start(); fsm.start(); om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))
    rej = []
    bus.subscribe(EventType.ORDER_REJECT, lambda e: rej.append(e))

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    # strategy proposes -> veto (signed allow) -> fsm forwards -> execution verifies + executes
    fsm._pending_veto["EURUSD"] = proposal
    bus.publish(_veto_payload(veto, "EURUSD", "allow", proposal))
    bus.run_replay()

    assert fills or broker.get_account_state().get("open_positions"), \
        "a valid signed veto should let the order execute"
    assert rej == [], "valid signed veto must not be rejected"


def test_tampered_veto_is_rejected_fail_closed():
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker)
    veto.start(); fsm.start(); om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    good = _veto_payload(veto, "EURUSD", "allow", proposal)
    # FORGERY: flip the decision after signing
    forged = make_event(
        EventType.RISK_VETO, "attacker",
        payload={**good.payload, "decision": "allow", "symbol": "EURUSD"},
        symbol_impact=["EURUSD"], timestamp=0.0,
    )
    forged.payload["decision"] = "allow"
    # ensure the forged payload's signature no longer matches the (tampered) content
    forged.payload["proposal"] = dict(proposal)
    # attacker cannot reproduce the signature without the secret:
    forged.payload["signature"] = "deadbeefdeadbeef"

    fsm._pending_veto["EURUSD"] = proposal
    bus.publish(forged)
    bus.run_replay()

    # FSM drops the forged veto (cannot verify signature) before any order is placed.
    assert metrics.snapshot().get("veto_signature_rejected", 0) > 0, \
        "forged veto must be rejected at the FSM"
    assert not broker.get_account_state().get("open_positions"), \
        "forged veto must NOT open a position (fail-closed)"


def test_missing_signature_veto_blocked():
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    veto = RiskVetoService(bus, cfg, clock, metrics)
    fsm = StrategyOrchestrator(bus, cfg, clock, metrics)
    broker = PaperBroker(cfg, clock)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker)
    veto.start(); fsm.start(); om.start()

    fills = []
    bus.subscribe(EventType.ORDER_FILL, lambda e: fills.append(e))

    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                "position_pct": 1.0, "passes_ev": True, "account_drawdown_pct": 0.0}
    unsigned = make_event(
        EventType.RISK_VETO, "risk.veto",
        payload={"symbol": "EURUSD", "decision": "allow", "reasons": [], "proposal": dict(proposal)},
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
    secret = "test-key"
    proposal = {"symbol": "XAUUSD", "position_pct": 1.0}
    good = {"decision": "allow", "proposal": proposal,
            "signature": RiskVetoService._sign_payload(secret, proposal, "allow")}
    assert RiskVetoService.verify(secret, good) is True
    bad = {**good, "proposal": {**proposal, "position_pct": 99.0}}  # tampered after signing
    assert RiskVetoService.verify(secret, bad) is False
    assert RiskVetoService.verify(secret, {"decision": "allow", "proposal": proposal}) is False
