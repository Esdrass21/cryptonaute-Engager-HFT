# tests/test_l4_l5_verification.py
"""Phase 2 L4/L5 migration tests.

Covers:
- verify_any accepts v2 dict signature over exact signed_payload
- verify_any accepts forwarded (veto_signature, veto_signed_payload) shape
- verify_any rejects missing/tampered dict signature
- L4 forwards the signature bundle verbatim to L5
- L5 rejects forged/unverified signatures
"""
import base64
import json
from pathlib import Path

import pytest

from core.bus import MessageBus
from core.config import Config
from core.event_log import InMemoryEventLog
from core.events import make_event
from core.keys import FileKeyProvider, KeyManager
from core.schema_registry import register_default_schemas
from execution.order_manager import OrderManager
from risk_management.signing import sign_veto, verify_any
from strategy_engine.state_machine import StrategyOrchestrator


class _Clock:
    def now(self): return 1_000_000.0
    def mode(self): return "virtual"


class _Metrics:
    def __init__(self): self.counters = {}
    def inc(self, k, n=1): self.counters[k] = self.counters.get(k, 0) + n
    def observe(self, *_, **__): pass
    def set(self, *_, **__): pass
    def snapshot(self): return dict(self.counters)


class _NullBroker:
    """Minimal stub so OrderManager doesn't need a real broker to construct."""
    def connect(self): return True
    def update_price(self, *_, **__): pass
    def get_account_state(self): return {"balance": 10_000.0, "drawdown_pct": 0.0}
    def send_market_order(self, *_, **__): return {"ok": False, "reason": "stub"}
    def close_position(self, *_, **__): return {"ok": False}


def _write_keyring(p: Path):
    p.write_text(json.dumps({
        "current": "v1",
        "keys": {"v1": base64.b64encode(b"1" * 32).decode()},
    }))


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.load()
    cfg.set("mode", "paper")
    cfg.set("core.keys.provider", "file")
    cfg.set("core.keys.keyring_path", str(tmp_path / "keyring.json"))
    cfg.set("symbols", ["EURUSD"])
    return cfg


# ---- verify_any -------------------------------------------------------- #


def test_verify_any_accepts_v2(tmp_path: Path):
    _write_keyring(tmp_path / "keyring.json")
    km = KeyManager(FileKeyProvider(tmp_path / "keyring.json"), mode="paper")
    proposal = {"proposal_id": "p1", "symbol": "EURUSD", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1_000_000.0)

    event_payload = {"signature": sig, "signed_payload": signed}
    assert verify_any(km, event_payload) is True

    # tamper the signed payload -> reject
    tampered = dict(signed)
    tampered["qty"] = 2.0
    assert verify_any(km, {"signature": sig, "signed_payload": tampered}) is False


def test_verify_any_forwarded_shape(tmp_path: Path):
    _write_keyring(tmp_path / "keyring.json")
    km = KeyManager(FileKeyProvider(tmp_path / "keyring.json"), mode="paper")
    proposal = {"proposal_id": "p1", "symbol": "EURUSD", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1_000_000.0)

    forwarded = {"veto_signature": sig, "veto_signed_payload": signed}
    assert verify_any(km, forwarded) is True


def test_verify_any_rejects_missing_dict_signature(tmp_path: Path):
    _write_keyring(tmp_path / "keyring.json")
    km = KeyManager(FileKeyProvider(tmp_path / "keyring.json"), mode="paper")
    assert verify_any(km, {}) is False


# ---- L4 forward carries signed payload -------------------------------- #


def test_l4_forwards_veto_signed_payload(tmp_path: Path):
    register_default_schemas()
    _write_keyring(tmp_path / "keyring.json")
    cfg = _cfg(tmp_path)
    km = KeyManager(FileKeyProvider(tmp_path / "keyring.json"), mode="paper")
    bus = MessageBus(event_log=InMemoryEventLog())

    fsm = StrategyOrchestrator(bus, cfg, _Clock(), _Metrics(), key_manager=km)
    fsm.start()

    # Arm the FSM with a proposal so on_veto accepts
    bus.publish(make_event(
        "order_proposal", "test",
        payload={"symbol": "EURUSD", "direction": 1.0, "qty": 1.0,
                 "passes_ev": True},
        timestamp=1_000_000.0,
    ))

    # Emit a properly signed veto
    proposal = {"proposal_id": "p1", "symbol": "EURUSD", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1_000_000.0)

    forwarded = []
    bus.subscribe("order_proposal", lambda e: forwarded.append(e))

    bus.publish(make_event(
        "risk_veto", "risk.veto",
        payload={
            "symbol": "EURUSD",
            "decision": "allow",
            "reasons": [],
            "signature": sig,
            "signed_payload": signed,
            "proposal": proposal,
            "proposal_ref": "x",
        },
        timestamp=1_000_000.0,
    ))

    # Find the forwarded execution order (execute=True)
    execs = [e for e in forwarded if e.payload.get("execute") is True]
    assert execs, "L4 did not forward an execution order"
    p = execs[0].payload
    assert p["veto_signed_payload"] == signed
    assert p["veto_signature"] == sig


# ---- L5 rejects unverified -------------------------------------------- #


def test_l5_rejects_forged_signature(tmp_path: Path):
    register_default_schemas()
    _write_keyring(tmp_path / "keyring.json")
    cfg = _cfg(tmp_path)
    km = KeyManager(FileKeyProvider(tmp_path / "keyring.json"), mode="paper")
    bus = MessageBus(event_log=InMemoryEventLog())
    om = OrderManager(bus, cfg, _Clock(), _Metrics(), broker=_NullBroker(), key_manager=km)
    om.start()

    rejects = []
    bus.subscribe("order_reject", lambda e: rejects.append(e))

    bus.publish(make_event(
        "order_proposal", "strategy.fsm",
        payload={
            "symbol": "EURUSD", "direction": 1.0, "qty": 1.0,
            "veto": "allow", "execute": True,
            # forged: dict signature with a made-up payload
            "veto_signature": {"key_id": "v1", "digest_b64": "AAAA", "v": 1},
            "veto_signed_payload": {"v": 1, "proposal_id": "p1",
                                    "symbol": "EURUSD", "side": "buy", "qty": 1.0,
                                    "decision": "allow", "reason": "", "ts": 1_000_000.0},
            "veto_decision": "allow",
        },
        timestamp=1_000_000.0,
    ))

    assert rejects, "L5 did not reject a forged signature"
    assert rejects[0].payload["reason"] == "veto-signature-unverified"
