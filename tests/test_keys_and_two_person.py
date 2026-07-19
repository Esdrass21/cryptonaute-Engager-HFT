# tests/test_keys_and_two_person.py
"""Tests for Phase 2: key management and two-person rule."""
import base64
import json
import os
from pathlib import Path
import pytest

from core.keys import KeyManager, EnvKeyProvider, FileKeyProvider, KeyProviderError, build_key_provider
from core.audit import AuditLog
from risk_management.two_person import (
    TwoPersonRule, Approval, TwoPersonError,
    ACTION_CIRCUIT_BREAKER_RESET, DEFAULT_ALLOWED_ACTIONS,
)


# -- key manager ------------------------------------------------------------ #

def _set_env_key(monkeypatch, b: bytes = b"x" * 32):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b).decode())
    return EnvKeyProvider()


def test_sign_and_verify_roundtrip(monkeypatch):
    km = KeyManager(_set_env_key(monkeypatch), mode="paper")
    sig = km.sign({"a": 1, "b": 2})
    assert km.verify({"a": 1, "b": 2}, sig)
    assert not km.verify({"a": 1, "b": 3}, sig)


def test_signature_is_deterministic(monkeypatch):
    km = KeyManager(_set_env_key(monkeypatch), mode="paper")
    assert km.sign({"a": 1}).digest_b64 == km.sign({"a": 1}).digest_b64


def test_signature_carries_key_id(monkeypatch):
    km = KeyManager(_set_env_key(monkeypatch), mode="paper")
    sig = km.sign({"x": 1})
    assert sig.key_id == "env:default"


def test_env_provider_refused_in_live(monkeypatch):
    monkeypatch.delenv("RISK_ALLOW_ENV_KEYS", raising=False)
    with pytest.raises(KeyProviderError):
        KeyManager(_set_env_key(monkeypatch), mode="live")


def test_env_provider_allowed_in_live_with_override(monkeypatch):
    monkeypatch.setenv("RISK_ALLOW_ENV_KEYS", "1")
    km = KeyManager(_set_env_key(monkeypatch), mode="live")
    assert km.current_key_id()


def test_env_provider_pads_short_raw_key(monkeypatch):
    """Dev convenience: short non-base64 keys are padded to 32 bytes."""
    monkeypatch.setenv("RISK_SIGNING_KEY", "short-key")
    provider = EnvKeyProvider()
    assert len(provider.current().key_bytes) >= 32


def test_file_provider_rotation_verifies_old_signatures(tmp_path: Path):
    keyring = tmp_path / "keyring.json"
    v1 = base64.b64encode(b"1" * 32).decode()
    v2 = base64.b64encode(b"2" * 32).decode()
    keyring.write_text(json.dumps({"current": "v1", "keys": {"v1": v1}}))

    km = KeyManager(FileKeyProvider(keyring), mode="paper")
    sig_v1 = km.sign({"x": 1})
    assert km.verify({"x": 1}, sig_v1)

    # rotate: add v2 as current, keep v1
    keyring.write_text(json.dumps({"current": "v2", "keys": {"v1": v1, "v2": v2}}))
    km.reload()
    assert km.current_key_id() == "v2"

    # old signature still verifies because key_id travels with it
    assert km.verify({"x": 1}, sig_v1)

    # new signatures use v2
    sig_v2 = km.sign({"x": 1})
    assert sig_v2.key_id == "v2"


def test_build_key_provider_env(monkeypatch):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"x" * 32).decode())
    provider = build_key_provider({"provider": "env"})
    assert isinstance(provider, EnvKeyProvider)


def test_build_key_provider_file(tmp_path: Path):
    keyring = tmp_path / "keyring.json"
    v1 = base64.b64encode(b"x" * 32).decode()
    keyring.write_text(json.dumps({"current": "v1", "keys": {"v1": v1}}))
    provider = build_key_provider({"provider": "file", "keyring_path": str(keyring)})
    assert isinstance(provider, FileKeyProvider)


# -- two-person ------------------------------------------------------------- #

def test_two_person_accepts_distinct_approver(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, DEFAULT_ALLOWED_ACTIONS, min_approvers=1)
    rule.check(Approval(actor="alice", approvers=("bob",),
                        action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))
    lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[-1])["outcome"] == "approved"


def test_two_person_rejects_self_approval(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, DEFAULT_ALLOWED_ACTIONS, min_approvers=1)
    with pytest.raises(TwoPersonError):
        rule.check(Approval(actor="alice", approvers=("alice",),
                            action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))
    lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[-1])["outcome"] == "denied"


def test_two_person_rejects_unknown_action(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, allowed_actions=("known",), min_approvers=1)
    with pytest.raises(TwoPersonError):
        rule.check(Approval(actor="a", approvers=("b",), action="unknown", payload={}))


def test_two_person_requires_min_two_approvers(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, DEFAULT_ALLOWED_ACTIONS, min_approvers=2)
    with pytest.raises(TwoPersonError):
        rule.check(Approval(actor="a", approvers=("b",),
                            action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))
    rule.check(Approval(actor="a", approvers=("b", "c"),
                        action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))


def test_two_person_denies_empty_actor(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, DEFAULT_ALLOWED_ACTIONS, min_approvers=1)
    with pytest.raises(TwoPersonError):
        rule.check(Approval(actor="", approvers=("bob",),
                            action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))


def test_two_person_removes_duplicate_approvers(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, DEFAULT_ALLOWED_ACTIONS, min_approvers=2)
    # "bob","bob","c" -> 2 unique -> passes
    rule.check(Approval(actor="alice", approvers=("bob", "bob", "c"),
                        action=ACTION_CIRCUIT_BREAKER_RESET, payload={}))
