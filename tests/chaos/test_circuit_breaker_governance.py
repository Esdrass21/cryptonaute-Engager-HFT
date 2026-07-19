"""Chaos: attempt to reset the breaker without two-person approval.
Must be denied and audited."""
import json
from pathlib import Path
import pytest

from core.audit import AuditLog
from core.bus import MessageBus
from core.config import Config
from core.event_log import InMemoryEventLog
from risk_management.circuit_breaker import CircuitBreaker
from risk_management.two_person import (
    TwoPersonError, TwoPersonRule,
)

pytestmark = pytest.mark.chaos

# Actions allowed by the two-person rule
_ACTIONS = {"circuit_breaker.reset", "live.activate", "risk.config.change", "key.rotation"}


class _Clock:
    def now(self): return 1.0


class _Metrics:
    def inc(self, *_, **__): pass
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def snapshot(self): return {}


def test_self_approval_is_denied_and_audited(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    rule = TwoPersonRule(audit, _ACTIONS, min_approvers=1)

    cfg = Config.load()
    cb = CircuitBreaker(MessageBus(event_log=InMemoryEventLog()), cfg,
                        _Clock(), _Metrics(), two_person=rule)
    cb._tripped = True

    with pytest.raises(TwoPersonError):
        cb.reset(actor="alice", approvers=("alice",))

    assert cb._tripped is True
    line = (tmp_path / "audit.jsonl").read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["outcome"] == "denied"
    assert "actor cannot approve their own action" in rec["details"]["reasons"][0]
