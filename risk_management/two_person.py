# risk_management/two_person.py
"""
Two-person rule for sensitive operations.

A sensitive op is defined by:
  - action       : str   (e.g. "circuit_breaker.reset", "live.activate")
  - actor        : str   (the operator requesting it)
  - approvers    : list[str]  (at least 1, different from actor)
  - payload      : dict  (canonical details, part of the audit record)

Policy:
  - actor must be authenticated (non-empty string).
  - at least `min_approvers` approvers, none equal to actor, all distinct.
  - action must be in the allow-list (explicit opt-in per action).
  - Every decision -- approved or denied -- is written to the audit log.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.audit import AuditLog, AuditRecord, now

log = logging.getLogger("risk.two_person")


class TwoPersonError(RuntimeError):
    pass


@dataclass(frozen=True)
class Approval:
    actor: str
    approvers: tuple[str, ...]
    action: str
    payload: dict[str, Any]


class TwoPersonRule:
    def __init__(
        self,
        audit: AuditLog,
        allowed_actions: Iterable[str],
        min_approvers: int = 1,
    ) -> None:
        self.audit = audit
        self.allowed = set(allowed_actions)
        self.min_approvers = max(1, int(min_approvers))

    def check(self, approval: Approval) -> None:
        """Raise TwoPersonError if the approval is invalid. Never silent."""
        reasons: list[str] = []

        if not approval.actor:
            reasons.append("actor is empty")
        if approval.action not in self.allowed:
            reasons.append(f"action {approval.action!r} not in allow-list")
        unique_approvers = list(dict.fromkeys(approval.approvers))
        if len(unique_approvers) < self.min_approvers:
            reasons.append(
                f"needs >= {self.min_approvers} distinct approver(s), got {len(unique_approvers)}"
            )
        if approval.actor in unique_approvers:
            reasons.append("actor cannot approve their own action")

        if reasons:
            self._record(approval, "denied", {"reasons": reasons})
            raise TwoPersonError("; ".join(reasons))

        self._record(approval, "approved", {})
        log.info("two-person OK: action=%s actor=%s approvers=%s",
                 approval.action, approval.actor, unique_approvers)

    def _record(self, approval: Approval, outcome: str, extra: dict[str, Any]) -> None:
        try:
            self.audit.write(AuditRecord(
                ts=now(),
                action=approval.action,
                actor=approval.actor,
                approvers=list(approval.approvers),
                outcome=outcome,
                details={**approval.payload, **extra},
            ))
        except Exception as exc:  # noqa: BLE001
            log.exception("two-person audit write failed: %s", exc)
            raise TwoPersonError("audit log write failed") from exc


# Canonical action names
ACTION_CIRCUIT_BREAKER_RESET = "circuit_breaker.reset"
ACTION_LIVE_ACTIVATE         = "live.activate"
ACTION_RISK_LIMITS_CHANGE    = "risk.limits.change"
ACTION_KEY_ROTATION          = "keys.rotate"

DEFAULT_ALLOWED_ACTIONS = (
    ACTION_CIRCUIT_BREAKER_RESET,
    ACTION_LIVE_ACTIVATE,
    ACTION_RISK_LIMITS_CHANGE,
    ACTION_KEY_ROTATION,
)
