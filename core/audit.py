# core/audit.py
"""
Immutable audit log for sensitive actions.

Separate from core.event_log (which records every bus event), this log
records only security-sensitive human operations: circuit-breaker resets,
live activations, risk-limit changes, and key rotations. Every decision
(approved or denied) is persisted.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("core.audit")


@dataclass(frozen=True)
class AuditRecord:
    ts: float
    action: str
    actor: str
    approvers: list[str]
    outcome: str            # "approved" | "denied" | "executed" | "failed"
    details: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    def __init__(self, path: str | Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def now() -> float:
    return time.time()
