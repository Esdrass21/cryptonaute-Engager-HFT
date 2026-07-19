# core/checkpoint.py
"""
Checkpoint store for crash recovery.

Persists a small JSON file next to the event log containing:
  * `offset`        — number of events successfully processed
  * `last_ts`       — timestamp of the last processed event
  * `state`         — broker balance, equity peak, open tickets, key_id
  * `written_at`    — wall-clock time of the write

On unclean shutdown, the orchestrator resumes from `offset` and rebuilds
state without replaying already-processed events. Idempotent: overwrite
the file atomically (write tmp + rename) so a crash mid-write cannot
leave it corrupt.

Usage:
    ck = Checkpoint("data/events.jsonl.checkpoint")
    state = ck.load()               # None if no checkpoint
    ck.write(offset=1234, last_ts=..., state={...})
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("core.checkpoint")


@dataclass
class CheckpointState:
    offset: int = 0
    last_ts: float = 0.0
    state: Dict[str, Any] = field(default_factory=dict)
    written_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Checkpoint:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[CheckpointState]:
        if not self.path.exists():
            return None
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            return CheckpointState(
                offset=int(doc.get("offset", 0)),
                last_ts=float(doc.get("last_ts", 0.0)),
                state=dict(doc.get("state", {})),
                written_at=float(doc.get("written_at", 0.0)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("checkpoint load failed: %s — treating as no checkpoint", exc)
            return None

    def write(self, state: CheckpointState) -> None:
        state.written_at = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, default=str),
                       encoding="utf-8")
        # Atomic rename on POSIX and Windows.
        os.replace(tmp, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
