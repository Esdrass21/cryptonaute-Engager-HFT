# validation/signal_log.py
"""
Secondary signal log for the replay harness.

Records NOWCAST, CONSENSUS, RELEASE_ACTUAL, RELEASE_EXPECTED — everything
that happens *around* a release — separately from the trade event log.
The drift analyzer consumes this file to measure signal quality without
polluting the trade P&L record.

Format: JSONL, one record per line, sorted keys, byte-stable.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

log = logging.getLogger("validation.signal_log")

TRACKED_TYPES = frozenset({
    "nowcast", "release_expected", "release_actual",
    "news",
})


class SignalLog:
    def __init__(self, path: str | Path, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def record(self, event_type: str, payload: Dict[str, Any], ts: float) -> None:
        if event_type == "news" and payload.get("source") != "consensus":
            return
        rec = {"type": event_type, "ts": float(ts), "payload": payload}
        line = json.dumps(rec, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())

    def replay(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
