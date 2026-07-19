# core/event_log.py
"""
Durable, append-only event log.

Design goals:
- Crash-safe: fsync after every append by default.
- Deterministic: JSON with sorted keys, one event per line.
- Backend-agnostic: File / In-memory today; Redis Streams later.
- No schema knowledge here — validation happens on the bus.

Events are stored as dicts (NormalizedEvent.to_dict() output). The log is
agnostic to the event type; it just persists dicts and replays them.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator


class EventLog(ABC):
    @abstractmethod
    def append(self, event: Dict[str, Any]) -> None: ...

    @abstractmethod
    def replay(self) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def replay_from(self, offset: int = 0) -> Iterator[Dict[str, Any]]: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class FileEventLog(EventLog):
    """Append-only JSONL event log. Safe for concurrent appends within a process."""

    def __init__(self, path: str | Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"), sort_keys=True)
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

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def replay_from(self, offset: int = 0) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                line = line.strip()
                if line:
                    yield json.loads(line)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class InMemoryEventLog(EventLog):
    """For tests and dry runs. Not durable."""

    def __init__(self) -> None:
        self.events: list[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def replay(self) -> Iterator[Dict[str, Any]]:
        with self._lock:
            snapshot = list(self.events)
        yield from snapshot

    def count(self) -> int:
        with self._lock:
            return len(self.events)

    def replay_from(self, offset: int = 0) -> Iterator[Dict[str, Any]]:
        with self._lock:
            snapshot = list(self.events)
        yield from snapshot[offset:]


def build_event_log(cfg: Dict[str, Any] | None) -> EventLog | None:
    """
    Build an EventLog from a config dict. Returns None if disabled.

    Expected shape:
        {"enabled": true, "backend": "file", "path": "data/events.jsonl", "fsync": true}
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    backend = cfg.get("backend", "file").lower()
    if backend == "file":
        return FileEventLog(cfg.get("path", "data/events.jsonl"), fsync=cfg.get("fsync", True))
    if backend == "memory":
        return InMemoryEventLog()
    raise ValueError(f"Unknown event log backend: {backend!r}")
