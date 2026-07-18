"""Synchronous message bus with a deterministic in-memory implementation plus an
optional Redis Streams backend.

Design goals
------------
* **One contract for the whole stack.** Producers call ``publish``, consumers call
  ``subscribe``. There is no polling in the critical path — the execution loop reacts
  to events as they arrive (architecture §5).
* **Deterministic by default.** The in-memory bus supports *replay mode*: feed it a
  pre-recorded ordered list of events and it will hand them to subscribers in exact
  order, which is what the validation/replay harness (validation/replay_harness.py)
  needs.
* **Optional Redis.** When ``REDIS_URL`` is configured and ``redis`` is importable, the
  bus fans events out to a Redis stream so multiple processes can share the pipeline.
  If Redis is unavailable the bus silently falls back to in-memory (never blocks).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .events import EventType, NormalizedEvent, make_event

log = logging.getLogger("core.bus")

# A handler is sync or async, receives a single NormalizedEvent.
Handler = Callable[[NormalizedEvent], Any]


class MessageBus:
    """Pub/sub hub.

    Parameters
    ----------
    backend : str
        ``"memory"`` (default, deterministic) or ``"redis"`` (fan-out to a stream).
    redis_url : Optional[str]
        Required for the redis backend. If the connection fails we downgrade to
        in-memory automatically and log a warning.
    replay : bool
        If True the bus buffers published events into a queue instead of dispatching,
        and ``run_replay()`` / ``step_replay()`` drive them through subscribers in
        insertion order. Used by the validation harness.
    """

    def __init__(
        self,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        replay: bool = False,
    ) -> None:
        self.backend = backend
        self.redis_url = redis_url
        self.replay = replay
        self._subs: Dict[str, List[Handler]] = defaultdict(list)
        self._subs_all: List[Handler] = []
        self._lock = threading.RLock()
        self._replay_queue: deque[NormalizedEvent] = deque()
        self._redis = None
        self._redis_stream = "cryptonaute:events"
        if backend == "redis":
            self._init_redis()

    # ------------------------------------------------------------------ #
    def _init_redis(self) -> None:
        try:  # pragma: no cover - optional backend
            import redis  # type: ignore

            self._redis = redis.Redis.from_url(self.redis_url or "redis://localhost:6379/0")
            self._redis.ping()
            log.info("MessageBus: redis backend connected (%s)", self.redis_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("MessageBus: redis unavailable (%s) — falling back to memory", exc)
            self.backend = "memory"

    # ------------------------------------------------------------------ #
    # subscription
    # ------------------------------------------------------------------ #
    def subscribe(self, event_type: EventType | str | None, handler: Handler) -> None:
        """Subscribe ``handler`` to one event type, or to *all* types if None."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        with self._lock:
            if key is None:
                self._subs_all.append(handler)
            else:
                self._subs[key].append(handler)

    def unsubscribe(self, event_type: EventType | str | None, handler: Handler) -> None:
        key = event_type.value if isinstance(event_type, EventType) else event_type
        with self._lock:
            if key is None:
                if handler in self._subs_all:
                    self._subs_all.remove(handler)
            else:
                if handler in self._subs[key]:
                    self._subs[key].remove(handler)

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def publish(self, event: NormalizedEvent) -> None:
        if self.replay:
            self._replay_queue.append(event)
            return
        if self.backend == "redis" and self._redis is not None:
            try:  # pragma: no cover - optional backend
                self._redis.xadd(self._redis_stream, {"ev": json.dumps(event.to_dict())})
            except Exception as exc:  # noqa: BLE001
                log.warning("MessageBus: redis xadd failed (%s) — dispatching in-memory", exc)
                self._dispatch(event)
        else:
            self._dispatch(event)

    def _dispatch(self, event: NormalizedEvent) -> None:
        handlers = list(self._subs_all)
        with self._lock:
            handlers += list(self._subs.get(event.type, []))
        for h in handlers:
            self._call(h, event)

    @staticmethod
    def _call(handler: Handler, event: NormalizedEvent) -> None:
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                # fire-and-forget; critical path must not await remote LLM calls
                asyncio.ensure_future(result)
        except Exception as exc:  # noqa: BLE001
            log.exception("MessageBus handler %s raised: %s", getattr(handler, "__name__", handler), exc)

    # ------------------------------------------------------------------ #
    # replay mode (validation harness)
    # ------------------------------------------------------------------ #
    def enqueue(self, event: NormalizedEvent) -> None:
        """Add an event to the replay queue without dispatching."""
        self._replay_queue.append(event)

    def step_replay(self) -> Optional[NormalizedEvent]:
        """Dispatch the next queued event (in order) and return it, or None if empty."""
        if not self._replay_queue:
            return None
        ev = self._replay_queue.popleft()
        self._dispatch(ev)
        return ev

    def run_replay(self) -> int:
        """Dispatch every queued event in order. Returns count dispatched."""
        n = 0
        while self._replay_queue:
            self.step_replay()
            n += 1
        return n

    def replay_size(self) -> int:
        return len(self._replay_queue)


# Convenience module-level singleton used by the orchestrator. Tests build their own.
_default_bus: Optional[MessageBus] = None


def get_bus() -> MessageBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = MessageBus()
    return _default_bus


def reset_bus() -> None:
    global _default_bus
    _default_bus = None
