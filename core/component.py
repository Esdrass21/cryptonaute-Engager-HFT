"""Base classes shared across layers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .bus import MessageBus
from .config import Clock, Config
from .events import NormalizedEvent
from .metrics import Metrics

log = logging.getLogger("core.component")


class Component(ABC):
    """Common lifecycle for every layer: wired with bus/cfg/clock/metrics, then
    ``start()`` / ``stop()``. Concrete layers subscribe to the bus in ``start()``."""

    def __init__(
        self,
        bus: MessageBus,
        cfg: Config,
        clock: Clock,
        metrics: Metrics,
        name: str = "component",
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.clock = clock
        self.metrics = metrics
        self.name = name
        self._running = False

    @abstractmethod
    def start(self) -> None: ...

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def emit(self, event: NormalizedEvent) -> None:
        self.bus.publish(event)

    def handle(self, event: NormalizedEvent) -> None:  # pragma: no cover - default noop
        """Subclasses override to react to subscribed events."""
        raise NotImplementedError
