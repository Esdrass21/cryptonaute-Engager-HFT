# analytics/nowcast/base.py
"""
Base class for release nowcasts.

A nowcast is a *probabilistic* forecast of an upcoming release, computed from
public inputs. It emits a NOWCAST event with (release_id, mean, stddev).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

log = logging.getLogger("analytics.nowcast")


@dataclass
class NowcastResult:
    release_id: str
    mean: float
    stddev: float
    n_features: int
    confidence: float          # 0..1 — model's self-assessed reliability
    feature_vector: Dict[str, float]


class NowcastBase(Component, ABC):
    """
    Subclasses implement:
      * `features()` — pull the current values of the input features
      * `predict(features)` — return (mean, stddev)
      * `release_ids()` — which releases this nowcast covers
    """

    def __init__(self, bus, cfg, clock, metrics, name: str = "analytics.nowcast") -> None:
        super().__init__(bus, cfg, clock, metrics, name=name)
        self._latest_inputs: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True
        # Learn the inputs we care about from a declared feature list.
        self.bus.subscribe(EventType.NEWS, self._on_news)
        self.bus.subscribe(EventType.TICK, self._on_tick)
        self.bus.subscribe(EventType.RELEASE_EXPECTED, self._on_expected)
        log.info("%s online (releases=%s)", self.name, self.release_ids())

    # ------------------------------------------------------------------ #
    @abstractmethod
    def release_ids(self) -> list[str]: ...

    @abstractmethod
    def features(self) -> Dict[str, float]: ...

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> tuple[float, float]: ...

    # ------------------------------------------------------------------ #
    def _on_news(self, ev) -> None:
        """
        Update internal inputs from a NEWS event. Subclasses override for
        feature-specific handling.
        """
        rid = ev.payload.get("release_id")
        val = ev.payload.get("actual")
        if rid and val is not None:
            self._latest_inputs[rid] = float(val)

    def _on_tick(self, ev) -> None:
        # Some inputs are price-derived (gas prices, oil). Subclasses can extend.
        pass

    def _on_expected(self, ev) -> None:
        rid = ev.payload.get("release_id")
        if rid not in self.release_ids():
            return
        consensus = ev.payload.get("consensus")
        prior = ev.payload.get("prior")
        try:
            feats = self.features()
        except Exception as exc:  # noqa: BLE001
            log.warning("nowcast features() failed: %s", exc)
            return
        mean, stddev = self.predict(feats)
        now = self.clock.now()
        self.emit(make_event(
            EventType.NOWCAST, self.name,
            payload={
                "release_id": rid,
                "mean": mean,
                "stddev": stddev,
                "consensus": consensus,
                "prior": prior,
                "edge_vs_consensus": (mean - (consensus or 0.0)),
                "feature_vector": feats,
                "ts": now,
            },
            timestamp=now,
        ))
        self.metrics.inc("nowcast_emitted", release_id=rid)
