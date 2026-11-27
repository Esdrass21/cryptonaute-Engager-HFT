"""Macro surprise engine (Layer 2 — §3.2).

Ingests the macro calendar and the published headline numbers, computes the
*surprise* of each release, and emits structured ``MACRO`` events that the
regime engine, the quality scorer and the strategy router consume:

    MACRO event payload
    -------------------
    actual      released headline number
    consensus   consensus forecast at release time
    surprise    actual - consensus          (signed, % or thousands)
    surprise_z  surprise / surprise_std     (normalized, comparable across releases)
    direction   +1 = positive surprise, -1 = negative surprise
    impact      High | Medium | Low         (from |surprise_z| magnitude)
    surprise_index  rolling mean of surprises for this release_id
    timestamp    when the news was published

Surprise computation
--------------------
    surprise      = actual - consensus
    surprise_z    = surprise / surprise_std(release_id)
    direction     = sign(surprise_z)

The ``surprise_std`` prior is per-release (config/data/default.yaml), mirroring the
values already used by the BLS/calendar feeds.  A large |surprise_z| is exactly the
kind of event that flags the regime engine to re-weight strategies (trend vs mean-
reversion, reduced sizing, news confirmation) — which is the job of the macro
surprise engine: carry the signal, never decide the trade.

Design note
-----------
Incoming news *text* is parsed into release_id/actual/census by the data feeders
(data/news_feeds/*).  This engine owns the surprise aggregation, not the parsing.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from core.component import Component
from core.events import EventType, make_event

log = logging.getLogger("analytics.macro_surprise")

_IMPACT = [(0.0, 0.8, "Low"), (0.8, 1.5, "Medium"), (1.5, float("inf"), "High")]


def _impact_class(z: float) -> str:
    for lo, hi, label in _IMPACT:
        if lo <= abs(z) < hi:
            return label
    return "Low"


class MacroSurpriseEngine(Component):
    """Compose a per-release macro surprise into a uniform MACRO signal.

    Subscribes to RELEASE_EXPECTED (schedule) and RELEASE_ACTUAL (headline at
    publication), then emits MACRO events with direction and impact so the
    regime model can re-weight strategies without seeing a decimal.
    """

    def __init__(self, bus, cfg, clock, metrics, window: int = 30, include_count: bool = True):
        super().__init__(bus, cfg, clock, metrics, name="analytics.macro_surprise")
        self.window = max(1, int(window))
        self.include_count = include_count
        # Scheduled releases: release_id -> {"consensus": float|None, "impact": str, "symbol_impact": list, "scheduled_at": float}
        self._schedule: dict[str, dict] = {}
        # Rolling surprises per release_id: release_id -> deque of float
        self._surprises: dict[str, deque] = {}
        # Rolling actuals per release_id: release_id -> deque of (actual, ts)
        self._actuals: dict[str, deque] = {}
        # Count of unexpected events by direction (for regime model)
        self._direction_count: dict[str, dict[int, int]] = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.RELEASE_EXPECTED, self._on_release_expected)
        self.bus.subscribe(EventType.RELEASE_ACTUAL, self._on_release_actual)
        log.info("MacroSurpriseEngine online")

    # ------------------------------------------------------------------
    def _on_release_expected(self, ev) -> None:
        p = ev.payload
        rid = p.get("release_id")
        if not rid:
            return
        self._schedule[rid] = {
            "consensus": p.get("consensus"),
            "impact": p.get("impact", p.get("release", rid).upper().replace("_", " ") if isinstance(p.get("release"), str) else "Unknown"),
            "symbol_impact": p.get("symbol_impact", []),
            "scheduled_at": p.get("scheduled_at") or self.clock.now(),
            "expected_event": p.get("release") or rid,
        }
        log.debug("scheduled %s (consensus=%s, impact=%s)", rid, p.get("consensus"), self._schedule[rid]["impact"])

    # ------------------------------------------------------------------
    def _on_release_actual(self, ev) -> None:
        p = ev.payload
        rid = p.get("release_id")
        if not rid:
            return

        actual = p.get("actual")
        consensus = p.get("consensus")
        if actual is None or consensus is None:
            # still archive for the schedule, no surprise yet
            self._append(rid, {"actual": actual, "consensus": consensus, "surprise": None})
            return

        # -- surprise ------------------------------------------------- #
        raw = float(actual) - float(consensus)
        std = float(p.get("surprise_std", 1.0)) or 1.0
        surprise = float(raw)
        z = float(raw / max(std, 1e-9))
        direction = 1 if z >= 0 else -1
        impact = _impact_class(abs(z))
        ts = p.get("timestamp") or self.clock.now()

        self._append(rid, {"actual": actual, "consensus": consensus, "surprise": surprise})

        # surprise index (rolling mean of surprises for this release)
        dq = self._surprises.setdefault(rid, deque(maxlen=self.window))
        dq.append(surprise)
        index = float(np.mean(dq))
        avg_n = float(len(dq))

        # ---- page-level surprise counts (for the regime model) -------- #
        dc = self._direction_count.setdefault(rid, {1: 0, -1: 0})
        dc[direction] = dc.get(direction, 0) + 1
        total = float(sum(dc.values()))

        # impact weight on the surprise index (high impact moves the index more)
        impact_w = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(impact.lower(), 0.3)

        self.emit(
            make_event(
                EventType.MACRO, "analytics.macro_surprise",
                payload={
                    "release_id": rid,
                    "event": p.get("release") or rid,
                    "actual": float(actual),
                    "consensus": float(consensus),
                    "surprise": surprise,
                    "surprise_z": z,
                    "direction": direction,
                    "impact": impact.title(),
                    "impact_weight": impact_w,
                    "surprise_index": index,
                    "n": avg_n,
                    "direction_count": sum(dc.values()),
                    "direction_bias": (dc.get(1, 0) - dc.get(-1, 0)) / max(total, 1e-9),
                    "surprise_quality": 1.0 - min(1.0, abs(z) / 3.0),  # 1.0 = high confidence
                    "timestamp": ts,
                },
                symbol_impact=p.get("symbol_impact", ["GLOBAL"]),
                timestamp=ts,
            ),
        )
        self.metrics.inc("macro_surprise", release_id=rid, impact=impact.lower())

    # ------------------------------------------------------------------
    def _append(self, rid: str, item: dict) -> None:
        self._actuals.setdefault(rid, deque(maxlen=self.window)).append(item)
        # keep the surprise index in step with the actual archive
        self._surprises.setdefault(rid, deque(maxlen=self.window))

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, dict]:
        out = {}
        for rid, dq in self._surprises.items():
            out[rid] = {
                "surprise_index": float(np.mean(dq)) if dq else 0.0,
                "surprise_last": dq[-1] if dq else 0.0,
                "n": len(dq),
                "direction_bias": self._direction_count.get(rid, {}).get(1, 0) - self._direction_count.get(rid, {}).get(-1, 0),
            }
        return out
