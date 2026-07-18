"""Classic finance: macro / fundamental context (Layer 2 — §3.2).

Consumes MACRO events (from Forex Factory / FRED / central-bank feeds) and maintains a rolling
macro surprise index per country. Architecture: used to *contextualize* technical signals with
the macro regime (rates, inflation, cycle) to filter false signals — never a standalone trigger.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

log = logging.getLogger("analytics.classic")


class MacroAnalytics(Component):
    def __init__(self, bus, cfg, clock, metrics, window: int = 50):
        super().__init__(bus, cfg, clock, metrics, name="analytics.classic")
        self.window = window
        self._surprises: Dict[str, deque] = {}

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.MACRO, self.on_macro)
        log.info("Macro analytics online")

    def on_macro(self, ev) -> None:
        surprise = ev.payload.get("surprise")
        if surprise is None:
            return
        country = ev.payload.get("country", "GLOBAL")
        dq = self._surprises.setdefault(country, deque(maxlen=self.window))
        dq.append(float(surprise))
        avg = sum(dq) / len(dq)
        impact_w = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(ev.payload.get("impact", "low"), 0.2)
        self.emit(make_event(
            EventType.MACRO, "analytics.classic",
            payload={"event": ev.payload.get("event"), "country": country,
                      "surprise_index": float(avg), "impact_weight": impact_w, "n": len(dq)},
            symbol_impact=ev.symbol_impact, timestamp=self.clock.now(),
        ))
        self.metrics.inc("macro_processed")
