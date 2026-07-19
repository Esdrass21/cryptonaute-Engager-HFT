# data_ingestion/news_feeds/challenger.py
"""
Challenger, Gray & Christmas job-cut announcements feed.

Public press release:
  https://www.challengergray.com/blog/

Headline: "U.S.-based employers announced N job cuts in <month>"
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

from .retry import HttpClient

log = logging.getLogger("data.challenger")

CHALLENGER_BLOG = "https://www.challengergray.com/blog/"

# "announced 55,000 job cuts in January"
_CUTS_RE = re.compile(
    r"announced\s+([\d,]+)\s+(?:job\s+)?cuts",
    re.IGNORECASE,
)


def parse_challenger_cuts(text: str) -> Optional[float]:
    m = _CUTS_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        v = float(raw)
        # Sanity bound — Challenger cuts range ~10k to 200k/month.
        if 1_000.0 <= v <= 500_000.0:
            return v
    except ValueError:
        pass
    return None


class ChallengerFeed(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.challenger")
        self.poll_interval_s = float(cfg.get("data.news.challenger.poll_interval_s", 86400.0))
        self.url = str(cfg.get("data.news.challenger.url", CHALLENGER_BLOG))
        self._last_poll = 0.0
        self._last_value: Optional[float] = None
        self._http: Optional[HttpClient] = None
        if bool(cfg.get("data.news.challenger.enabled", True)):
            self._http = HttpClient(rate_per_sec=0.5, burst=1.0)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.warning("Challenger feed disabled by config")
        else:
            log.info("Challenger feed online (poll=%ss)", self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        try:
            resp = self._http.get(self.url, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            log.debug("Challenger fetch failed: %s", exc)
            self.metrics.inc("challenger_fetch_failed")
            return
        if resp.status_code != 200:
            self.metrics.inc("challenger_fetch_failed")
            return
        val = parse_challenger_cuts(resp.text)
        if val is None:
            self.metrics.inc("challenger_parse_failed")
            return
        if self._last_value is not None and abs(self._last_value - val) < 1e-6:
            return
        self._last_value = val
        self.emit(make_event(
            EventType.NEWS, "data.challenger",
            payload={
                "id": f"challenger:{int(now)}",
                "ts": now,
                "headline": f"Challenger job cuts: {int(val)}",
                "source": "CHALLENGER",
                "release_id": "CHALLENGER_CUTS",
                "actual": val,
                "units": "count",
            },
            timestamp=now,
        ))
        self.metrics.inc("challenger_observation")

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
