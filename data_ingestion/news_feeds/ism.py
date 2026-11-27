# data_ingestion/news_feeds/ism.py
"""
ISM manufacturing + services employment sub-index feed.

ISM is proprietary. The monthly press release is public, and we parse the
*employment* sub-index — the input the NFP nowcast actually consumes.

Sources:
  https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/
  (link rotates monthly; the RSS feed points at the current release)

Fallback: if the scrape fails, we do NOT substitute a proxy. We emit nothing
and increment a metric. The nowcast degrades gracefully (missing feature → 0).

Replay-safe: no network in virtual mode.
"""
from __future__ import annotations

import logging
import re

from core.component import Component
from core.events import EventType, make_event

from .retry import HttpClient

log = logging.getLogger("data.ism")

ISM_RSS = "https://www.ismworld.org/feed/?post_type=reports"
ISM_MFG_URL = "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/"
ISM_SVC_URL = "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/services/"

_EMP_RE = re.compile(
    r"Employment Index (?:registered|was|stood at|read)\s+([\d.]+)",
    re.IGNORECASE,
)


def parse_ism_employment(html: str) -> float | None:
    m = _EMP_RE.search(html)
    if not m:
        return None
    try:
        v = float(m.group(1))
        # Sanity bound — ISM indices live in [30, 70]; reject noise.
        if 20.0 <= v <= 80.0:
            return v
    except ValueError:
        pass
    return None


class ISMFeed(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.ism")
        self.poll_interval_s = float(cfg.get("data.news.ism.poll_interval_s", 86400.0))
        self.mfg_url = str(cfg.get("data.news.ism.mfg_url", ISM_MFG_URL))
        self.svc_url = str(cfg.get("data.news.ism.svc_url", ISM_SVC_URL))
        self._last_poll = 0.0
        self._last_value: dict[str, float | None] = {}
        self._http: HttpClient | None = None
        if bool(cfg.get("data.news.ism.enabled", True)):
            self._http = HttpClient(rate_per_sec=0.5, burst=1.0)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.warning("ISM feed disabled by config")
        else:
            log.info("ISM feed online (poll=%ss)", self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        self._poll("ISM_MANUFACTURING_EMP", self.mfg_url)
        self._poll("ISM_SERVICES_EMP", self.svc_url)

    def _poll(self, release_id: str, url: str) -> None:
        try:
            resp = self._http.get(url, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            log.debug("ISM fetch failed %s: %s", release_id, exc)
            self.metrics.inc("ism_fetch_failed", release_id=release_id)
            return
        if resp.status_code != 200:
            self.metrics.inc("ism_fetch_failed", release_id=release_id)
            return
        val = parse_ism_employment(resp.text)
        if val is None:
            self.metrics.inc("ism_parse_failed", release_id=release_id)
            return
        prev = self._last_value.get(release_id)
        if prev is not None and abs(prev - val) < 1e-6:
            return
        self._last_value[release_id] = val
        now = self.clock.now()
        self.emit(make_event(
            EventType.NEWS, "data.ism",
            payload={
                "id": f"ism:{release_id}:{int(now)}",
                "ts": now,
                "headline": f"{release_id}: {val}",
                "source": "ISM",
                "release_id": release_id,
                "actual": val,
                "units": "index",
            },
            timestamp=now,
        ))
        self.metrics.inc("ism_observation", release_id=release_id)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
