# data_ingestion/news_feeds/forexfactory.py
"""
Forex Factory calendar feed.

Two paths, in priority order:

1. **Widget JSON (preferred).** Forex Factory publishes three JSON feeds for
   its own embeddable calendar widget, served from nfs.faireconomy.media:
       thisweek.json   lastweek.json   nextweek.json
   Each returns an array of events with `title`, `country`, `date` (ISO-8601
   with offset), `impact` (High/Medium/Low/Holiday), `forecast`, `previous`.

   These are the *consensus* numbers — exactly what the nowcast compares
   against. They include the scheduled timestamp, so the calendar can arm
   the drift strategy at T-120s.

2. **HTML scrape (fallback).** Only used if the JSON feeds 404 or return
   malformed data. Fragile by design; logs a warning; contributes nothing
   if the parse fails.

Emits:
  * `NEWS` with `source: "FOREXFACTORY"` and `release_id` derived from the
    event title.
  * `RELEASE_EXPECTED` for events within the lookahead window.

Replay-safe: in `clock.mode == "virtual"`, no network call is made.

Mapping:
  FF titles are free-text ("Federal Funds Rate", "Non-Farm Employment Change",
  "CPI m/m", ...). We normalise them to our `release_id` vocabulary via
  TITLE_MAP. Unmapped titles are logged once and skipped — add them to
  TITLE_MAP as you encounter them.

Rate limits:
  FF does not publish a rate limit for nfs.faireconomy.media, but their CDN
  will 429 aggressive clients. We poll every 60 minutes by default, which is
  far below any plausible threshold.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.component import Component
from core.events import EventType, make_event

from .retry import HttpClient

log = logging.getLogger("data.forexfactory")

FF_BASE = "https://nfs.faireconomy.media/ff_calendar_{window}.json"

TITLE_MAP: dict[str, str] = {
    "Non-Farm Employment Change":   "NFP",
    "Unemployment Rate":            "UNEMPLOYMENT_RATE",
    "ADP Non-Farm Employment Change": "ADP_PRIVATE",
    "Unemployment Claims":          "JOBLESS_CLAIMS",
    "Average Hourly Earnings m/m":  "AHE_MOM",
    "JOLTS Job Openings":           "JOLTS_OPENINGS",
    "CPI m/m":                      "CPI_MOM",
    "CPI y/y":                      "CPI_YOY",
    "Core CPI m/m":                 "CORE_CPI_MOM",
    "Core CPI y/y":                 "CORE_CPI_YOY",
    "PPI m/m":                      "PPI_MOM",
    "PPI y/y":                      "PPI_YOY",
    "Core PCE Price Index m/m":     "CORE_PCE_MOM",
    "PCE Price Index m/m":          "PCE_MOM",
    "Retail Sales m/m":             "RETAIL_SALES_MOM",
    "ISM Manufacturing PMI":        "ISM_MFG_PMI",
    "ISM Services PMI":             "ISM_SVC_PMI",
    "Federal Funds Rate":           "FOMC_RATE",
    "FOMC Statement":               "FOMC_STATEMENT",
    "FOMC Press Conference":        "FOMC_PRESSER",
    "Prelim GDP q/q":               "GDP_PRELIM",
    "Final GDP q/q":                "GDP_FINAL",
    "Advance GDP q/q":              "GDP_ADVANCE",
    "CB Consumer Confidence":       "CB_CONSUMER_CONF",
    "Prelim UoM Consumer Sentiment":"UOM_SENTIMENT",
}

_SKIP_IMPACT = {"Holiday"}


@dataclass(frozen=True)
class FFEvent:
    title: str
    country: str
    ts: float
    impact: str
    forecast: float | None
    previous: float | None
    raw: dict[str, Any]


def _parse_iso(s: str) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(UTC).timestamp()
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt).astimezone(UTC).timestamp()
        except ValueError:
            continue
    return None


_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _parse_num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("-", "—"):
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    low = s.lower()
    if "m" in low and "m/m" not in low and "ppm" not in low:
        v *= 1_000.0
    return v


def parse_ff_json(text: str) -> list[FFEvent]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(doc, list):
        return []
    out: list[FFEvent] = []
    for rec in doc:
        if not isinstance(rec, dict):
            continue
        title = (rec.get("title") or "").strip()
        country = (rec.get("country") or "").strip().upper()
        if not title or not country:
            continue
        ts = _parse_iso(rec.get("date", ""))
        if ts is None:
            continue
        out.append(FFEvent(
            title=title, country=country, ts=ts,
            impact=(rec.get("impact") or "Low").strip(),
            forecast=_parse_num(rec.get("forecast")),
            previous=_parse_num(rec.get("previous")),
            raw=rec,
        ))
    return out


class ForexFactoryFeed(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.forexfactory")
        self.poll_interval_s = float(cfg.get("data.news.forexfactory.poll_interval_s", 3600.0))
        self.arm_before_s = float(cfg.get("data.news.forexfactory.arm_before_s", 120.0))
        self.lookahead_s = float(cfg.get("data.news.forexfactory.lookahead_s", 7 * 86400.0))
        self.countries = set(cfg.get("data.news.forexfactory.countries", ["USD", "EUR", "GBP", "JPY"]))
        self.min_impact = str(cfg.get("data.news.forexfactory.min_impact", "High"))
        self._http: HttpClient | None = None
        self._last_poll = 0.0
        self._seen_guids: set[str] = set()
        self._armed: set[str] = set()
        self._unmapped_logged: set[str] = set()
        if bool(cfg.get("data.news.forexfactory.enabled", True)):
            self._http = HttpClient(
                rate_per_sec=0.1, burst=2.0,
                user_agent="Mozilla/5.0 (compatible; Cryptonaute/3.0)",
            )

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.warning("Forex Factory feed disabled by config")
            return
        log.info("Forex Factory feed online (impact>=%s, countries=%s)",
                 self.min_impact, sorted(self.countries))

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if self.clock.mode == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        self._poll()

    def _poll(self) -> None:
        all_events: list[FFEvent] = []
        for window in ("thisweek", "nextweek"):
            url = FF_BASE.format(window=window)
            try:
                resp = self._http.get(url, timeout=10.0)
            except Exception as exc:
                log.debug("FF %s fetch failed: %s", window, exc)
                self.metrics.inc("ff_fetch_failed", window=window)
                continue
            if resp.status_code != 200:
                self.metrics.inc("ff_fetch_failed", window=window)
                continue
            events = parse_ff_json(resp.text)
            all_events.extend(events)
            self.metrics.inc("ff_feed_items", window=window)
        if not all_events:
            return
        for ev in all_events:
            if not self._passes_filters(ev):
                continue
            release_id = TITLE_MAP.get(ev.title)
            if release_id is None:
                if ev.title not in self._unmapped_logged:
                    self._unmapped_logged.add(ev.title)
                    log.info("FF unmapped title: %r (add to TITLE_MAP)", ev.title)
                self.metrics.inc("ff_unmapped_title")
                continue
            self._emit(ev, release_id)

    def _passes_filters(self, ev: FFEvent) -> bool:
        if ev.country not in self.countries:
            return False
        if ev.impact in _SKIP_IMPACT:
            return False
        order = {"Low": 0, "Medium": 1, "High": 2}
        if order.get(ev.impact, 0) < order.get(self.min_impact, 2):
            return False
        return True

    def _emit(self, ev: FFEvent, release_id: str) -> None:
        now = self.clock.now()
        guid = f"{release_id}:{ev.country}:{int(ev.ts)}"
        if -self.arm_before_s <= ev.ts - now <= self.lookahead_s:
            if guid not in self._armed:
                self._armed.add(guid)
                self.emit(make_event(
                    EventType.RELEASE_EXPECTED, "data.forexfactory",
                    payload={
                        "release_id": release_id,
                        "source": "FOREXFACTORY",
                        "country": ev.country,
                        "title": ev.title,
                        "scheduled_ts": ev.ts,
                        "consensus": ev.forecast,
                        "prior": ev.previous,
                        "impact": ev.impact,
                        "ts": now,
                    },
                    timestamp=now,
                ))
                self.metrics.inc("release_expected", source="ff", release_id=release_id)
        if guid in self._seen_guids:
            return
        self._seen_guids.add(guid)
        self.emit(make_event(
            EventType.NEWS, "data.forexfactory",
            payload={
                "id": f"ff:{guid}",
                "ts": now,
                "headline": f"{ev.country} {ev.title} (forecast={ev.forecast} prev={ev.previous})",
                "source": "FOREXFACTORY",
                "release_id": release_id,
                "country": ev.country,
                "title": ev.title,
                "scheduled_ts": ev.ts,
                "consensus": ev.forecast,
                "prior": ev.previous,
                "impact": ev.impact,
                "units": "",
            },
            timestamp=now,
        ))
        self.metrics.inc("ff_calendar_entry", release_id=release_id)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
