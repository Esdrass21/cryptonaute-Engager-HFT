# data_ingestion/news_feeds/gdelt.py
"""
GDELT Project feed.

GDELT monitors news media worldwide and extracts structured events + article
metadata. The Doc API v2 is public, no auth, generous rate limits.

Endpoints used:
  * ArtList  — most recent articles matching a query.
    GET https://api.gdeltproject.org/api/v2/doc/doc
        ?query=<q>&mode=artlist&format=json&maxrecords=250&sort=hybridrel

  * TimelineVol — volume of coverage over time for a query.
    (used later for "how much is the world talking about X" signals)

Signals we extract:
  * Article titles + source domains + language + seen timestamp.
  * Optional: a topic-tagging layer (L2 NLP) consumes the NEWS events we emit.

Why it matters: GDELT catches geopolitical and macro events (war, elections,
rate decisions, sanctions) in near-real-time across 65 languages, without
needing a Twitter/X account or a Bloomberg terminal.

Replay-safe: in `clock.mode == "virtual"`, no network call is made.

Rate limits: GDELT doesn't publish them; empirically ~1 req/sec is fine.
We default to 0.5 req/sec.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

from .retry import HttpClient

log = logging.getLogger("data.gdelt")

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_GDELT_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z")


def parse_gdelt_ts(s: str) -> Optional[float]:
    m = _GDELT_TS_RE.match(s or "")
    if not m:
        return None
    try:
        y, mo, d, hh, mm, ss = (int(x) for x in m.groups())
        return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class GDELTArticle:
    url: str
    title: str
    domain: str
    language: str
    country: str
    ts: float
    social_image: str = ""


def parse_gdelt_artlist(text: str) -> List[GDELTArticle]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    arts = doc.get("articles") if isinstance(doc, dict) else None
    if not isinstance(arts, list):
        return []
    out: List[GDELTArticle] = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        ts = parse_gdelt_ts(a.get("seendate", ""))
        if ts is None:
            continue
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue
        out.append(GDELTArticle(
            url=url, title=title,
            domain=(a.get("domain") or "").strip(),
            language=(a.get("language") or "").strip(),
            country=(a.get("sourcecountry") or "").strip(),
            ts=ts,
            social_image=(a.get("socialimage") or "").strip(),
        ))
    return out


class GDELTFeed(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.gdelt")
        self.poll_interval_s = float(cfg.get("data.news.gdelt.poll_interval_s", 300.0))
        self.max_records = int(cfg.get("data.news.gdelt.max_records", 250))
        self.queries = cfg.get("data.news.gdelt.queries", []) or []
        self._http: Optional[HttpClient] = None
        self._last_poll = 0.0
        self._seen_urls: set[str] = set()
        self._seen_cap = 20_000
        if bool(cfg.get("data.news.gdelt.enabled", False)) and self.queries:
            self._http = HttpClient(rate_per_sec=0.5, burst=1.0)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.info("GDELT feed disabled (enabled=false or no queries)")
            return
        log.info("GDELT feed online (queries=%d, poll=%ss)",
                 len(self.queries), self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if self.clock.mode == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        for q in self.queries:
            self._poll_query(q)

    def _poll_query(self, q: Dict) -> None:
        query_id = str(q.get("id", "?"))
        qstr = q.get("q")
        if not qstr:
            return
        params = {
            "query": qstr,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(self.max_records, 250)),
            "sort": "hybridrel",
            "timespan": "15min",
        }
        try:
            resp = self._http.get(GDELT_DOC_API, params=params, timeout=15.0)
        except Exception as exc:
            log.debug("GDELT %s fetch failed: %s", query_id, exc)
            self.metrics.inc("gdelt_fetch_failed", query=query_id)
            return
        if resp.status_code != 200:
            self.metrics.inc("gdelt_fetch_failed", query=query_id)
            return
        articles = parse_gdelt_artlist(resp.text)
        self.metrics.observe("gdelt_articles_fetched", float(len(articles)))
        symbols = q.get("symbols") or []
        for a in articles:
            if a.url in self._seen_urls:
                continue
            self._seen_urls.add(a.url)
            if len(self._seen_urls) > self._seen_cap:
                self._seen_urls = set(list(self._seen_urls)[-self._seen_cap // 2:])
            now = self.clock.now()
            self.emit(make_event(
                EventType.NEWS, "data.gdelt",
                payload={
                    "id": f"gdelt:{query_id}:{a.url}",
                    "ts": now,
                    "headline": a.title,
                    "source": "GDELT",
                    "query_id": query_id,
                    "domain": a.domain,
                    "language": a.language,
                    "country": a.country,
                    "url": a.url,
                    "published_ts": a.ts,
                    "symbols": list(symbols),
                },
                timestamp=now,
            ))
            self.metrics.inc("gdelt_news", query=query_id)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
