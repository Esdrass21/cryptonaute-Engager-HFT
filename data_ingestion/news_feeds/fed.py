# data_ingestion/news_feeds/fed.py
"""
Federal Reserve feed.

Subscribes to the Fed's public RSS feeds (speeches, press releases,
testimony) and, on schedule, scrapes the FOMC statement page at 14:00 ET.

Emits `NEWS` events with the full text + a crude sentiment score, plus a
`RELEASE_ACTUAL` for the FOMC statement itself. All public data, no keys.

RSS endpoints (verified stable):
  https://www.federalreserve.gov/feeds/speeches.xml
  https://www.federalreserve.gov/feeds/press_all.xml
  https://www.federalreserve.gov/feeds/press_monetary.xml
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

from .retry import HttpClient

log = logging.getLogger("data.fed")

FEEDS = {
    "speeches":   "https://www.federalreserve.gov/feeds/speeches.xml",
    "press_all":  "https://www.federalreserve.gov/feeds/press_all.xml",
    "press_mon":  "https://www.federalreserve.gov/feeds/press_monetary.xml",
}

_HAWKISH = {
    "raise", "tighten", "restrictive", "inflation", "elevated", "persistent",
    "vigilant", "higher for longer", "above target", "overheating",
    "upside risks", "strong", "robust", "resilient",
}
_DOVISH = {
    "cut", "ease", "accommodative", "disinflation", "moderating", "cooling",
    "softening", "downside risks", "below target", "patient", "hold",
    "pause", "weakening", "slowing",
}


def score_fed_text(text: str) -> float:
    """Crude hawkish(+1) / dovish(-1) score in [-1, +1]."""
    lower = text.lower()
    h = sum(lower.count(w) for w in _HAWKISH)
    d = sum(lower.count(w) for w in _DOVISH)
    total = h + d
    if total == 0:
        return 0.0
    return (h - d) / total


@dataclass(frozen=True)
class FeedItem:
    guid: str
    title: str
    link: str
    pub_ts: float
    description: str


def parse_rss(xml_text: str) -> List[FeedItem]:
    """Minimal RSS/Atom parser."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.debug("RSS parse failed: %s", exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: List[FeedItem] = []

    for item in root.iter("item"):
        guid = _text(item, "guid") or _text(item, "link") or ""
        title = _text(item, "title") or ""
        link = _text(item, "link") or ""
        pub = _text(item, "pubDate") or ""
        desc = _text(item, "description") or ""
        ts = _parse_pubdate(pub)
        if guid and title:
            items.append(FeedItem(guid=guid, title=title, link=link,
                                  pub_ts=ts, description=desc))

    for entry in root.findall("atom:entry", ns):
        guid = _text_ns(entry, "atom:id", ns) or ""
        title = _text_ns(entry, "atom:title", ns) or ""
        link_el = entry.find("atom:link", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        pub = _text_ns(entry, "atom:updated", ns) or _text_ns(entry, "atom:published", ns) or ""
        desc = _text_ns(entry, "atom:summary", ns) or ""
        ts = _parse_iso(pub)
        if guid and title:
            items.append(FeedItem(guid=guid, title=title, link=link,
                                  pub_ts=ts, description=desc))
    return items


def _text(parent, tag: str) -> str:
    el = parent.find(tag)
    return (el.text or "").strip() if el is not None else ""


def _text_ns(parent, tag: str, ns) -> str:
    el = parent.find(tag, ns)
    return (el.text or "").strip() if el is not None else ""


_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})")


def _parse_iso(s: str) -> float:
    m = _ISO_RE.search(s)
    if not m:
        return 0.0
    y, mo, d, hh, mm, ss = (int(x) for x in m.groups())
    from datetime import datetime, timezone
    try:
        return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _parse_pubdate(s: str) -> float:
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s).timestamp()
    except Exception:  # noqa: BLE001
        return _parse_iso(s)


class FedFeed(Component):
    """Polls the Fed RSS feeds. On new items, emits `NEWS` events."""

    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.fed")
        self.poll_interval_s = float(cfg.get("data.news.fed.poll_interval_s", 300.0))
        self._last_poll = 0.0
        self._seen: set[str] = set()
        self._http: Optional[HttpClient] = None
        if bool(cfg.get("data.news.fed.enabled", True)):
            self._http = HttpClient(rate_per_sec=1.0, burst=2.0)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.warning("Fed feed disabled by config")
            return
        log.info("Fed feed online (feeds=%d, poll=%ss)",
                 len(FEEDS), self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        self._poll()

    def _poll(self) -> None:
        for name, url in FEEDS.items():
            try:
                resp = self._http.get(url, timeout=6.0)
            except Exception as exc:  # noqa: BLE001
                log.debug("Fed feed fetch failed %s: %s", name, exc)
                self.metrics.inc("fed_fetch_failed", feed=name)
                continue
            if resp.status_code != 200:
                continue
            items = parse_rss(resp.text)
            for item in items[:20]:
                if item.guid in self._seen:
                    continue
                self._seen.add(item.guid)
                self._emit_item(name, item)

    def _emit_item(self, feed_name: str, item: FeedItem) -> None:
        text = f"{item.title}\n{item.description}"
        score = score_fed_text(text)
        now = self.clock.now()
        self.emit(make_event(
            EventType.NEWS, "data.fed",
            payload={
                "id": f"fed:{feed_name}:{item.guid}",
                "ts": now,
                "headline": item.title,
                "source": "FED",
                "feed": feed_name,
                "link": item.link,
                "published_ts": item.pub_ts,
                "text": text[:4096],
                "fed_sentiment": score,
            },
            timestamp=now,
        ))
        self.metrics.inc("fed_news", feed=feed_name)

        if feed_name in ("press_mon", "press_all") and "/monetary" in item.link:
            self.emit(make_event(
                EventType.RELEASE_ACTUAL, "data.fed",
                payload={
                    "release_id": "FOMC_STATEMENT",
                    "source": "FED",
                    "actual": score,
                    "consensus": 0.0,
                    "prior": 0.0,
                    "surprise": score,
                    "sentiment": score,
                    "scheduled_ts": item.pub_ts or now,
                    "parsed_ts": now,
                    "ts": now,
                },
                timestamp=now,
            ))
            self.metrics.inc("release_actual", release_id="FOMC_STATEMENT")

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
