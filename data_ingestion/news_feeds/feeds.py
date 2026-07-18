"""News & macro feeds (Layer 1 — §2.2).

Backends selected by ``cfg.data.news.source``: ``sim`` (deterministic headlines, default),
``rss`` (feedparser), ``factory`` (Forex Factory calendar). Each emits NEWS / MACRO; the NLP
layer later scores relevance + sentiment and rebroadcasts NEWS_SCORE.
"""

from __future__ import annotations

import logging
import time
from typing import List

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent
from core.metrics import Metrics

from ..normalizer import normalize_macro, normalize_news

log = logging.getLogger("data.news")

_SIM_HEADLINES = [
    ("{sym} spikes as risk sentiment improves", "bullish"),
    ("Dovish central bank comments pressure {sym}", "bearish"),
    ("{sym} retreats after strong data dump", "bearish"),
    ("Safe-haven bids lift {sym} amid uncertainty", "bullish"),
    ("{sym} volatility compresses ahead of key release", "neutral"),
    ("Hawkish rhetoric triggers {sym} selloff", "bearish"),
    ("{sym} breaks resistance on momentum flows", "bullish"),
]


class NewsFeed(Component):
    def __init__(self, bus, cfg, clock, metrics, name="news"):
        super().__init__(bus, cfg, clock, metrics, name)
        self.symbols: List[str] = cfg.get("symbols", ["EURUSD"])


class SimNewsFeed(NewsFeed):
    """Deterministic headline generator. Emits a headline every ``every_n_ticks`` ticks."""

    def __init__(self, bus, cfg, clock, metrics, seed: int = 7):
        super().__init__(bus, cfg, clock, metrics, name="sim.news")
        import random

        self._rng = random.Random(seed)
        self._counter = 0
        self.every_n_ticks = int(cfg.get("data.news.sim.every_n_ticks", 20))

    def on_tick(self, ev: NormalizedEvent) -> None:
        self._counter += 1
        if self._counter % self.every_n_ticks != 0:
            return
        sym = self._rng.choice(self.symbols)
        tmpl, _ = self._rng.choice(_SIM_HEADLINES)
        title = tmpl.format(sym=sym)
        self.emit(normalize_news("sim.news", title=title, summary=title, time=self.clock.now()))

    def start(self) -> None:
        super().start()
        self.bus.subscribe(EventType.TICK, self.on_tick)


class RSSNewsFeed(NewsFeed):
    def start(self) -> None:
        super().start()
        try:
            import feedparser

            feeds = self.cfg.get("data.news.rss_feeds", []) or [
                "https://www.forexlive.com/feed",
                "https://www.investing.com/rss/news_25.rss",
            ]
            self._feeds = feeds
            self._seen = set()
            self._poll()
            log.info("RSS feed watching %d sources", len(feeds))
        except Exception as exc:
            log.warning("RSS feed unavailable: %s", exc)

    def _poll(self) -> None:
        import feedparser

        for url in self._feeds:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                key = entry.get("id") or entry.get("link")
                if key in self._seen:
                    continue
                self._seen.add(key)
                self.emit(normalize_news("rss", title=entry.get("title", ""),
                                         summary=entry.get("summary", ""), url=entry.get("link", ""),
                                         time=time.time()))


class ForexFactoryFeed(NewsFeed):
    def start(self) -> None:
        super().start()
        log.info("ForexFactory feed prepared (scrape on demand)")


def build_news_feed(bus: MessageBus, cfg: Config, clock: Clock, metrics: Metrics) -> NewsFeed:
    source = cfg.get("data.news.source", "sim")
    if source == "rss":
        return RSSNewsFeed(bus, cfg, clock, metrics)
    if source == "factory":
        return ForexFactoryFeed(bus, cfg, clock, metrics)
    return SimNewsFeed(bus, cfg, clock, metrics)
