"""Tests for Layer 0 (core contracts) and Layer 1 (data ingestion)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.events import EventType, NormalizedEvent, make_event
from core.bus import MessageBus
from core.config import Config, Clock
from core.metrics import Metrics
from data_ingestion.normalizer import normalize_candle, normalize_news, normalize_tick
from data_ingestion.market_feeds.feeds import SimMarketFeed


def test_event_roundtrip():
    ev = make_event(EventType.TICK, "sim", {"bid": 1.0, "ask": 1.1}, symbol_impact=["EURUSD"])
    d = ev.to_dict()
    ev2 = NormalizedEvent.from_dict(d)
    assert ev2.type == ev.type and ev2.source == ev.source and ev2.symbol_impact == ev.symbol_impact
    assert ev2.payload == ev.payload


def test_bus_pubsub_order():
    bus = MessageBus()
    seen = []

    def h(ev):
        seen.append(ev.payload["v"])

    bus.subscribe(EventType.CANDLE, h)
    for v in range(5):
        bus.publish(make_event(EventType.CANDLE, "t", {"v": v}))
    assert seen == [0, 1, 2, 3, 4]


def test_bus_replay_reaction_resolves_before_next_raw():
    """A handler's emitted event must be processed before the next raw (recorded) event.

    This is what lets the full decision cascade (proposal -> veto -> fill) complete within
    the same replay step instead of being deferred to the end of the recording.
    """
    bus = MessageBus(replay=True)
    seen = []
    bus.subscribe(EventType.CANDLE, lambda e: (
        seen.append(("raw", e.payload["v"])),
        bus.publish(make_event(EventType.TECH_SIGNAL, "derived", {"v": e.payload["v"]})),
    ))
    bus.subscribe(EventType.TECH_SIGNAL, lambda e: seen.append(("derived", e.payload["v"])))
    for v in range(3):
        bus.enqueue(make_event(EventType.CANDLE, "t", {"v": v}))
    bus.run_replay()
    assert seen == [
        ("raw", 0), ("derived", 0),
        ("raw", 1), ("derived", 1),
        ("raw", 2), ("derived", 2),
    ]


def test_normalizer_shapes():
    t = normalize_tick("mt5", "EURUSD", bid=1.0, ask=1.02, last=1.01)
    assert t.type == EventType.TICK.value and t.symbol_impact == ["EURUSD"]
    c = normalize_candle("mt5", "XAUUSD", "M5", open_=2300, high=2310, low=2290, close=2305)
    assert c.payload["close"] == 2305 and c.symbol_impact == ["XAUUSD"]
    n = normalize_news("rss", title="EURUSD rallies")
    assert n.type == EventType.NEWS.value


def test_sim_market_feed_emits_tick_and_candle():
    cfg = Config.load()
    clock = Clock()
    metrics = Metrics()
    feed = SimMarketFeed(MessageBus(), cfg, clock, metrics, seed=42)
    feed.start()
    seen = []
    feed.bus.subscribe(EventType.TICK, lambda e: seen.append(e.type))
    feed.bus.subscribe(EventType.CANDLE, lambda e: seen.append(e.type))
    feed.run_once()
    assert EventType.TICK.value in seen
    # candle appears after candles_per_tick ticks
    for _ in range(cfg.get("data.market.sim.candles_per_tick", 5) - 1):
        feed.run_once()
    assert EventType.CANDLE.value in seen
