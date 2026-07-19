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


def test_crypto_feed_symbol_normalization():
    """BTCUSDT -> BTC/USDT; MT5-style symbols (EURUSD/XAUUSD) are skipped (None)."""
    from data_ingestion.market_feeds.feeds import CryptoMarketFeed

    assert CryptoMarketFeed._to_ccxt_symbol("BTCUSDT") == "BTC/USDT"
    assert CryptoMarketFeed._to_ccxt_symbol("ETH/USDT") == "ETH/USDT"
    assert CryptoMarketFeed._to_ccxt_symbol("SOLUSDC") == "SOL/USDC"
    assert CryptoMarketFeed._to_ccxt_symbol("EURUSD") is None       # USD quote ambiguous -> skip
    assert CryptoMarketFeed._to_ccxt_symbol("XAUUSD") is None       # metal, not crypto -> skip


def test_crypto_feed_emits_from_mocked_ccxt(monkeypatch):
    """CryptoMarketFeed emits TICK+CANDLE from a fake ccxt exchange (no network)."""
    from data_ingestion.market_feeds import feeds as feeds_mod

    cfg = Config.load()
    cfg.set("symbols", ["BTCUSDT"])
    cfg.set("execution.exchange", "binance")
    feed = feeds_mod.CryptoMarketFeed(MessageBus(), cfg, Clock(), Metrics())

    class FakeEx:
        def fetch_ticker(self, pair):
            return {"bid": 60000.0, "ask": 60010.0, "last": 60005.0, "baseVolume": 12.5}

        def fetch_ohlcv(self, pair, timeframe="1m", limit=1):
            return [[0, 60000.0, 60100.0, 59900.0, 60005.0, 12.5]]

    feed._ex = FakeEx()  # inject fake, bypass start()/network
    feed._running = True
    seen = []
    feed.bus.subscribe(EventType.TICK, lambda e: seen.append("tick"))
    feed.bus.subscribe(EventType.CANDLE, lambda e: seen.append("candle"))
    feed.run_once()
    assert "tick" in seen and "candle" in seen


def test_rss_news_feed_emits_from_mocked_feedparser(monkeypatch):
    """RSSNewsFeed emits NEWS events from a fake feedparser (no network)."""
    import types
    from data_ingestion.news_feeds import feeds as nfeeds

    fake = types.SimpleNamespace(entries=[
        {"id": "1", "title": "BTC rallies on ETF inflows", "summary": "...", "link": "http://x/1"},
        {"id": "2", "title": "EURUSD dips after CPI", "summary": "...", "link": "http://x/2"},
    ])
    fake_module = types.SimpleNamespace(parse=lambda url: fake)
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)

    cfg = Config.load()
    cfg.set("data.news.rss_feeds", ["http://fake/feed"])
    feed = nfeeds.RSSNewsFeed(MessageBus(), cfg, Clock(), Metrics())
    out = []
    feed.bus.subscribe(EventType.NEWS, lambda e: out.append(e))
    feed.start()  # initial poll pulls both entries
    assert len(out) == 2
    # de-dup: a second poll of the same entries emits nothing new
    feed._poll()
    assert len(out) == 2


def test_check_keys_reports_absence_without_leaking(capsys, monkeypatch):
    """--check-keys returns 0 and reports missing creds; never prints secret values."""
    import main as m

    for k in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "MISTRAL_API_KEY",
              "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        monkeypatch.delenv(k, raising=False)
    rc = m.cmd_check_keys(m.build_parser().parse_args(["--check-keys"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "BINANCE_API_KEY" in out and "MISTRAL_API_KEY" in out
    assert "keys absent" in out  # no live auth attempted when absent
