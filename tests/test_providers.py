"""Tests for external market-data providers (Alpaca / Alpha Vantage / TwelveData / Finazon)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType
from core.metrics import Metrics
from data_ingestion.market_feeds.providers import (
    AlpacaFeed,
    AlphaVantageFeed,
    FinazonFeed,
    TwelveDataFeed,
    fetch_history,
)


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def _mk(feed_cls, monkeypatch, data, cfg_extra=None):
    cfg = Config.load()
    cfg.set("symbols", ["BTCUSDT"])
    for dotted, value in (cfg_extra or {}).items():
        cfg.set(dotted, value)
    clock = Clock()
    metrics = Metrics()
    feed = feed_cls(MessageBus(), cfg, clock, metrics)
    monkeypatch.setattr(type(feed._session), "get",
                        lambda self, *a, **kw: FakeResponse(data), raising=True)
    return feed


ALPACA_CRYPTO = {
    "bars": {"BTC/USD": [
        {"t": "2026-01-02T05:05:00Z", "o": 100.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 12},
        {"t": "2026-01-02T05:10:00Z", "o": 105.0, "h": 115.0, "l": 100.0, "c": 112.0, "v": 15},
    ]}
}

TWELVEDATA = {
    "status": "ok",
    "values": [
        {"datetime": "2026-01-02 05:10:00", "open": "105.0", "high": "115.0",
         "low": "100.0", "close": "112.0", "volume": "15"},
        {"datetime": "2026-01-02 05:05:00", "open": "100.0", "high": "110.0",
         "low": "95.0", "close": "105.0", "volume": "12"},
    ],
}

AV_FX = {
    "Time Series FX (5min)": {
        "2026-01-02 05:10:00": {"1. open": "1.05", "2. high": "1.06",
                                "3. low": "1.04", "4. close": "1.055"},
        "2026-01-02 05:05:00": {"1. open": "1.04", "2. high": "1.05",
                                "3. low": "1.03", "4. close": "1.045"},
    }
}

FINAZON = {"data": [
    {"t": 1767330600, "o": 100.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 12},
    {"t": 1767330900, "o": 105.0, "h": 115.0, "l": 100.0, "c": 112.0, "v": 15},
]}


def test_alpaca_fetch_bars(monkeypatch):
    feed = _mk(AlpacaFeed, monkeypatch, ALPACA_CRYPTO)
    events = feed.fetch_bars("BTCUSDT", "M5", limit=10)
    assert len(events) == 2
    assert events[0].payload["open"] == 100.0 and events[-1].payload["close"] == 112.0
    assert events[0].timestamp < events[-1].timestamp  # ascending
    assert feed.map_symbol("XAUUSD") is None  # unmapped -> skipped


def test_alpaca_fx_route(monkeypatch):
    feed = _mk(AlpacaFeed, monkeypatch, {
        "bars": [{"t": "2026-01-02T05:10:00Z", "o": 1.05, "h": 1.06,
                  "l": 1.04, "c": 1.055, "v": 0}]})
    events = feed.fetch_bars("EURUSD", "M5", limit=10)
    assert len(events) == 1
    assert abs(events[0].payload["close"] - 1.055) < 1e-9


def test_twelvedata_parses_and_sorts(monkeypatch):
    feed = _mk(TwelveDataFeed, monkeypatch, TWELVEDATA)
    events = feed.fetch_bars("EURUSD", "M5", limit=10)
    assert [abs(e.payload["close"]) for e in events] == [105.0, 112.0]  # ascending despite API order
    assert feed.map_symbol("XAUUSD") == "XAU/USD"


def test_twelvedata_error_raises(monkeypatch):
    feed = _mk(TwelveDataFeed, monkeypatch, {"status": "error", "message": "bad key"})
    with pytest.raises(RuntimeError):
        feed.fetch_bars("EURUSD", "M5", limit=10)


def test_alphavantage_fx(monkeypatch):
    feed = _mk(AlphaVantageFeed, monkeypatch, AV_FX)
    events = feed.fetch_bars("EURUSD", "M5", limit=10)
    assert len(events) == 2
    assert abs(events[1].payload["close"] - 1.055) < 1e-9


def test_alphavantage_rate_limit_note_raises(monkeypatch):
    feed = _mk(AlphaVantageFeed, monkeypatch, {"Note": "API call frequency limit"})
    with pytest.raises(RuntimeError):
        feed.fetch_bars("EURUSD", "M5", limit=10)


def test_finazon_parses(monkeypatch):
    feed = _mk(FinazonFeed, monkeypatch, FINAZON)
    events = feed.fetch_bars("XAUUSD", "M5", limit=10)
    assert len(events) == 2
    assert abs(events[-1].payload["close"] - 112.0) < 1e-9


def test_run_once_emits_new_bars_only(monkeypatch):
    feed = _mk(AlpacaFeed, monkeypatch, ALPACA_CRYPTO)
    seen = []
    feed.bus.subscribe(None, lambda e: seen.append(e))
    feed.start()
    feed.run_once()
    n_first = len(seen)
    assert n_first == 4  # 2 candles + 2 ticks
    feed.run_once()
    assert len(seen) == n_first  # no new bars -> nothing emitted


def test_fetch_history_ccxt_chunking(tmp_path, monkeypatch):
    """_fetch_ccxt walks paginated OHLCV and returns ascending candle events."""
    ccxt = pytest.importorskip("ccxt")

    import tools.fetch_history as fh

    state = {"calls": 0}

    def fake_fetch_ohlcv(self, pair, timeframe=None, since=None, limit=None):
        state["calls"] += 1
        base = since if since else 0
        return [[base + i * 300000, 1.0 + i, 1.1 + i, 0.9 + i, 1.05 + i, 10 + i]
                for i in range(96)]  # ~2 days of M5

    monkeypatch.setattr(ccxt.binance, "fetch_ohlcv", fake_fetch_ohlcv, raising=True)
    cfg = Config.load()
    events = fh._fetch_ccxt(cfg, ["BTCUSDT"], "M5", start_ms=0, end_ms=1000 * 60 * 60 * 24 * 10)
    assert events and state["calls"] > 1, "pagination did not advance"
    types = {e.type for e in events}
    assert types == {"candle", "tick"}  # companion tick per candle for SL/TP + MTM
    candles = [e for e in events if e.type == "candle"]
    stamps = [e.timestamp for e in candles]
    assert stamps == sorted(stamps)
