import json
from pathlib import Path
from unittest.mock import patch
import pytest

from data_ingestion.news_feeds.fred import (
    FredClient, FredFeed, SERIES_MAP, Observation, _to_float,
)
from data_ingestion.news_feeds.consensus import (
    ConsensusFeed, Consensus, _RELEASE_TO_FXM,
)
from data_ingestion.news_feeds.retry import HttpClient, _Bucket


# ---------- HttpClient / token bucket ------------------------------------ #

def test_bucket_limits_rate():
    import time
    b = _Bucket(rate_per_sec=100.0, capacity=2.0)
    t0 = time.monotonic()
    for _ in range(5):
        b.take()
    elapsed = time.monotonic() - t0
    # 5 tokens, capacity 2, rate 100/s -> 3 extra tokens / 100 = 30ms min
    assert elapsed >= 0.025


def test_http_disabled_raises():
    h = HttpClient(enabled=False)
    with pytest.raises(RuntimeError):
        h.get("https://example.invalid")


# ---------- FRED --------------------------------------------------------- #

def test_to_float_handles_fred_sentinels():
    assert _to_float(".") is None
    assert _to_float("") is None
    assert _to_float(None) is None
    assert _to_float("3.14") == 3.14
    assert _to_float("-2") == -2.0


def test_series_map_covers_key_releases():
    ids = {meta[0] for meta in SERIES_MAP.values()}
    for required in ("JOBLESS_CLAIMS", "ADP_PRIVATE", "NFP",
                     "CPI_YOY", "UNEMPLOYMENT_RATE", "FED_FUNDS_EFFECTIVE"):
        assert required in ids


def test_fred_client_rejects_short_key():
    with pytest.raises(ValueError):
        FredClient("short")


def test_fred_observations_parses_json():
    fake = {
        "observations": [
            {"date": "2026-01-15", "value": "245.0",
             "realtime_start": "2026-01-15", "realtime_end": "2026-12-31"},
            {"date": "2026-01-08", "value": ".", "realtime_start": "2026-01-08",
             "realtime_end": "2026-12-31"},
        ]
    }
    client = FredClient("x" * 32)
    with patch.object(client, "_get", return_value=fake):
        obs = client.observations("ICSA")
    assert len(obs) == 2
    assert obs[0].value == 245.0
    assert obs[1].value is None


def test_fred_vintages_mutually_exclusive():
    client = FredClient("x" * 32)
    with pytest.raises(ValueError):
        client.vintages("GDPC1", realtime_start="2020-01-01",
                        vintage_dates=["2020-01-30"])


def test_fred_vintage_dates_parses():
    fake = {"vintage_dates": ["2025-01-15", "2025-02-15", "2025-03-15"]}
    client = FredClient("x" * 32)
    with patch.object(client, "_get", return_value=fake):
        dates = client.vintage_dates("GDPC1")
    assert dates == ["2025-01-15", "2025-02-15", "2025-03-15"]


# ---------- FRED feed ---------------------------------------------------- #

class _Clock:
    def __init__(self, virtual=False):
        self._v = virtual
        self._t = 1000.0
    def now(self):
        return self._t
    @property
    def mode(self):
        return "virtual" if self._v else "real"


class _Metrics:
    def __init__(self): self.c = {}
    def inc(self, k, n=1, **__): self.c[k] = self.c.get(k, 0) + n
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def snapshot(self): return dict(self.c)


def _bus():
    from core.bus import MessageBus
    from core.event_log import InMemoryEventLog
    from core.schema_registry import register_default_schemas
    register_default_schemas()
    return MessageBus(event_log=InMemoryEventLog())


def test_fred_feed_no_api_key_is_inert():
    from core.config import Config
    cfg = Config.load()
    bus = _bus()
    feed = FredFeed(bus, cfg, _Clock(), _Metrics())
    feed.start()
    assert feed._client is None          # disabled
    assert feed._running is True


def test_fred_feed_replay_mode_never_calls_network():
    from core.config import Config
    cfg = Config.load()
    cfg.set("data.fred.series", ["ICSA"])
    bus = _bus()
    clock = _Clock(virtual=True)
    feed = FredFeed(bus, cfg, clock, _Metrics())
    feed._client = object()              # pretend it has a client
    feed.start()
    # In virtual mode on_tick must early-return; no exception, no HTTP.
    feed.on_tick(None)
    assert feed._last_poll == 0.0


# ---------- Consensus ----------------------------------------------------- #

def test_consensus_override_wins():
    from core.config import Config
    cfg = Config.load()
    cfg.set("data.consensus.overrides", [
        {"release_id": "CPI_YOY", "consensus": 3.2, "prior": 3.1},
    ])
    feed = ConsensusFeed(_bus(), cfg, _Clock(), _Metrics())
    c = feed.resolve("CPI_YOY")
    assert c.value == 3.2
    assert c.prior == 3.1
    assert c.source == "config"


def test_consensus_no_key_falls_back_to_none():
    from core.config import Config
    cfg = Config.load()
    feed = ConsensusFeed(_bus(), cfg, _Clock(), _Metrics())
    c = feed.resolve("NFP")               # no override, no API key
    assert c.value is None
    assert c.source == "none"


def test_consensus_fxm_parses_response():
    from core.config import Config
    cfg = Config.load()
    feed = ConsensusFeed(_bus(), cfg, _Clock(), _Metrics())
    feed.api_key = "test"
    feed._http = HttpClient()

    fake_resp = type("R", (), {
        "status_code": 200,
        "json": lambda self=None: {
            "data": [
                {"prediction_type": "fxmacrodata", "predicted_value": 3.5,
                 "announcement_id": "usd_inflation_2026-02-28"},
                {"prediction_type": "market_consensus", "predicted_value": 2.9,
                 "announcement_id": "usd_inflation_2026-02-28"},
            ]
        },
    })()

    with patch.object(feed._http, "get", return_value=fake_resp):
        c = feed._fetch_fxm("CPI_YOY")
    assert c is not None
    assert c.value == 2.9                  # prefers market_consensus
    assert c.source == "fxmacrodata"


def test_consensus_mapping_covers_key_releases():
    for required in ("CPI_YOY", "NFP", "UNEMPLOYMENT_RATE", "FOMC_RATE"):
        assert required in _RELEASE_TO_FXM


def test_consensus_emits_news_on_expected():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("data.consensus.overrides", [
        {"release_id": "NFP", "consensus": 180.0, "prior": 175.0},
    ])
    bus = _bus()
    feed = ConsensusFeed(bus, cfg, _Clock(), _Metrics())
    feed.start()

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    bus.publish(make_event(
        "release_expected", "test",
        payload={"release_id": "NFP", "scheduled_ts": 2_000_000.0,
                 "consensus": None, "prior": None, "ts": 1_999_880.0},
        timestamp=1_999_880.0,
    ))
    assert captured, "expected a consensus NEWS event"
    p = captured[0].payload
    assert p["release_id"] == "NFP"
    assert p["consensus"] == 180.0
    assert p["consensus_source"] == "config"
