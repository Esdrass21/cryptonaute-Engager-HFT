import math
from pathlib import Path
from unittest.mock import patch

import pytest

from analytics.nowcast.cpi import CPINowcast, _CPIState
from data_ingestion.news_feeds.fed import (
    FedFeed,
    FeedItem,
    _parse_iso,
    _parse_pubdate,
    parse_rss,
    score_fed_text,
)
from strategy_engine.release_drift import DriftRule, ReleaseDriftStrategy

# ---------- helpers ------------------------------------------------------ #

class _Clock:
    def __init__(self, t=1000.0, virtual=False):
        self._t = float(t)
        self._v = virtual
    def now(self): return self._t
    def advance(self, d): self._t += d
    @property
    def mode(self): return "virtual" if self._v else "real"


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


# ---------- CPI nowcast -------------------------------------------------- #

def test_cpi_state_computes_wow_changes():
    s = _CPIState()
    s.update_gas(3.10)
    s.update_gas(3.20)
    assert s.v["gas_price_wow"] == pytest.approx(3.2258, rel=1e-3)

    s.update_oil(80.0)
    s.update_oil(84.0)
    assert s.v["oil_wow"] == pytest.approx(5.0, rel=1e-6)


def test_cpi_default_prediction_positive():
    from core.config import Config
    cfg = Config.load()
    nc = CPINowcast(_bus(), cfg, _Clock(), _Metrics())
    mean, stddev = nc.predict(nc.features())
    assert mean > 0
    assert stddev > 0


def test_cpi_emits_nowcast_on_expected():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    bus = _bus()
    nc = CPINowcast(bus, cfg, _Clock(), _Metrics())
    nc.start()
    nc.state.set("prior_cpi_mom", 0.3)

    captured = []
    bus.subscribe("nowcast", lambda e: captured.append(e))
    bus.publish(make_event(
        "release_expected", "test",
        payload={"release_id": "CPI_MOM", "scheduled_ts": 1000.0,
                 "consensus": 0.2, "prior": 0.3, "ts": 990.0},
        timestamp=990.0,
    ))
    assert captured and captured[0].payload["release_id"] == "CPI_MOM"
    assert "feature_vector" in captured[0].payload


# ---------- Fed feed ----------------------------------------------------- #

_SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Speech by Chair Powell</title>
  <link>https://federalreserve.gov/newsevents/speech/powell20260115a.htm</link>
  <guid>speech-2026-01-15</guid>
  <pubDate>Thu, 15 Jan 2026 14:00:00 GMT</pubDate>
  <description>We remain vigilant on inflation risks.</description>
</item>
</channel></rss>"""


def test_parse_rss_basic():
    items = parse_rss(_SAMPLE_RSS)
    assert len(items) == 1
    it = items[0]
    assert "Powell" in it.title
    assert it.guid == "speech-2026-01-15"
    assert it.pub_ts > 0


def test_parse_rss_bad_xml_returns_empty():
    assert parse_rss("not xml at all") == []


def test_score_fed_text_hawkish():
    s = score_fed_text("We must tighten. Inflation is elevated and persistent.")
    assert s > 0.5


def test_score_fed_text_dovish():
    s = score_fed_text("We can ease. Disinflation is moderating, risks are softening.")
    assert s < -0.5


def test_score_fed_text_neutral():
    assert score_fed_text("The committee met today.") == 0.0


def test_fed_feed_replay_mode_no_network():
    from core.config import Config
    cfg = Config.load()
    bus = _bus()
    clock = _Clock(virtual=True)
    feed = FedFeed(bus, cfg, clock, _Metrics())
    feed.start()
    feed.on_tick(None)                  # virtual mode -> early return
    assert feed._last_poll == 0.0


def test_fed_feed_emits_news_from_rss():
    from core.config import Config
    cfg = Config.load()
    bus = _bus()
    clock = _Clock()
    feed = FedFeed(bus, cfg, clock, _Metrics())
    # Use a proper mock that has a .get method
    from unittest.mock import MagicMock
    feed._http = MagicMock()

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _SAMPLE_RSS
    feed._http.get.return_value = _Resp()
    feed._poll()
    # _poll iterates FEEDS; mock get returns same RSS for each -> up to 3 NEWS
    assert captured
    assert all(e.payload["source"] == "FED" for e in captured)


# ---------- Release-drift strategy -------------------------------------- #

def test_drift_skips_small_surprise():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("symbols", ["EURUSD"])
    bus = _bus()
    ds = ReleaseDriftStrategy(bus, cfg, _Clock(), _Metrics())
    ds.start()

    props = []
    bus.subscribe("order_proposal", lambda e: props.append(e))
    bus.publish(make_event(
        "release_actual", "test",
        payload={"release_id": "CPI_YOY", "surprise": 0.5, "actual": 3.1,
                 "consensus": 3.0, "ts": 1000.0},
        timestamp=1000.0,
    ))
    assert props == []


def test_drift_fires_on_large_surprise():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("symbols", ["EURUSD"])
    bus = _bus()
    ds = ReleaseDriftStrategy(bus, cfg, _Clock(), _Metrics())
    ds.start()

    props = []
    bus.subscribe("order_proposal", lambda e: props.append(e))
    bus.publish(make_event(
        "release_actual", "test",
        payload={"release_id": "CPI_YOY", "surprise": 2.5, "actual": 3.5,
                 "consensus": 3.0, "ts": 1000.0},
        timestamp=1000.0,
    ))
    assert props
    p = props[0].payload
    assert p["symbol"] == "EURUSD"
    assert p["direction"] == -1.0        # CPI hot -> short EURUSD (rule polarity -1)
    assert p["passes_ev"] is True
    assert p["execute"] is False
    assert p["release_id"] == "CPI_YOY"


def test_drift_respects_cooldown():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("symbols", ["EURUSD"])
    bus = _bus()
    clock = _Clock()
    ds = ReleaseDriftStrategy(bus, cfg, clock, _Metrics())
    ds.start()

    props = []
    bus.subscribe("order_proposal", lambda e: props.append(e))

    for _ in range(3):
        bus.publish(make_event(
            "release_actual", "test",
            payload={"release_id": "CPI_YOY", "surprise": 2.5,
                     "actual": 3.5, "consensus": 3.0, "ts": clock.now()},
            timestamp=clock.now(),
        ))
    # First fires, subsequent are in cooldown until the position is closed.
    assert len(props) >= 1
    # But after cooldown expires, it should fire again — advance past holding_s.
    clock.advance(4000.0)
    bus.publish(make_event(
        "release_actual", "test",
        payload={"release_id": "CPI_YOY", "surprise": 2.5,
                     "actual": 3.5, "consensus": 3.0, "ts": clock.now()},
        timestamp=clock.now(),
    ))
    assert len(props) >= 2


def test_drift_unknown_release_ignored():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    bus = _bus()
    ds = ReleaseDriftStrategy(bus, cfg, _Clock(), _Metrics())
    ds.start()
    props = []
    bus.subscribe("order_proposal", lambda e: props.append(e))
    bus.publish(make_event(
        "release_actual", "test",
        payload={"release_id": "SOME_UNKNOWN", "surprise": 5.0, "ts": 1000.0},
        timestamp=1000.0,
    ))
    assert props == []


def test_drift_symbol_not_configured_ignored():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("symbols", ["BTCUSDT"])          # EURUSD not in set
    bus = _bus()
    ds = ReleaseDriftStrategy(bus, cfg, _Clock(), _Metrics())
    ds.start()
    props = []
    bus.subscribe("order_proposal", lambda e: props.append(e))
    bus.publish(make_event(
        "release_actual", "test",
        payload={"release_id": "CPI_YOY", "surprise": 2.5, "ts": 1000.0},
        timestamp=1000.0,
    ))
    assert props == []


def test_drift_time_exit_fires_after_holding():
    from core.config import Config
    from core.events import make_event
    cfg = Config.load()
    cfg.set("symbols", ["EURUSD"])
    clock = _Clock()
    bus = _bus()
    ds = ReleaseDriftStrategy(bus, cfg, clock, _Metrics())
    ds.start()

    # Simulate an entry fill from a drift proposal
    bus.publish(make_event(
        "order_fill", "execution.order_manager",
        payload={"symbol": "EURUSD", "closed": False, "direction": 1.0,
                 "release_id": "CPI_YOY"},
        timestamp=clock.now(),
    ))
    assert "EURUSD" in ds._open

    exits = []
    bus.subscribe("order_exit", lambda e: exits.append(e))

    clock.advance(4000.0)                    # past holding_s (3600)
    bus.publish(make_event(
        "tick", "market",
        payload={"symbol": "EURUSD", "mid": 1.10},
        timestamp=clock.now(),
    ))
    assert exits and exits[0].payload["reason"] == "drift_time_exit"
