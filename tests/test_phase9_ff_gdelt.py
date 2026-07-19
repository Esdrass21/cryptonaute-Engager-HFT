from pathlib import Path
from unittest.mock import patch
import pytest


class _Clock:
    def __init__(self, t=1000.0, virtual=False):
        self._t = float(t); self._v = virtual
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


# ---------- Forex Factory parsing -------------------------------------- #

_FF_SAMPLE = """[
  {"title":"Non-Farm Employment Change","country":"USD",
   "date":"2026-02-06T08:30:00-05:00","impact":"High",
   "forecast":"180K","previous":"175K"},
  {"title":"CPI m/m","country":"USD",
   "date":"2026-02-12T08:30:00-05:00","impact":"High",
   "forecast":"0.3%","previous":"0.2%"},
  {"title":"Bank Holiday","country":"USD",
   "date":"2026-02-16T00:00:00-05:00","impact":"Holiday",
   "forecast":"","previous":""},
  {"title":"Some Unmapped Event","country":"USD",
   "date":"2026-02-20T10:00:00-05:00","impact":"High",
   "forecast":"","previous":""}
]"""


def test_parse_ff_json_basic():
    from data_ingestion.news_feeds.forexfactory import parse_ff_json
    evs = parse_ff_json(_FF_SAMPLE)
    assert len(evs) == 4
    nfp = next(e for e in evs if e.title == "Non-Farm Employment Change")
    assert nfp.country == "USD"
    assert nfp.impact == "High"
    assert nfp.forecast == 180.0
    assert nfp.previous == 175.0
    assert nfp.ts > 0


def test_parse_ff_num_handles_suffixes():
    from data_ingestion.news_feeds.forexfactory import _parse_num
    assert _parse_num("5.50%") == 5.50
    assert _parse_num("180K") == 180.0
    assert _parse_num("1.2M") == 1200.0
    assert _parse_num("0.3") == 0.3
    assert _parse_num("") is None
    assert _parse_num("-") is None


def test_parse_ff_ts_with_offset():
    from data_ingestion.news_feeds.forexfactory import _parse_iso
    ts = _parse_iso("2026-02-06T08:30:00-05:00")
    assert ts is not None
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.hour == 13 and dt.minute == 30


def test_parse_ff_json_tolerates_bad_records():
    from data_ingestion.news_feeds.forexfactory import parse_ff_json
    assert parse_ff_json("not json") == []
    assert parse_ff_json('{"a":1}') == []
    assert parse_ff_json('[{"title":"","country":""}]') == []
    assert parse_ff_json('[{"title":"X","country":"USD","date":"bogus"}]') == []


def test_ff_feed_emits_news_for_mapped_titles():
    from core.config import Config
    from data_ingestion.news_feeds.forexfactory import ForexFactoryFeed
    cfg = Config.load()
    cfg.set("data.news.forexfactory.countries", ["USD"])
    cfg.set("data.news.forexfactory.min_impact", "High")
    bus = _bus()
    feed = ForexFactoryFeed(bus, cfg, _Clock(), _Metrics())

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _FF_SAMPLE

    class _Http:
        def get(self, url, timeout=10.0):
            return _Resp()

    feed._http = _Http()
    feed._poll()

    rel_ids = [e.payload.get("release_id") for e in captured]
    assert "NFP" in rel_ids
    assert "CPI_MOM" in rel_ids
    titles = [e.payload.get("impact") for e in captured]
    assert "Holiday" not in titles


def test_ff_feed_replay_mode_inert():
    from core.config import Config
    from data_ingestion.news_feeds.forexfactory import ForexFactoryFeed
    cfg = Config.load()
    feed = ForexFactoryFeed(_bus(), cfg, _Clock(virtual=True), _Metrics())
    feed.start()
    feed.on_tick(None)
    assert feed._last_poll == 0.0


def test_ff_feed_unmapped_title_counted():
    from core.config import Config
    from data_ingestion.news_feeds.forexfactory import ForexFactoryFeed
    cfg = Config.load()
    cfg.set("data.news.forexfactory.min_impact", "High")
    bus = _bus()
    m = _Metrics()
    feed = ForexFactoryFeed(bus, cfg, _Clock(), m)

    class _Resp:
        status_code = 200
        text = _FF_SAMPLE

    class _Http:
        def get(self, url, timeout=10.0):
            return _Resp()

    feed._http = _Http()
    feed._poll()
    assert m.snapshot().get("ff_unmapped_title", 0) >= 1


def test_ff_feed_filters_by_country():
    from core.config import Config
    from data_ingestion.news_feeds.forexfactory import ForexFactoryFeed
    cfg = Config.load()
    cfg.set("data.news.forexfactory.countries", ["EUR"])
    cfg.set("data.news.forexfactory.min_impact", "High")
    bus = _bus()
    feed = ForexFactoryFeed(bus, cfg, _Clock(), _Metrics())

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _FF_SAMPLE

    class _Http:
        def get(self, url, timeout=10.0):
            return _Resp()

    feed._http = _Http()
    feed._poll()
    assert captured == []


# ---------- GDELT parsing ---------------------------------------------- #

_GDELT_SAMPLE = """{
  "articles": [
    {"url":"https://example.com/a1","title":"Fed holds rates steady",
     "domain":"example.com","language":"English","sourcecountry":"United States",
     "seendate":"20260206T143000Z","socialimage":""},
    {"url":"https://example.com/a2","title":"ECB signals cut",
     "domain":"example.com","language":"English","sourcecountry":"Germany",
     "seendate":"20260206T143500Z","socialimage":""},
    {"url":"","title":"broken","domain":"x","language":"English",
     "sourcecountry":"US","seendate":"20260206T143500Z"}
  ]
}"""


def test_parse_gdelt_artlist_basic():
    from data_ingestion.news_feeds.gdelt import parse_gdelt_artlist
    arts = parse_gdelt_artlist(_GDELT_SAMPLE)
    assert len(arts) == 2
    assert arts[0].title == "Fed holds rates steady"
    assert arts[0].domain == "example.com"
    assert arts[0].ts > 0


def test_parse_gdelt_artlist_tolerates_garbage():
    from data_ingestion.news_feeds.gdelt import parse_gdelt_artlist
    assert parse_gdelt_artlist("not json") == []
    assert parse_gdelt_artlist('{"articles": "not a list"}') == []
    assert parse_gdelt_artlist('{"articles": [{"title":"x"}]}') == []


def test_gdelt_ts_parsing():
    from data_ingestion.news_feeds.gdelt import parse_gdelt_ts
    ts = parse_gdelt_ts("20260206T143000Z")
    assert ts is not None
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.year == 2026 and dt.month == 2 and dt.day == 6
    assert dt.hour == 14 and dt.minute == 30


def test_gdelt_feed_replay_mode_inert():
    from core.config import Config
    from data_ingestion.news_feeds.gdelt import GDELTFeed
    cfg = Config.load()
    cfg.set("data.news.gdelt.enabled", True)
    cfg.set("data.news.gdelt.queries",
            [{"id": "T", "q": "test", "symbols": ["EURUSD"]}])
    feed = GDELTFeed(_bus(), cfg, _Clock(virtual=True), _Metrics())
    feed.start()
    feed.on_tick(None)
    assert feed._last_poll == 0.0


def test_gdelt_feed_emits_news_for_query():
    from core.config import Config
    from data_ingestion.news_feeds.gdelt import GDELTFeed
    cfg = Config.load()
    cfg.set("data.news.gdelt.enabled", True)
    cfg.set("data.news.gdelt.queries",
            [{"id": "FED", "q": "fomc", "symbols": ["EURUSD"]}])
    bus = _bus()
    feed = GDELTFeed(bus, cfg, _Clock(), _Metrics())

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _GDELT_SAMPLE

    class _Http:
        def get(self, url, params=None, timeout=15.0):
            return _Resp()

    feed._http = _Http()
    feed._poll_query({"id": "FED", "q": "fomc", "symbols": ["EURUSD"]})

    assert len(captured) == 2
    for e in captured:
        assert e.payload["source"] == "GDELT"
        assert e.payload["query_id"] == "FED"
        assert e.payload["symbols"] == ["EURUSD"]


def test_gdelt_feed_dedupes_urls():
    from core.config import Config
    from data_ingestion.news_feeds.gdelt import GDELTFeed
    cfg = Config.load()
    cfg.set("data.news.gdelt.enabled", True)
    cfg.set("data.news.gdelt.queries",
            [{"id": "FED", "q": "fomc", "symbols": ["EURUSD"]}])
    bus = _bus()
    feed = GDELTFeed(bus, cfg, _Clock(), _Metrics())

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _GDELT_SAMPLE

    class _Http:
        def get(self, url, params=None, timeout=15.0):
            return _Resp()

    feed._http = _Http()
    feed._poll_query({"id": "FED", "q": "fomc", "symbols": ["EURUSD"]})
    feed._poll_query({"id": "FED", "q": "fomc", "symbols": ["EURUSD"]})
    assert len(captured) == 2
