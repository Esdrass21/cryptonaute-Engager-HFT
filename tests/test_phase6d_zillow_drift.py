import json, math
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from data_ingestion.news_feeds.zillow import (
    parse_zori_csv, zori_yoy, parse_manheim_html, ZillowFeed, _month_minus_12,
)
from validation.drift_analysis import (
    _load_events, _normalise, build_price_series, analyse, suggest_rules,
    DriftBucket, DEFAULT_HORIZONS_S,
)


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


# ---------- ZORI parsing ------------------------------------------------- #

def _make_zori_csv():
    dates = [f"2025-{m:02d}-01" for m in range(1, 13)] + ["2026-01-01"]
    header = "RegionID,SizeRank,RegionName,RegionType,StateName," + ",".join(dates)
    jan25 = 100.0
    values = [jan25 + (i * 0.4) for i in range(12)] + [105.0]
    row = "1,1,United States,country,," + ",".join(str(v) for v in values)
    other = "2,2,Austin,tx,US," + ",".join("200" for _ in dates)
    return f"{header}\n{row}\n{other}\n"


def test_parse_zori_csv_extracts_us_row():
    points = parse_zori_csv(_make_zori_csv(), region="United States")
    assert len(points) == 13
    assert points[0].date == "2025-01-01"
    assert points[-1].date == "2026-01-01"
    assert points[-1].value == 105.0


def test_zori_yoy_computes_correctly():
    points = parse_zori_csv(_make_zori_csv(), region="United States")
    yoy = zori_yoy(points)
    # 105 vs 100 = 5%
    assert yoy == pytest.approx(5.0, rel=1e-3)


def test_zori_yoy_returns_none_on_insufficient_history():
    points = parse_zori_csv(_make_zori_csv(), region="United States")[:5]
    assert zori_yoy(points) is None


def test_month_minus_12():
    assert _month_minus_12("2026-01-01") == "2025-01-01"
    assert _month_minus_12("2025-03-01") == "2024-03-01"


# ---------- Manheim parsing --------------------------------------------- #

def test_parse_manheim_html_finds_index():
    html = "<p>The Manheim Used Vehicle Value Index was 208.4 in January.</p>"
    assert parse_manheim_html(html) == 208.4


def test_parse_manheim_html_none_when_missing():
    assert parse_manheim_html("<p>no number here</p>") is None


# ---------- Zillow feed -------------------------------------------------- #

def test_zillow_feed_replay_mode_inert():
    from core.config import Config
    cfg = Config.load()
    bus = _bus()
    feed = ZillowFeed(bus, cfg, _Clock(virtual=True), _Metrics())
    feed.start()
    feed.on_tick(None)
    assert feed._last_poll == 0.0


def test_zillow_feed_emits_news_on_new_value():
    from core.config import Config
    cfg = Config.load()
    bus = _bus()
    clock = _Clock()
    feed = ZillowFeed(bus, cfg, clock, _Metrics())
    feed._http = MagicMock()

    captured = []
    bus.subscribe("news", lambda e: captured.append(e))

    class _Resp:
        status_code = 200
        text = _make_zori_csv()
    feed._http.get.return_value = _Resp()
    feed._poll_zori()

    assert captured
    assert captured[0].payload["release_id"] == "ZORI_YOY"
    assert captured[0].payload["actual"] == pytest.approx(5.0, rel=1e-3)


# ---------- Drift analysis ----------------------------------------------- #

def _synth_events_with_drift(
    n_releases: int = 24,
    drift_bps: float = 8.0,
    surprise_std: float = 1.5,
    seed: int = 0,
) -> list[dict]:
    """Build a synthetic event stream with releases + candles + drift."""
    import numpy as np
    rng = np.random.default_rng(seed)

    events: list[dict] = []
    px = 1.1000
    t0 = 1_700_000_000.0
    minute = 60.0
    hour = 3600.0

    release_ts = [t0 + i * 30 * 86400 for i in range(n_releases)]
    surprises = rng.normal(0.0, surprise_std, size=n_releases)

    end_ts = release_ts[-1] + 6 * hour

    t = t0
    while t <= end_ts:
        for rts, sz in zip(release_ts, surprises):
            dt = t - rts
            if 60.0 <= dt <= hour and sz != 0.0:
                decay = max(0.0, 1.0 - (dt - 60.0) / (hour - 60.0))
                drift = (drift_bps * sz / 10000.0) * decay * (minute / hour)
                px += drift
        px *= (1.0 + rng.normal(0.0, 0.00005))
        events.append({
            "type": "candle",
            "ts": t,
            "payload": {"symbol": "EURUSD", "close": px,
                        "open": px, "high": px, "low": px, "volume": 1.0},
        })
        t += minute

    for rts, sz in zip(release_ts, surprises):
        events.append({
            "type": "release_actual",
            "ts": rts,
            "payload": {"release_id": "CPI_YOY", "actual": 3.0 + sz,
                        "consensus": 3.0, "surprise": sz},
        })
    events.sort(key=lambda e: e["ts"])
    return events


def test_load_and_normalise_roundtrip(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    rows = [
        {"type": "candle", "ts": 1.0, "payload": {"symbol": "X", "close": 100.0}},
        {"type": "release_actual", "ts": 2.0,
         "payload": {"release_id": "CPI_YOY", "surprise": 1.0}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    evs = _load_events(p)
    assert len(evs) == 2
    n = _normalise(evs[0])
    assert n["type"] == "candle" and n["ts"] == 1.0


def test_price_series_construction():
    evs = _synth_events_with_drift(n_releases=3)
    series = build_price_series(evs, ["EURUSD"])
    assert "EURUSD" in series
    s = series["EURUSD"]
    assert len(s.ts) == len(s.px)
    assert all(s.ts[i] <= s.ts[i + 1] for i in range(len(s.ts) - 1))


def test_drift_analysis_detects_synthetic_edge():
    evs = _synth_events_with_drift(n_releases=24, drift_bps=8.0, seed=1)
    buckets = analyse(evs, ["EURUSD"],
                      horizons=[900.0, 1800.0, 3600.0, 7200.0],
                      min_obs=8)
    assert buckets, "no buckets produced"
    best = max(buckets, key=lambda b: b.sharpe_annualised)
    assert best.sharpe_annualised > 0.3
    assert best.hit_rate > 0.5
    assert best.n >= 8


def test_suggest_rules_produces_valid_drift_rule():
    evs = _synth_events_with_drift(n_releases=24, drift_bps=10.0, seed=2)
    buckets = analyse(evs, ["EURUSD"],
                      horizons=[900.0, 1800.0, 3600.0, 7200.0],
                      min_obs=8)
    rules = suggest_rules(buckets, min_sharpe=0.2, min_obs=8)
    assert "CPI_YOY" in rules
    r = rules["CPI_YOY"]
    for k in ("symbol", "polarity", "holding_s", "sl_pct", "tp_pct", "position_pct"):
        assert k in r
    assert r["polarity"] in (-1, 1)
    assert r["holding_s"] > 0
    assert 0.0 < r["sl_pct"] <= 5.0
    assert 0.0 < r["tp_pct"] <= 10.0


def test_no_drift_produces_no_suggestions():
    """With zero drift, nothing should clear a positive Sharpe threshold."""
    evs = _synth_events_with_drift(n_releases=24, drift_bps=0.0, seed=3)
    buckets = analyse(evs, ["EURUSD"], horizons=[1800.0], min_obs=8)
    rules = suggest_rules(buckets, min_sharpe=1.5, min_obs=8)
    assert len(rules) == 0
