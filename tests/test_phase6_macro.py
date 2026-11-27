import base64
import json
import math
from datetime import UTC, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from analytics.nowcast.nfp import NFPFeatures, NFPNowcast
from data_ingestion.news_feeds.bls import BLSReleaseFeed, parse_release
from data_ingestion.news_feeds.calendar import (
    ReleaseCalendar,
    _et,
    _first_weekday,
)

ET = ZoneInfo("America/New_York")
UTC = UTC


# ---------- Calendar ------------------------------------------------------ #

def test_nfp_is_first_friday_at_0830_et():
    cal = ReleaseCalendar()
    start = _et(2025, 7, 1, 0, 0)
    end   = _et(2025, 7, 31, 23, 59)
    releases = cal.releases_between(start, end)
    nfp = [r for r in releases if r.release_id == "NFP"]
    assert len(nfp) == 1
    dt = datetime.fromtimestamp(nfp[0].scheduled_ts, tz=UTC).astimezone(ET)
    assert dt.year == 2025 and dt.month == 7
    assert dt.weekday() == 4          # Friday
    assert dt.day <= 7
    assert dt.hour == 8 and dt.minute == 30


def test_fomc_dates_are_included():
    cal = ReleaseCalendar()
    start = _et(2025, 1, 1, 0, 0)
    end   = _et(2025, 12, 31, 23, 59)
    fomc = [r for r in cal.releases_between(start, end) if r.release_id == "FOMC_RATE"]
    assert len(fomc) == 8
    # first one is Jan 29 2025, 14:00 ET
    dt = datetime.fromtimestamp(fomc[0].scheduled_ts, tz=UTC).astimezone(ET)
    assert dt.year == 2025 and dt.month == 1 and dt.day == 29
    assert dt.hour == 14 and dt.minute == 0


def test_jobless_claims_on_every_thursday():
    cal = ReleaseCalendar()
    start = _et(2025, 6, 1, 0, 0)
    end   = _et(2025, 6, 30, 23, 59)
    claims = [r for r in cal.releases_between(start, end) if r.release_id == "JOBLESS_CLAIMS"]
    # June 2025 has 4 Thursdays (5, 12, 19, 26)
    assert len(claims) >= 4
    for c in claims:
        day = datetime.fromtimestamp(c.scheduled_ts, tz=UTC).astimezone(ET).day
        assert datetime(2025, 6, day).weekday() == 3


# ---------- Parsers ------------------------------------------------------- #

def test_parse_cpi_yoy():
    html = ("The Consumer Price Index for All Urban Consumers (CPI-U) rose 3.2 percent "
            "over the last 12 months, ...")
    assert parse_release("CPI_YOY", html) == 3.2


def test_parse_nfp():
    text = "Total nonfarm payroll employment rose by 254,000 in September, ..."
    assert parse_release("NFP", text) == 254000.0


def test_parse_unemployment_rate():
    text = "The unemployment rate was 4.1 percent in September, ..."
    assert parse_release("UNEMPLOYMENT_RATE", text) == 4.1


def test_parse_failure_returns_none():
    assert parse_release("NFP", "this is not a release page") is None
    assert parse_release("UNKNOWN_RELEASE", "anything") is None


# ---------- NFP nowcast -------------------------------------------------- #

class _MockClock:
    def __init__(self, t=1_000_000.0):
        self._t = t
        self.mode = "real"
    def now(self):
        return self._t


class _MockMetrics:
    def __init__(self): self.c = {}
    def inc(self, k, n=1, **__): self.c[k] = self.c.get(k, 0) + n
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def snapshot(self): return dict(self.c)


def _make_nfp_nowcast(tmp_path: Path, coef_path: Path | None = None):
    from core.bus import MessageBus
    from core.config import Config
    from core.event_log import InMemoryEventLog
    from core.schema_registry import register_default_schemas

    register_default_schemas()
    cfg = Config.load()
    if coef_path is not None:
        cfg.set("nowcast.nfp.coefficients_path", str(coef_path))

    bus = MessageBus(event_log=InMemoryEventLog())
    return NFPNowcast(bus, cfg, _MockClock(), _MockMetrics()), bus


def test_nfp_nowcast_default_coefficients():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        nc, _ = _make_nfp_nowcast(Path(tmp))
        mean, stddev = nc.predict(nc.features())
        assert stddev > 0
        # with all features zero, mean should be intercept
        assert abs(mean - 150.0) < 1e-6


def test_nfp_nowcast_from_coefficients_file(tmp_path: Path):
    coef = {
        "coefficients": {
            "intercept": 100.0,
            "jobless_claims_4w_avg": -0.5,
            "jobless_claims_change": 0.0,
            "adp_private": 0.0,
            "jolts_hires": 0.0,
            "ism_manufacturing_emp": 0.0,
            "ism_services_emp": 0.0,
            "challenger_cuts": 0.0,
            "prior_nfp": 0.0,
            "temp_help_change": 0.0,
        },
        "rmse": 50.0,
        "regime_scale": 2.0,
    }
    p = tmp_path / "coef.json"
    p.write_text(json.dumps(coef))
    nc, _ = _make_nfp_nowcast(tmp_path, coef_path=p)
    nc.feature_state.update("jobless_claims_4w_avg", 220.0)
    mean, stddev = nc.predict(nc.features())
    # 100 + (-0.5 * 220) = -10
    assert abs(mean - (-10.0)) < 1e-6
    assert abs(stddev - 100.0) < 1e-6     # 50 * 2.0


def test_nfp_nowcast_coverage_tracks_inputs():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        nc, _ = _make_nfp_nowcast(Path(tmp))
        assert nc.feature_state.coverage() == 0.0
        nc.feature_state.update("jobless_claims_4w_avg", 220.0)
        assert nc.feature_state.coverage() == pytest.approx(1 / 9)
        for f in ["adp_private", "prior_nfp", "ism_services_emp"]:
            nc.feature_state.update(f, 1.0)
        assert nc.feature_state.coverage() == pytest.approx(4 / 9)


# ---------- End-to-end through the bus ---------------------------------- #

def test_release_expected_triggers_nowcast(tmp_path: Path):
    nc, bus = _make_nfp_nowcast(tmp_path)
    nc.start()
    nc.feature_state.update("jobless_claims_4w_avg", 220.0)
    nc.feature_state.update("prior_nfp", 180.0)

    captured = []
    bus.subscribe("nowcast", lambda e: captured.append(e))

    from core.events import make_event
    bus.publish(make_event(
        "release_expected", "test",
        payload={"release_id": "NFP", "scheduled_ts": 1_000_060.0,
                 "consensus": 175.0, "prior": 180.0, "ts": 1_000_000.0},
        timestamp=1_000_000.0,
    ))

    assert captured, "expected a NOWCAST event"
    p = captured[0].payload
    assert p["release_id"] == "NFP"
    assert "mean" in p and "stddev" in p
    assert p["coverage"] > 0
