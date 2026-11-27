import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


# ---------- Track 1: ISM + Challenger ---------------------------------- #

def test_parse_ism_employment():
    from data_ingestion.news_feeds.ism import parse_ism_employment
    assert parse_ism_employment("The Employment Index registered 47.5 percent") == 47.5
    assert parse_ism_employment("no index here") is None
    assert parse_ism_employment("Employment Index registered 5.0 percent") is None


def test_parse_challenger_cuts():
    from data_ingestion.news_feeds.challenger import parse_challenger_cuts
    assert parse_challenger_cuts("announced 55,000 job cuts in January") == 55000.0
    assert parse_challenger_cuts("no cuts announced") is None
    assert parse_challenger_cuts("announced 5 job cuts") is None


def test_ism_feed_replay_mode_inert():
    from core.config import Config
    from data_ingestion.news_feeds.ism import ISMFeed
    cfg = Config.load()
    feed = ISMFeed(_bus(), cfg, _Clock(virtual=True), _Metrics())
    feed.start()
    feed.on_tick(None)
    assert feed._last_poll == 0.0


def test_challenger_feed_replay_mode_inert():
    from core.config import Config
    from data_ingestion.news_feeds.challenger import ChallengerFeed
    cfg = Config.load()
    feed = ChallengerFeed(_bus(), cfg, _Clock(virtual=True), _Metrics())
    feed.start()
    feed.on_tick(None)
    assert feed._last_poll == 0.0


# ---------- Track 2: Signal log ---------------------------------------- #

def test_signal_log_roundtrip(tmp_path: Path):
    from validation.signal_log import SignalLog
    p = tmp_path / "signals.jsonl"
    log = SignalLog(p, fsync=False)
    log.record("nowcast", {"release_id": "NFP", "mean": 180.0, "stddev": 50.0}, 1.0)
    log.record("release_actual", {"release_id": "NFP", "actual": 195.0}, 2.0)
    log.record("news", {"source": "FRED", "headline": "noise"}, 3.0)
    log.record("news", {"source": "consensus", "consensus": 175.0}, 3.5)
    log.close()

    recs = list(SignalLog(p).replay())
    assert len(recs) == 3
    assert [r["type"] for r in recs] == ["nowcast", "release_actual", "news"]


# ---------- Track 3: Book calibration ---------------------------------- #

def test_calibrate_book_fits_known_params():
    from tools.calibrate_book import fit_params
    mid = 1.1000
    tick = mid * 1e-4 / 2
    base_notional = 20_000.0
    growth = 1.5

    records = []
    for i in range(50):
        bids, asks = [], []
        for k in range(5):
            bid_px = mid - tick * (k + 1)
            ask_px = mid + tick * (k + 1)
            notion = base_notional * (growth ** k)
            bids.append([bid_px, notion / bid_px])
            asks.append([ask_px, notion / ask_px])
        records.append({"ts": float(i), "symbol": "EURUSD", "bids": bids, "asks": asks})

    p = fit_params(records, symbol="EURUSD")
    assert p["n"] == 50
    assert p["tick_size_bps"] == pytest.approx(0.5, rel=0.05)  # half-spread = 0.5 bps for 1bp full spread
    assert p["depth_growth"] == pytest.approx(1.5, rel=0.05)
    assert p["top_size_notional"] == pytest.approx(base_notional, rel=0.05)


def test_calibrate_book_ignores_other_symbols():
    from tools.calibrate_book import fit_params
    records = [
        {"ts": 0.0, "symbol": "EURUSD",
         "bids": [[1.0999, 1000]], "asks": [[1.1001, 1000]]},
        {"ts": 1.0, "symbol": "BTCUSDT",
         "bids": [[50000, 1]], "asks": [[50001, 1]]},
    ]
    p = fit_params(records, symbol="EURUSD")
    assert p["n"] == 1


# ---------- Track 4: alert generation ---------------------------------- #

def test_alert_generation_includes_critical_rules(tmp_path: Path):
    from infra.monitoring import generate_alerts
    text = generate_alerts.render()
    for expected in ("CircuitBreakerTripped", "VetoSignatureRejectedAtL4",
                     "VetoSignatureRejectedAtL5", "DrawdownWarning",
                     "HighOrderRejectRate"):
        assert expected in text
    assert "runbook:" in text


# ---------- Track 5: checkpoint ---------------------------------------- #

def test_checkpoint_roundtrip(tmp_path: Path):
    from core.checkpoint import Checkpoint, CheckpointState
    ck = Checkpoint(tmp_path / "events.checkpoint")
    assert ck.load() is None
    ck.write(CheckpointState(offset=1234, last_ts=99.0, state={"key_id": "v2"}))
    st = ck.load()
    assert st is not None
    assert st.offset == 1234
    assert st.state["key_id"] == "v2"


def test_checkpoint_atomic_write(tmp_path: Path):
    from core.checkpoint import Checkpoint, CheckpointState
    ck = Checkpoint(tmp_path / "events.checkpoint")
    ck.write(CheckpointState(offset=1))
    assert not (tmp_path / "events.checkpoint.tmp").exists()


def test_file_event_log_replay_from_offset(tmp_path: Path):
    from core.event_log import FileEventLog
    p = tmp_path / "events.jsonl"
    log = FileEventLog(p, fsync=False)
    for i in range(5):
        log.append({"type": "tick", "version": 1, "ts": i,
                    "payload": {"symbol": "X", "price": i}})
    log.close()

    log2 = FileEventLog(p, fsync=False)
    assert log2.count() == 5
    resumed = list(log2.replay_from(offset=3))
    assert len(resumed) == 2
    assert resumed[0]["ts"] == 3
    log2.close()


def test_in_memory_log_count_and_replay_from():
    from core.event_log import InMemoryEventLog
    log = InMemoryEventLog()
    for i in range(4):
        log.append({"type": "tick", "version": 1, "ts": i, "payload": {}})
    assert log.count() == 4
    assert len(list(log.replay_from(offset=2))) == 2
