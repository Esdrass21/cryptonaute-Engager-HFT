"""Phase 3 integration tests: replay harness + tuning meta + real Sharpe."""
import base64
import json
import math
from pathlib import Path
import numpy as np
import pytest

from core.config import Config
from validation.replay_harness import ReplayHarness
from validation.tuning import tune, write_meta, load_meta, pbo_from_results


class _Ev:
    def __init__(self, type_, ts, payload):
        self.type = type_
        self.timestamp = ts
        self.payload = payload
        self.event_id = f"{type_}-{ts}"


def _mk_candle(sym, ts, close):
    return _Ev("candle", ts, {"symbol": sym, "open": close, "high": close,
                              "low": close, "close": close, "volume": 1.0})


def _synth_history(n=600, seed=0):
    rng = np.random.default_rng(seed)
    p = 100.0
    out = []
    for i in range(n):
        p *= (1 + 0.001 + 0.005 * rng.standard_normal())
        out.append(_mk_candle("EURUSD", float(i), p))
    return out


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.load()
    cfg.set("mode", "paper")
    cfg.set("core.keys.provider", "env")
    cfg.set("core.keys.env_var", "RISK_SIGNING_KEY")
    cfg.set("symbols", ["EURUSD"])
    return cfg


# ---------------- Harness -------------------------------------------------- #


def test_harness_equity_returns_aligned_across_trials(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"1" * 32).decode())
    cfg = _cfg(tmp_path)
    events = _synth_history(400)
    h1 = ReplayHarness(cfg, events); h1.run()
    h2 = ReplayHarness(cfg, events); h2.run()
    # Same recorded events -> same number of equity samples.
    assert len(h1.equity_curve()) == len(h2.equity_curve())
    assert len(h1.equity_returns()) == len(h2.equity_returns())


def test_harness_trade_records_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"1" * 32).decode())
    cfg = _cfg(tmp_path)
    events = _synth_history(400)
    h = ReplayHarness(cfg, events); h.run()
    for ts, pnl in h.trade_records():
        assert isinstance(ts, float)
        assert isinstance(pnl, float)


# ---------------- Meta persistence --------------------------------------- #


def test_tune_meta_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"1" * 32).decode())
    cfg = _cfg(tmp_path)
    events = _synth_history(300)
    report = tune(events, cfg, trials=4, min_trades=1, max_dd_pct=50.0,
                  is_frac=0.8, run_oos=True)
    meta_path = write_meta(report, str(tmp_path / "meta.json"))
    loaded = load_meta(meta_path)
    assert loaded.n_trials == 4
    assert math.isclose(loaded.is_frac, 0.8, rel_tol=1e-9)


# ---------------- Sharpe / DSR sanity ------------------------------------- #


def test_tune_returns_real_sharpes_not_all_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"1" * 32).decode())
    cfg = _cfg(tmp_path)
    # Generate data with a strong trend that produces trades
    from data_ingestion.generate_sample import generate
    cfg.set("data.market.sim.trend_strength", 0.002)
    cfg.set("data.market.sim.mean_reversion", 0.003)
    events = generate(cfg, steps=2000, seed=42, trend_strength=0.002, mean_reversion=0.003)
    report = tune(events, cfg, trials=6, min_trades=1, max_dd_pct=50.0, is_frac=0.8)
    # With real returns, trial Sharpes should not all collapse to exactly 0.
    sharpes = [r.is_sharpe for r in report.results]
    assert any(abs(s) > 1e-9 for s in sharpes), (
        "all trial Sharpes are zero — synthetic-return bug is back"
    )


def test_pbo_nan_on_too_few_trials(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_SIGNING_KEY", base64.b64encode(b"1" * 32).decode())
    cfg = _cfg(tmp_path)
    events = _synth_history(200)
    report = tune(events, cfg, trials=1, min_trades=1, max_dd_pct=50.0, is_frac=0.8)
    assert report.pbo != report.pbo  # NaN — not enough columns
