# tests/test_strategy_lab.py
"""Smoke tests for the vectorized strategy lab."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.strategy_lab import (
    COST_LEVELS,
    build_position,
    extract_trades,
    load_candles,
    sig_tsmom,
    simulate,
    stats_from_returns,
    strategy_grid,
)


def _write_history(tmp_path: Path, n: int = 2200) -> Path:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts = ((idx - epoch) / pd.Timedelta(1, "s")).astype(float)
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.02, n))
    path = tmp_path / "hist.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            o = close[i - 1] if i else close[0]
            rec = {
                "timestamp": float(ts[i]), "type": "candle", "source": "test",
                "payload": {"symbol": "TESTUSDT", "timeframe": "M5", "open": float(o),
                            "high": float(max(o, close[i]) + 0.01),
                            "low": float(min(o, close[i]) - 0.01),
                            "close": float(close[i]), "volume": 1.0, "closed": True},
            }
            fh.write(json.dumps(rec) + "\n")
    return path


import json  # noqa: E402


def test_load_and_simulate_buyhold(tmp_path: Path):
    path = _write_history(tmp_path)
    candles = load_candles(path)
    assert list(candles) == ["TESTUSDT"]
    df = candles["TESTUSDT"]
    close = df["close"].to_numpy()
    pos = np.ones(len(df))
    r, held = simulate(close, pos, 0.0)
    np.testing.assert_allclose(held[1:], 1.0)
    total = float(np.prod(1.0 + r))
    np.testing.assert_allclose(total, close[-1] / close[0], rtol=1e-9)
    st = stats_from_returns(r, df.index)
    assert st["sharpe"] != 0.0


def test_costs_reduce_returns_and_create_trades(tmp_path: Path):
    path = _write_history(tmp_path)
    df = load_candles(path)["TESTUSDT"]
    close = df["close"].to_numpy()
    sig = np.ones(len(df))
    sig[1000:] = -1.0  # deterministic flip: one open, one close
    r0, _ = simulate(close, sig, 0.0)
    r10, held = simulate(close, sig, 10.0)
    assert float(np.sum(r10)) <= float(np.sum(r0))
    trades = extract_trades(held, close, df.index, 10.0)
    assert len(trades) == 2
    for t in trades:
        assert t["net_pct_at_cost"] <= t["gross_pct"]


def test_position_is_lagged_no_lookahead(tmp_path: Path):
    path = _write_history(tmp_path)
    df = load_candles(path)["TESTUSDT"]
    pos_target = build_position("tsmom_mop", dict(fn=sig_tsmom, lookback=100, vol_target=0.5), df)
    r, held = simulate(df["close"].to_numpy(), pos_target, 0.0)
    # position held on bar i must equal target on bar i-1 (first bar flat)
    tgt = np.nan_to_num(pos_target, nan=0.0)
    np.testing.assert_allclose(held[1:], np.clip(tgt[:-1], -1.0, 1.0))
    assert held[0] == 0.0


def test_strategy_grid_complete():
    grid = strategy_grid()
    names = {name for name, _, _ in grid}
    expected = {
        "buy_hold", "tsmom_mop", "tsmom_blend_hop", "sma_cross", "macd",
        "stoch_cross", "donchian", "bollinger_mr", "rsi2", "keltner_breakout",
        "vwap_reversion", "noise_band",
    }
    assert expected <= names
    assert len(grid) >= 25


def test_costs_cover_expected_levels():
    assert COST_LEVELS == [0.0, 1.0, 5.0, 10.0]
