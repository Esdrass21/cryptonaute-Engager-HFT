"""Tests for the overnight swap / rollover financing model."""
from __future__ import annotations

import calendar
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from execution.broker_adapter import PaperBroker
from execution.financing import (
    SwapRate,
    SwapSchedule,
    default_schedule,
    financing_series,
    infer_family,
    load_swap_schedule,
    rollover_flags,
    schedule_from_mapping,
    summarize_financing,
)
from validation.strategy_lab import extract_trades, simulate

DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]  # Mon..Thu


def _rollover_index(days=DAYS) -> pd.DatetimeIndex:
    stamps = []
    for day in days:
        stamps.extend([
            pd.Timestamp(f"{day} 20:55", tz="UTC"),
            pd.Timestamp(f"{day} 21:00", tz="UTC"),
            pd.Timestamp(f"{day} 21:05", tz="UTC"),
        ])
    return pd.DatetimeIndex(stamps)


def test_rollover_flags_one_per_day_at_or_after_cut():
    idx = _rollover_index()
    flags = rollover_flags(idx, rollover_hour_utc=21)
    assert flags.sum() == len(DAYS)
    picked = idx[flags]
    assert list(picked.hour) == [21, 21, 21, 21]
    assert [d.day for d in picked] == [5, 6, 7, 8]


def test_financing_uses_side_rate_and_triples_wednesday():
    sched = SwapSchedule(
        families={"fx": SwapRate(1.0, 5.0)},
        triple_swap_weekday=2,  # Wednesday
    )
    idx = _rollover_index()
    pos = np.ones(len(idx))
    fin = financing_series(idx, pos, sched, "EURUSD")
    # Mon + Tue + Wed(x3) + Thu = 6 nights at 1 bps of notional.
    np.testing.assert_allclose(fin.sum(), 6.0 / 1e4)

    short = financing_series(idx, -pos, sched, "EURUSD")
    # Short rate is 5 bps: 1 + 1 + 3 + 1 = 6 * 5 bps.
    np.testing.assert_allclose(short.sum(), 30.0 / 1e4)


def test_flat_and_disabled_positions_pay_nothing():
    idx = _rollover_index()
    sched = default_schedule()
    np.testing.assert_allclose(financing_series(idx, np.zeros(len(idx)), sched, "EURUSD"), 0.0)
    off = SwapSchedule(families={"fx": SwapRate(10.0, 10.0)}, enabled=False)
    np.testing.assert_allclose(financing_series(idx, np.ones(len(idx)), off, "EURUSD"), 0.0)


def test_rollover_charges_the_held_position_on_that_bar():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-05 21:00", tz="UTC")])
    sched = SwapSchedule(families={"fx": SwapRate(10.0, 10.0)})
    np.testing.assert_allclose(financing_series(idx, np.ones(1), sched, "EURUSD"), 10.0 / 1e4)
    np.testing.assert_allclose(financing_series(idx, np.zeros(1), sched, "EURUSD"), 0.0)


def test_family_inference_and_symbol_rate_override():
    assert infer_family("EURUSD") == "fx"
    assert infer_family("XAUUSD") == "metal"
    assert infer_family("USOIL") == "energy"
    assert infer_family("BTCUSD") == "crypto"
    sched = schedule_from_mapping({
        "families": {"fx": {"long_bps": 1.0, "short_bps": 2.0}},
        "symbols": {"EURUSD": [9.0, 8.0]},
    })
    assert sched.rate_for("EURUSD") == SwapRate(9.0, 8.0)
    assert sched.rate_for("GBPUSD") == SwapRate(1.0, 2.0)


def test_schedule_from_full_config_unwraps_execution_swap(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "execution:\n"
        "  swap:\n"
        "    enabled: true\n"
        "    rollover_hour_utc: 22\n"
        "    families:\n"
        "      fx: { long_bps: 0.5, short_bps: 0.25 }\n",
        encoding="utf-8",
    )
    sched = load_swap_schedule(path)
    assert sched.enabled and sched.rollover_hour_utc == 22
    assert sched.rate_for("EURUSD") == SwapRate(0.5, 0.25)


def test_simulate_with_financing_reduces_return():
    n = 40
    idx = pd.date_range("2026-01-05 20:55", periods=n, freq="5min", tz="UTC")
    close = np.full(n, 100.0)  # no price move: only financing can create P&L
    pos = np.ones(n)
    fin = np.zeros(n)
    fin[10] = 0.001
    r, held = simulate(close, pos, 0.0, financing=fin)
    np.testing.assert_allclose(held[1:], 1.0)
    np.testing.assert_allclose(r.sum(), -0.001)

    sched = SwapSchedule(families={"fx": SwapRate(1.0, 1.0)})
    r2, _ = simulate(close, pos, 0.0, index=idx, swap_schedule=sched, symbol="EURUSD")
    np.testing.assert_allclose(r2.sum(), financing_series(idx, pos, sched, "EURUSD").sum() * -1.0)


def test_extract_trades_charges_financing_over_the_holding_period():
    n = 10
    idx = pd.date_range("2026-01-05 20:55", periods=n, freq="5min", tz="UTC")
    close = np.full(n, 100.0)
    pos = np.ones(n)
    fin = np.zeros(n)
    fin[5] = 0.002
    trades = extract_trades(pos, close, idx, cost_bps=0.0, financing=fin)
    assert len(trades) == 1
    assert trades[0]["financing_pct"] == 0.2
    assert trades[0]["net_pct_at_cost"] == -0.2


def test_summarize_financing_counts_nights():
    fin = np.zeros(10)
    fin[2] = 0.0001
    fin[5] = 0.0003
    s = summarize_financing(fin)
    assert s["nights_charged"] == 2
    np.testing.assert_allclose(s["total_pct"], 0.04)


class _Clock:
    def __init__(self, t: float) -> None:
        self._t = float(t)

    def now(self) -> float:
        return self._t


def _epoch(*args) -> float:
    return float(calendar.timegm(dt.datetime(*args).timetuple()))


def test_paper_broker_charges_swap_once_per_rollover() -> None:
    cfg = Config.load()
    cfg.set('execution.commission_bps', 0.0)
    cfg.set('execution.swap', {
        'enabled': True,
        'rollover_hour_utc': 21,
        'triple_swap_weekday': None,
        'families': {'fx': {'long_bps': 2.0, 'short_bps': 2.0}},
    })
    clock = _Clock(_epoch(2026, 1, 5, 20, 0))
    broker = PaperBroker(cfg, clock, starting_balance=10_000.0)
    broker.update_price('EURUSD', 1.10)
    broker.send_market_order('EURUSD', 'buy', 1000.0)
    after_fill = broker.balance

    broker.update_price('EURUSD', 1.10)          # pre-cut same day: no charge
    assert broker.balance == pytest.approx(after_fill)

    clock._t = _epoch(2026, 1, 5, 22, 0)         # first tick after the cut: 2 bps
    broker.update_price('EURUSD', 1.10)
    assert broker.balance == pytest.approx(after_fill - 1000.0 * 2.0 / 1e4)

    broker.update_price('EURUSD', 1.10)          # already charged today
    assert broker.balance == pytest.approx(after_fill - 1000.0 * 2.0 / 1e4)

    ticket = broker.get_open_positions()[0]['ticket']
    closed = broker.close_position('EURUSD', ticket)
    assert closed['financing'] == pytest.approx(1000.0 * 2.0 / 1e4)

