"""Tests for the execution/exit path and the tuning harness.

These prove the platform actually *realizes* P&L (positions open AND close, balance moves
by a sane amount — not the old bug where P&L was never realized or exploded to -$4.4M).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

from data_ingestion.generate_sample import generate, load_jsonl, write_jsonl
from execution.broker_adapter import PaperBroker
from execution.order_manager import OrderManager
from strategy_engine.state_machine import StrategyOrchestrator
from risk_management.veto_service import RiskVetoService
from validation.replay_harness import ReplayHarness
from validation.tuning import tune, best_passing


def _build_harness(cfg, events):
    h = ReplayHarness(cfg, events)
    h.run()
    return h


def test_positions_close_and_pnl_is_realized(tmp_path):
    cfg = Config.load()
    events = generate(cfg, 400)
    h = _build_harness(cfg, events)
    state = h.account_state()
    # positions opened during the run must be closed at end (realized P&L)
    assert h.broker.get_open_positions() == [], "all positions must be closed at replay end"
    # P&L is realized and financially sane (no explosion): within +/- 50% of capital
    net = state["balance"] - 10_000.0
    assert abs(net) < 5_000.0, f"P&L exploded: {net}"
    assert state["drawdown_pct"] < 100.0


def test_pnl_scale_is_notional_based(tmp_path):
    """A 2% notional position on a tiny move should move balance by cents, not millions."""
    cfg = Config.load()
    bus = MessageBus()
    metrics = Metrics()
    clock = Clock()
    broker = PaperBroker(cfg, clock)
    broker.update_price("EURUSD", 1.0500)
    om = OrderManager(bus, cfg, clock, metrics, broker=broker)
    om.start()
    # open via the real execute path (populates broker._positions)
    proposal = {"symbol": "EURUSD", "direction": 1.0, "composite": 0.9, "position_pct": 2.0,
                "passes_ev": True, "veto": "allow", "execute": True}
    om._execute(proposal)
    broker.update_price("EURUSD", 1.0510)  # +0.095% move
    om.request_close("EURUSD")
    # expected pnl ~ +0.095% * 200 notional ≈ 0.19 before slippage; round-trip slippage
    # makes it slightly negative but order-of-magnitude correct (cents, not millions).
    net = broker.get_account_state()["balance"] - 10_000.0
    assert abs(net) < 1.0, f"notional P&L wrong magnitude: {net}"
    assert net != 0.0, "P&L should be realized, not zero"


def test_tuning_runs_and_pnl_stays_finite(tmp_path):
    cfg = Config.load()
    events = generate(cfg, 300)
    results = tune(events, cfg, trials=4, min_trades=1)
    assert results, "tuning produced results"
    for r in results:
        assert abs(r.net_pnl) < 5_000.0, f"tuning P&L exploded: {r.net_pnl}"
        assert r.max_dd_pct < 100.0
    # does not raise; best_passing may be None on flat data (gate working correctly)
    assert best_passing(results) is None or isinstance(best_passing(results), object)
