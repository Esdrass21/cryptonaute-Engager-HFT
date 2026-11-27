# tests/test_tsmom.py
"""Tests for the live TSMOM (time-series momentum blend) strategy."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics
from strategy_engine.tsmom import TSMOMStrategy


def _mk_strategy(cfg_overrides=None):
    cfg = Config.load()
    cfg.set("strategy.tsmom.lookbacks", [2, 4])
    cfg.set("strategy.tsmom.min_bars", 5)
    cfg.set("strategy.tsmom.rebalance_s", 0.0)
    cfg.set("strategy.tsmom.vol_win", 5)
    for dotted, value in (cfg_overrides or {}).items():
        cfg.set(dotted, value)
    clock = Clock()
    metrics = Metrics()
    bus = MessageBus()
    strat = TSMOMStrategy(bus, cfg, clock, metrics)
    strat.start()
    return strat


def _candle(strat, sym, price, ts, closed=True):
    strat.bus.publish(make_event(
        EventType.CANDLE, "test", {"symbol": sym, "timeframe": "M5", "open": price,
                                   "high": price * 1.001, "low": price * 0.999,
                                   "close": price, "volume": 1.0, "closed": closed},
        symbol_impact=[sym], timestamp=ts))


def test_warmup_no_proposals():
    strat = _mk_strategy()
    proposals = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    price = 100.0
    for i in range(4):
        price *= 1.0001
        _candle(strat, "EURUSD", price, float(i))
    assert proposals == []


def test_uptrend_emits_long_proposal_vol_scaled():
    strat = _mk_strategy()
    proposals = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    price = 100.0
    ts = 0.0
    for i in range(8):
        price *= 1.0002 * (1.0 + 0.0001 if i % 2 else 0.9999)  # drift + jitter so vol > 0
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert proposals
    payload = proposals[-1].payload
    assert payload["symbol"] == "EURUSD"
    assert payload["direction"] == 1.0
    assert payload["execute"] is False and payload["passes_ev"] is True
    assert payload["source"] == "tsmom"
    assert 0.0 < payload["position_pct"] <= strat.max_leverage


def test_flip_emits_exit_then_reopens():
    strat = _mk_strategy()
    proposals = []
    exits = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    strat.bus.subscribe(EventType.ORDER_EXIT, lambda e: exits.append(e))
    price = 100.0
    ts = 0.0
    for i in range(8):
        price *= 1.0002 * (1.0 + 0.0001 if i % 2 else 0.9999)
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert proposals
    strat.bus.publish(make_event(
        EventType.ORDER_FILL, "execution",
        {"symbol": "EURUSD", "direction": 1.0, "closed": False, "price": price},
        timestamp=ts))
    assert "EURUSD" in strat._open
    # strong reversal over the lookbacks -> sign flips at next rebalance
    for i in range(8):
        price *= 0.998 * (1.0 + 0.0001 if i % 2 else 0.9999)
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert len(exits) == 1
    assert exits[0].payload["reason"] == "tsmom_flip"
    assert "EURUSD" not in strat._open
    assert proposals[-1].payload["direction"] == -1.0  # reopened short


def test_rebalance_interval_respected():
    strat = _mk_strategy({"strategy.tsmom.rebalance_s": 86400.0})
    proposals = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    price = 100.0
    ts = 0.0
    for i in range(10):
        price *= 1.0002 * (1.0 + 0.0001 if i % 2 else 0.9999)
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert len(proposals) == 1  # warm-up done once; no further rebalance within the day


def test_fill_from_other_strategy_not_adopted():
    strat = _mk_strategy()
    strat.bus.publish(make_event(
        EventType.ORDER_FILL, "execution",
        {"symbol": "EURUSD", "direction": 1.0, "closed": False, "price": 100.0},
        timestamp=1.0))
    assert "EURUSD" not in strat._open


def test_ignores_unclosed_candles():
    strat = _mk_strategy()
    proposals = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    price = 100.0
    for i in range(10):
        price *= 1.0002
        _candle(strat, "EURUSD", price, float(i), closed=False)
    assert proposals == []
