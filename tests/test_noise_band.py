"""Tests for the noise-band breakout strategy."""

from __future__ import annotations

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bus import MessageBus
from core.config import Config, Clock
from core.events import EventType, make_event
from core.metrics import Metrics
from strategy_engine.noise_band import NoiseBandStrategy


def _mk_strategy(cfg_overrides=None):
    cfg = Config.load()
    cfg.set("strategy.noise_band.confirm_n", 2)
    cfg.set("strategy.noise_band.min_bars", 5)
    cfg.set("strategy.noise_band.lookback", 40)
    cfg.set("strategy.noise_band.cooldown_s", 0.0)
    for dotted, value in (cfg_overrides or {}).items():
        cfg.set(dotted, value)
    clock = Clock()
    metrics = Metrics()
    bus = MessageBus()
    strat = NoiseBandStrategy(bus, cfg, clock, metrics)
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
    for i in range(10):
        price *= 1.00001 if i % 2 else 0.99999   # 0.1 bps noise — deep below the band
        _candle(strat, "EURUSD", price, float(i))
    assert proposals == []
    assert strat.metrics.snapshot().get("noise_band_proposal") is None


def test_breakout_emits_proposal_with_direction():
    strat = _mk_strategy()
    proposals = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    price = 100.0
    ts = 0.0
    for i in range(20):
        price *= 1.00001 if i % 2 else 0.99999   # 0.1 bps noise — below the band
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    # breakout: two strong consecutive up-candles (~40 bps each, >> band)
    for i in range(2):
        price *= 1.004
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert len(proposals) == 1
    payload = proposals[0].payload
    assert payload["symbol"] == "EURUSD"
    assert payload["direction"] == 1.0
    assert payload["execute"] is False and payload["passes_ev"] is True
    assert payload["source"] == "noise_band"
    assert payload["sl_pct"] > 0 and payload["tp_pct"] > payload["sl_pct"]


def test_reentry_emits_exit():
    strat = _mk_strategy()
    proposals = []
    exits = []
    strat.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: proposals.append(e))
    strat.bus.subscribe(EventType.ORDER_EXIT, lambda e: exits.append(e))
    price = 100.0
    ts = 0.0
    for i in range(20):
        price *= 1.00001 if i % 2 else 0.99999
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    for i in range(2):
        price *= 1.004
        ts += 300.0
        _candle(strat, "EURUSD", price, ts)
    assert proposals
    # simulated entry fill for our direction
    strat.bus.publish(make_event(
        EventType.ORDER_FILL, "execution",
        {"symbol": "EURUSD", "direction": 1.0, "closed": False, "price": price},
        timestamp=ts))
    assert "EURUSD" in strat._open
    # noise resumes: tiny candle back inside the band (ret << exit threshold)
    price *= 1.00001
    ts += 300.0
    _candle(strat, "EURUSD", price, ts)
    assert len(exits) == 1
    assert exits[0].payload["reason"] == "noise_reentry"
    assert "EURUSD" not in strat._open


def test_fill_from_other_strategy_not_adopted():
    strat = _mk_strategy()
    exits = []
    strat.bus.subscribe(EventType.ORDER_EXIT, lambda e: exits.append(e))
    # a fill with no matching armed proposal must NOT be tracked
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
    for i in range(30):
        price *= 1.004  # huge persistent drift, every candle unclosed
        _candle(strat, "EURUSD", price, float(i), closed=False)
    assert proposals == []  # unclosed candles are never counted
