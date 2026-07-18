"""Tests for Layer 2 (analytics), Layer 3 (self-correction), Layer 4 (strategy), Layer 5/6 (exec/risk)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.bus import MessageBus
from core.config import Config, Clock
from core.events import EventType, make_event
from core.metrics import Metrics

from analytics.quant.indicators import QuantAnalytics, rsi, sma, ema
from analytics.quant.regime import RegimeDetector
from analytics.nlp.news_scorer import NewsScorer
from strategy_engine.signal_fusion import SignalFusion
from strategy_engine.expected_value import ExpectedValueEngine
from strategy_engine.state_machine import StrategyOrchestrator, SymbolFSM
from execution.broker_adapter import PaperBroker
from execution.order_manager import OrderManager
from risk_management.veto_service import RiskVetoService
from risk_management.circuit_breaker import CircuitBreaker
from self_correction.mistral_client import MistralClient, PostTradeReviewer


# ---- Layer 2 ---------------------------------------------------------------
def test_indicators_numerical():
    x = np.arange(1, 101, dtype=float)
    assert np.allclose(sma(x, 5)[4:], np.convolve(x, np.ones(5) / 5, mode="valid"))
    r = rsi(x, 14)
    assert not np.any(np.isnan(r)) or np.isnan(r).sum() < len(r)
    assert float(np.nanmin(r)) >= 0 and float(np.nanmax(r)) <= 100


def test_quant_emits_tech_signal():
    cfg = Config.load()
    qa = QuantAnalytics(MessageBus(), cfg, Clock(), Metrics())
    qa.start()
    out = []
    qa.bus.subscribe(EventType.TECH_SIGNAL, lambda e: out.append(e))
    # feed enough candles to compute MA cross
    for i in range(40):
        qa.on_candle(make_event(EventType.CANDLE, "sim", {
            "symbol": "EURUSD", "timeframe": "M5", "open": 1 + i * 0.001,
            "high": 1 + i * 0.001, "low": 1 + i * 0.001, "close": 1 + i * 0.001, "volume": 1.0,
        }, symbol_impact=["EURUSD"], timestamp=float(i)))
    assert out, "no TECH_SIGNAL produced"
    assert "direction" in out[0].payload and "strength" in out[0].payload


def test_regime_classification():
    cfg = Config.load()
    rd = RegimeDetector(MessageBus(), cfg, Clock(), Metrics())
    rd.start()
    out = []
    rd.bus.subscribe(EventType.REGIME, lambda e: out.append(e))
    # strong uptrend -> trending
    for i in range(30):
        rd.on_candle(make_event(EventType.CANDLE, "sim", {
            "symbol": "EURUSD", "timeframe": "M5", "open": 1 + i * 0.01,
            "high": 1 + i * 0.01, "low": 1 + i * 0.01, "close": 1 + i * 0.01, "volume": 1.0,
        }, symbol_impact=["EURUSD"], timestamp=float(i)))
    assert out and out[-1].payload["regime"] == "trending"


def test_news_scorer_lexicon():
    cfg = Config.load()
    ns = NewsScorer(MessageBus(), cfg, Clock(), Metrics())
    ns.start()
    out = []
    ns.bus.subscribe(EventType.NEWS_SCORE, lambda e: out.append(e))
    ns.on_news(make_event(EventType.NEWS, "sim",
                          {"title": "EURUSD rallies on strong data", "summary": ""},
                          symbol_impact=[], timestamp=0.0))
    assert out, "relevant news not scored"
    assert out[0].symbol_impact == ["EURUSD"]
    assert out[0].payload["score"] > 0  # bullish


def test_news_scorer_irrelevant_dropped():
    cfg = Config.load()
    ns = NewsScorer(MessageBus(), cfg, Clock(), Metrics())
    ns.start()
    out = []
    ns.bus.subscribe(EventType.NEWS_SCORE, lambda e: out.append(e))
    ns.on_news(make_event(EventType.NEWS, "sim",
                          {"title": "local bakery opens", "summary": ""},
                          symbol_impact=[], timestamp=0.0))
    assert out == [], "irrelevant news should be dropped"


# ---- Layer 4 ---------------------------------------------------------------
def test_signal_fusion_combines():
    cfg = Config.load()
    sf = SignalFusion(MessageBus(), cfg, Clock(), Metrics())
    sf.start()
    out = []
    sf.bus.subscribe(EventType.COMPOSITE_SIGNAL, lambda e: out.append(e))
    sf._ingest(make_event(EventType.TECH_SIGNAL, "q", {
        "symbol": "EURUSD", "composite": 1.0, "strength": 1.0}), "tech")
    assert out, "no composite produced from a strong tech signal"
    assert out[0].payload["composite"] > 0


def test_ev_engine_gates_on_min_ev():
    cfg = Config.load()
    ev = ExpectedValueEngine(MessageBus(), cfg, Clock(), Metrics())
    ev.start()
    out = []
    ev.bus.subscribe(EventType.ORDER_PROPOSAL, lambda e: out.append(e))
    ev.on_composite(make_event(EventType.COMPOSITE_SIGNAL, "fusion", {
        "symbol": "EURUSD", "composite": 0.9, "confidence": 0.9, "above_threshold": True},
        symbol_impact=["EURUSD"], timestamp=0.0))
    assert out, "no order proposal"
    # with no calibration data, prior EV ~ 0, so passes_ev depends on min_ev (0.0 default)
    assert "ev" in out[0].payload and "passes_ev" in out[0].payload


def test_fsm_transitions():
    fsm = SymbolFSM("EURUSD")
    assert fsm.state == "idle"
    fsm.detect(); assert fsm.state == "signal_detected"
    fsm.confirm(); assert fsm.state == "confirmed"
    fsm.open(); fsm.manage(); assert fsm.state == "managing"
    fsm.close(); fsm.reset(); assert fsm.state == "idle"


# ---- Layer 5 ---------------------------------------------------------------
def test_paper_broker_fill_and_close():
    cfg = Config.load()
    b = PaperBroker(cfg, Clock(), starting_balance=10000.0)
    b.update_price("EURUSD", 1.05)
    res = b.send_market_order("EURUSD", "buy", 0.1)
    assert res["ok"] and res["ticket"] > 0
    b.update_price("EURUSD", 1.10)  # price moved in our favor
    close = b.close_position("EURUSD", res["ticket"])
    assert close["ok"] and close["pnl"] != 0.0
    assert b.get_account_state()["balance"] != 10000.0


def test_order_manager_retry_backoff():
    cfg = Config.load()
    clock = Clock()
    failing = PaperBroker(cfg, clock)

    class Flaky(PaperBroker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.calls = 0

        def send_market_order(self, *a, **k):
            self.calls += 1
            if self.calls < 2:
                return {"ok": False, "reason": "transient"}
            return super().send_market_order(*a, **k)

    flaky = Flaky(cfg, clock)
    om = OrderManager(MessageBus(), cfg, clock, Metrics(), broker=flaky)
    om.start()
    om.on_tick(make_event(EventType.TICK, "sim", {"symbol": "EURUSD", "mid": 1.05},
                          symbol_impact=["EURUSD"], timestamp=0.0))
    out = []
    om.bus.subscribe(EventType.ORDER_FILL, lambda e: out.append(e))
    om._execute({"symbol": "EURUSD", "direction": 1.0, "composite": 0.9, "ev": 0.0,
                 "passes_ev": True, "veto": "allow", "execute": True})
    assert out, "order never filled despite retry"
    assert flaky.calls >= 2


# ---- Layer 6 ---------------------------------------------------------------
def test_risk_veto_signs_and_blocks():
    cfg = Config.load()
    rv = RiskVetoService(MessageBus(), cfg, Clock(), Metrics())
    rv.start()
    out = []
    rv.bus.subscribe(EventType.RISK_VETO, lambda e: out.append(e))
    rv.on_proposal(make_event(EventType.ORDER_PROPOSAL, "ev", {
        "symbol": "EURUSD", "position_pct": 99.0, "passes_ev": True},
        symbol_impact=["EURUSD"], timestamp=0.0))
    assert out and out[0].payload["decision"] == "deny"
    assert len(out[0].payload["signature"]) == 16


def test_circuit_breaker_trips():
    cfg = Config.load()
    cb = CircuitBreaker(MessageBus(), cfg, Clock(), Metrics())
    cb.start()
    out = []
    cb.bus.subscribe(EventType.RISK_BREACH, lambda e: out.append(e))
    cb.on_metric(make_event(EventType.METRIC, "m",
                            {"name": "account_drawdown_pct", "value": 5.0},
                            timestamp=0.0))
    assert out and cb.tripped


# ---- Layer 3 ---------------------------------------------------------------
def test_mistral_stub_is_deterministic():
    cfg = Config.load()
    c = MistralClient(cfg)
    assert c.backend == "stub"
    r1 = c.complete("post-trade review for a trade")
    r2 = c.complete("post-trade review for a trade")
    assert r1 == r2
    import json
    assert "recommendation" in json.loads(r1)


def test_post_trade_review_emits_on_close():
    cfg = Config.load()
    pr = PostTradeReviewer(MessageBus(), cfg, Clock(), Metrics())
    pr.start()
    out = []
    pr.bus.subscribe(EventType.POST_TRADE_REVIEW, lambda e: out.append(e))
    pr.on_fill(make_event(EventType.ORDER_FILL, "exec", {
        "symbol": "EURUSD", "closed": True, "side": "buy", "price_entry": 1.05,
        "price": 1.07, "pnl": 10.0}, symbol_impact=["EURUSD"], timestamp=0.0))
    assert out, "post-trade review not emitted on close"
    assert "review" in out[0].payload
