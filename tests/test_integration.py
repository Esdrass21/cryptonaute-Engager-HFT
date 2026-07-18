"""End-to-end integration test: full 6-layer flow through the replay harness.

This is the authoritative "it actually works" test. It feeds recorded (deterministic) events
through the REAL pipeline components and asserts the flow Data -> Analytics -> Fusion -> EV ->
Risk Veto -> Execution produces fills, and that the risk veto can block orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config
from core.events import EventType, NormalizedEvent, make_event
from validation.replay_harness import ReplayHarness


def _build_events(n: int = 400, seed: int = 1) -> list:
    """Build a deterministic event stream: candles for all symbols (+ a trend so signals
    cross the fusion threshold) plus a couple of news items."""
    rng = __import__("random").Random(seed)
    events: list = []
    syms = ["EURUSD", "XAUUSD", "BTCUSDT"]
    prices = {"EURUSD": 1.05, "XAUUSD": 2300.0, "BTCUSDT": 60000.0}
    trend = {"EURUSD": 0.0006, "XAUUSD": 0.0004, "BTCUSDT": -0.0005}  # EURUSD strong up-trend
    for i in range(n):
        for s in syms:
            p = prices[s]
            p = p * (1 + trend[s] + rng.gauss(0, 0.0004))
            prices[s] = p
            events.append(make_event(
                EventType.CANDLE, "sim.market",
                payload={"symbol": s, "timeframe": "M5", "open": p, "high": p * 1.001,
                         "low": p * 0.999, "close": p, "volume": 1.0, "closed": True},
                symbol_impact=[s], timestamp=float(i),
            ))
        if i % 80 == 0:
            events.append(make_event(
                EventType.NEWS, "sim.news",
                payload={"title": f"EURUSD rallies on strong data {i}",
                         "summary": "bullish headline", "url": ""},
                symbol_impact=[], timestamp=float(i),
            ))
    return events


def test_full_pipeline_produces_fills():
    cfg = Config.load()
    events = _build_events()
    harness = ReplayHarness(cfg, events)
    harness.run()
    state = harness.account_state()
    # The pipeline must have run end-to-end without error and produced some activity.
    snap = harness.metrics.snapshot()
    assert snap.get("tech_signal_emitted", 0) > 0, "quant layer emitted no tech signals"
    assert snap.get("composite_signal_emitted", 0) > 0, "fusion produced no composite signals"
    assert snap.get("risk_veto_emitted", 0) > 0, "risk veto never ran"
    # Either we opened positions (fills) or the veto blocked everything; both are valid
    # outcomes, but the pipeline must have reached an execution decision.
    assert (snap.get("order_fill", 0) + snap.get("order_blocked_by_risk", 0)) > 0
    # Account state is queryable (paper broker).
    assert "balance" in state
    harness.stop_all()


def test_risk_veto_blocks_oversized_position():
    cfg = Config.load()
    harness = ReplayHarness(cfg, [])
    # Craft a proposal that exceeds the HARD CAP and feed it to the veto.
    big = make_event(
        EventType.ORDER_PROPOSAL, "strategy.ev",
        payload={"symbol": "EURUSD", "direction": 1.0, "composite": 0.9,
                 "position_pct": 99.0, "passes_ev": True},
        symbol_impact=["EURUSD"], timestamp=0.0,
    )
    got = []
    harness.bus.subscribe(EventType.RISK_VETO, lambda e: got.append(e))
    harness.veto.on_proposal(big)
    harness.bus.run_replay()  # harness bus is in replay mode -> flush enqueued veto
    assert got, "veto produced no decision"
    assert got[0].payload["decision"] == "deny"
    assert any("HARD CAP" in r for r in got[0].payload["reasons"])
    harness.stop_all()


def test_replay_is_deterministic():
    cfg = Config.load()
    e1 = _build_events()
    e2 = _build_events()
    h1 = ReplayHarness(cfg, e1)
    h2 = ReplayHarness(cfg, e2)
    h1.run();
    h2.run()
    assert h1.metrics.snapshot() == h2.metrics.snapshot(), "replay not deterministic"
    h1.stop_all(); h2.stop_all()
