"""Regression tests for the validation gate going GREEN (architecture §7.3 / §8).

These encode the hard-won properties that, if they silently regress, make the platform
unable to deploy a tradeable strategy:

1. The replay bus must resolve a handler's emitted event (the proposal -> veto -> fill
   cascade) BEFORE the next raw (recorded) event. If it defers reactions to the back of the
   queue, positions only open at the very end of the recording and the gate can never pass.
2. On trended history the tuning harness must actually discover at least one passing
   configuration (positive expectancy + acceptable drawdown + sufficient trade count) by
   running candidates through the REAL 6-layer pipeline — not a reimplementation.
"""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config
from validation.tuning import tune, best_passing
from data_ingestion.generate_sample import generate, load_jsonl


def test_replay_cascade_resolves_before_next_raw_event():
    """Re-enacts the bug that broke the gate: reactions deferred to end-of-recording.

    A handler that emits a derived event while processing a raw event must see that derived
    event dispatched before the next raw event, so the full decision cascade completes inline.
    """
    from core.bus import MessageBus
    from core.events import EventType, make_event

    bus = MessageBus(replay=True)
    seen = []

    def on_raw(ev):
        seen.append(("raw", ev.payload["v"]))
        # a strategy reacting to the raw event (e.g. composite -> proposal -> veto -> fill)
        bus.publish(make_event(EventType.TECH_SIGNAL, "derived", {"v": ev.payload["v"]}))

    bus.subscribe(EventType.CANDLE, on_raw)
    bus.subscribe(EventType.TECH_SIGNAL, lambda e: seen.append(("derived", e.payload["v"])))

    for v in range(3):
        bus.enqueue(make_event(EventType.CANDLE, "t", {"v": v}))
    bus.run_replay()

    assert seen == [
        ("raw", 0), ("derived", 0),
        ("raw", 1), ("derived", 1),
        ("raw", 2), ("derived", 2),
    ], seen


def test_tuning_gate_passes_on_trended_history(tmp_path):
    """The harness must find a passing config on history with a real, persistent edge.

    Uses the project's own generator with a modest injected trend and short, frequently
    reversing regimes so the trend-follower gets round-trip opportunities. If this fails,
    the full pipeline (indicators -> fusion -> EV -> signed veto -> execution -> P&L) is
    not actually trading during replay — the first symptom of the cascade bug above.
    """
    cfg = Config.load()
    cfg.set("data.market.sim.trend_strength", 0.0015)
    cfg.set("data.market.sim.mean_reversion", 0.004)

    events = generate(cfg, steps=2000, seed=11, trend_strength=0.0015, mean_reversion=0.004)
    assert events, "generator produced no events"

    results = tune(events, cfg, trials=12, min_trades=3, max_dd_pct=5.0)
    assert results, "tuning produced no trials"
    passing = best_passing(results)
    assert passing is not None, "no tuned config cleared the gate on trended history"
    assert passing.net_pnl > 0.0
    assert passing.trades >= 3
    assert passing.max_dd_pct <= 5.0
