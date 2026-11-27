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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config
from data_ingestion.generate_sample import generate, load_jsonl
from main import _gate_override, evaluate_gate
from validation.tuning import best_passing, tune


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

    events = generate(cfg, steps=3000, seed=11, trend_strength=0.0015, mean_reversion=0.004)
    assert events, "generator produced no events"

    report = tune(events, cfg, trials=24, min_trades=3, max_dd_pct=5.0, is_frac=0.8)
    assert report.results, "tuning produced no trials"
    passing = best_passing(report.results)
    assert passing is not None, "no tuned config cleared the gate on trended history"
    assert passing.net_pnl > 0.0
    assert passing.trades >= 3
    assert passing.max_dd_pct <= 5.0


def test_tuned_config_closes_deploy_loop(tmp_path):
    """config/tuned.yaml (written by --tune) must be auto-loaded by the CLI config loader.

    This is the closure of the deployment loop: --generate --trendy -> --tune ->
    config/tuned.yaml -> --validate / --paper actually USE it. If _load_cfg ignored the
    tuned file, a validated, gate-passing strategy would never deploy.
    """
    import yaml as _yaml

    from main import TUNED_CONFIG, _load_cfg

    tuned = tmp_path / "tuned.yaml"
    _yaml.safe_dump({
        "strategy": {"composite": {"threshold": 0.42}, "exit_flip_conviction": 0.13},
        "risk": {"max_daily_drawdown_pct": 3.0},
    }, tuned.open("w"))

    import main
    orig = main.TUNED_CONFIG
    try:
        main.TUNED_CONFIG = tuned
        cfg = _load_cfg(None)
        assert cfg.get("strategy.composite.threshold") == 0.42, cfg.get("strategy.composite.threshold")
        assert cfg.get("strategy.exit_flip_conviction") == 0.13
        assert cfg.get("risk.max_daily_drawdown_pct") == 3.0
    finally:
        main.TUNED_CONFIG = orig


def test_validate_gate_verdict_logic():
    """The gate verdict derives from realized net P&L and drawdown, and FAILs on noise.

    Exercises the exact predicate used by cmd_validate: positive net expectancy AND
    drawdown within the configured limit => PASS. Pure-GBM history must not pass.
    """
    cfg = Config.load()
    cfg.set("data.market.sim.trend_strength", 0.0)  # pure noise, no edge

    from data_ingestion.generate_sample import generate
    from validation.replay_harness import ReplayHarness

    events = generate(cfg, steps=2000, seed=5, trend_strength=0.0)
    h = ReplayHarness(cfg, events)
    h.run()
    state = h.account_state()
    cap = cfg.get("validation.backtest.initial_capital", 10000.0)
    net = state["balance"] - cap
    dd = state.get("drawdown_pct", 0.0)
    max_dd = float(cfg.get("risk.max_daily_drawdown_pct", 2.0))

    gate_pass = (net > 0.0) and (dd <= max_dd)
    # noise has no real edge -> gate must reject
    assert not gate_pass, f"noise unexpectedly passed the gate (net={net}, dd={dd})"


def test_gate_decision_helper_paper_vs_live():
    """Gate decision helper respects per-mode overrides (reviewer #1)."""
    class Args:
        live = False

    gate = {
        "min_dsr": 0.0, "max_pbo": 0.5, "max_daily_drawdown_pct": 2.0,
        "live": {"min_dsr": 0.95, "max_pbo": 0.3, "max_daily_drawdown_pct": 0.5},
    }
    paper = _gate_override(gate, Args())
    Args.live = True
    live = _gate_override(gate, Args())

    assert paper["min_dsr"] < live["min_dsr"]          # 0.0 < 0.95
    assert paper["max_pbo"] > live["max_pbo"]          # 0.5 > 0.3
    assert paper["max_daily_drawdown_pct"] > live["max_daily_drawdown_pct"]

    assert evaluate_gate(True, True) == (True, "PASS")
    assert evaluate_gate(False, True) == (False, "FAIL")
    assert evaluate_gate(True, False) == (False, "FAIL")  # DSR/PBO meta fail


def test_gate_decision_helper_evaluates_noise_and_edge():
    """Pure noise (GBM, net<=0) must FAIL; trended+tuned (net>0) must PASS."""
    assert evaluate_gate(False, True) == (False, "FAIL")  # noise: base fails
    assert evaluate_gate(True, True) == (True, "PASS")    # edge + meta pass
    assert evaluate_gate(True, False) == (False, "FAIL")  # overfit: meta fails


def test_config_gate_is_single_source_of_truth():
    """The gate section (DSR / PBO / drawdown, per-mode overrides) is the
    single source of truth the CLI reads for --validate."""
    import yaml

    from main import _gate_override

    default = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    with open(default) as f:
        cfg = yaml.safe_load(f)

    gate = cfg["validation"]["gate"]
    assert gate["min_dsr"] == 0.0
    assert gate["max_pbo"] == 0.5
    assert gate["max_daily_drawdown_pct"] == 2.0
    assert gate["live"]["min_dsr"] == 0.95
    assert gate["live"]["max_pbo"] == 0.3
    assert gate["live"]["max_daily_drawdown_pct"] == 0.5

    class Args:
        live = False
    assert _gate_override(gate, Args()) == {
        "min_dsr": 0.0, "max_pbo": 0.5, "max_daily_drawdown_pct": 2.0,
    }
    Args.live = True
    assert _gate_override(gate, Args()) == {
        "min_dsr": 0.95, "max_pbo": 0.3, "max_daily_drawdown_pct": 0.5,
    }
