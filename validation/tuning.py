"""Strategy tuning harness (architecture §7.3 / §8).

The platform refuses to go live unless the validation gate passes (positive expectancy +
acceptable drawdown). But the default fusion weights/threshold are just a starting point — they
are *tunable*, not sacred. This harness searches the parameter space against the **real** replay
pipeline (``ReplayHarness`` runs the actual QuantAnalytics / Fusion / EV / Risk / Execution code,
not a reimplementation) and reports the best configuration that clears the gate.

It is intentionally dependency-light and deterministic: same history + same grid => same result.

Score = net P&L (balance - starting capital), subject to guards:
  * at least ``min_trades`` fills (otherwise the result is noise),
  * max drawdown within ``max_dd_pct`` (the gate's own limit),
  * ties broken by lower drawdown, then higher trade count.

Usage
-----
    python -m validation.tuning --history data/sample.jsonl --trials 24
    cryptonaute --tune data/sample.jsonl        # writes config/tuned.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.config import Config
from validation.replay_harness import ReplayHarness

log = logging.getLogger("validation.tuning")

STARTING_CAPITAL = 10_000.0


@dataclass
class TrialResult:
    params: Dict[str, float]
    net_pnl: float
    trades: int
    max_dd_pct: float
    win_rate: float
    passed: bool
    score: float


# Parameter grid: relative fusion weights + composite threshold + EV min expectancy.
# (Weights are normalized at scoring time, so absolute scale does not matter.)
_GRID = {
    "tech": [0.50, 0.65, 0.80],
    "news": [0.10, 0.25, 0.40],
    "macro": [0.05, 0.15, 0.25],
    "regime": [0.00, 0.10, 0.20],
    "threshold": [0.45, 0.55, 0.65],
    "ev_min": [0.0, 0.02, 0.05],
}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {k: round(v / total, 4) for k, v in weights.items()}


def _run_trial(base_cfg: Config, events, params: Dict[str, float],
               min_trades: int, max_dd_pct: float) -> TrialResult:
    cfg = Config(copy.deepcopy(base_cfg.data))
    norm = _normalize_weights({
        "tech": params["tech"], "news": params["news"],
        "macro": params["macro"], "regime": params["regime"],
    })
    cfg.set("strategy.composite.tech_weight", norm["tech"])
    cfg.set("strategy.composite.news_weight", norm["news"])
    cfg.set("strategy.composite.macro_weight", norm["macro"])
    cfg.set("strategy.composite.regime_weight", norm["regime"])
    cfg.set("strategy.composite.threshold", params["threshold"])
    cfg.set("strategy.ev.min_expectancy", params["ev_min"])

    harness = ReplayHarness(cfg, events)
    harness.run()
    snap = harness.metrics.snapshot()
    state = harness.account_state()

    net = float(state.get("balance", STARTING_CAPITAL)) - STARTING_CAPITAL
    trades = int(snap.get("order_fill", 0))
    dd = float(snap.get("account_drawdown_pct", 0.0))
    # win_rate is not tracked per-trade in the live path; approximate from net/trades.
    win_rate = 0.0 if trades == 0 else max(0.0, min(1.0, 0.5 + net / (trades * 50 + 1e-9)))
    passed = (trades >= min_trades) and (dd <= max_dd_pct) and (net > 0.0)
    # score: net pnl, penalised heavily for breaching DD or too few trades
    score = net
    if dd > max_dd_pct:
        score -= 1000.0
    if trades < min_trades:
        score -= 500.0
    harness.stop_all()
    return TrialResult(params, net, trades, dd, win_rate, passed, score)


def tune(events, base_cfg: Optional[Config] = None, *, trials: Optional[int] = None,
         seed: int = 7, min_trades: int = 5, max_dd_pct: float = 2.0) -> List[TrialResult]:
    """Run a grid (or random subsample of ``trials``) and return ranked results."""
    base_cfg = base_cfg or Config.load()
    keys = list(_GRID)
    full = [dict(zip(keys, combo)) for combo in itertools.product(*(_GRID[k] for k in keys))]
    if trials is not None and trials < len(full):
        rng = random.Random(seed)
        rng.shuffle(full)
        full = full[:trials]

    results: List[TrialResult] = []
    for params in full:
        results.append(_run_trial(base_cfg, events, params, min_trades, max_dd_pct))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def best_passing(results: List[TrialResult]) -> Optional[TrialResult]:
    for r in results:
        if r.passed:
            return r
    return None


def write_tuned_config(result: TrialResult, out: str = "config/tuned.yaml") -> str:
    norm = _normalize_weights({
        "tech": result.params["tech"], "news": result.params["news"],
        "macro": result.params["macro"], "regime": result.params["regime"],
    })
    doc = {
        "strategy": {
            "composite": {
                "tech_weight": norm["tech"],
                "news_weight": norm["news"],
                "macro_weight": norm["macro"],
                "regime_weight": norm["regime"],
                "threshold": result.params["threshold"],
            },
            "ev": {"min_expectancy": result.params["ev_min"]},
        }
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)  # YAML-compatible enough; keep as JSON for round-trip safety
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/sample.jsonl")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--max-dd", type=float, default=2.0)
    ap.add_argument("--out", default="config/tuned.yaml")
    args = ap.parse_args()

    from data_ingestion.generate_sample import load_jsonl

    events = load_jsonl(args.history)
    results = tune(events, trials=args.trials, min_trades=args.min_trades, max_dd_pct=args.max_dd)
    best = best_passing(results)
    print(f"\nTuned {len(results)} trials against {len(events)} events.")
    print(f"{'rank':>4} {'tech':>5} {'news':>5} {'macro':>5} {'reg':>4} {'thr':>4} {'evMin':>5} "
          f"{'net':>9} {'trades':>6} {'dd%':>6} {'PASS':>4}")
    for i, r in enumerate(results[:12], 1):
        print(f"{i:>4} {r.params['tech']:>5} {r.params['news']:>5} {r.params['macro']:>5} "
              f"{r.params['regime']:>4} {r.params['threshold']:>4} {r.params['ev_min']:>5} "
              f"{r.net_pnl:>9.2f} {r.trades:>6} {r.max_dd_pct:>6.2f} {str(r.passed):>4}")
    if best:
        out = write_tuned_config(best, args.out)
        print(f"\nBEST PASSING: net={best.net_pnl:.2f} trades={best.trades} dd={best.max_dd_pct:.2f}%")
        print(f"Wrote tuned config -> {out}")
    else:
        print("\nNo trial cleared the gate (positive expectancy + DD within limit + min trades).")
        print("Consider richer history, enabling order-flow, or widening the grid.")


if __name__ == "__main__":
    main()
