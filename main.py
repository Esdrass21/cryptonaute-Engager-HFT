"""Command-line entry point.

Examples
--------
    # Paper-trading simulation (default): sim market + paper broker, run 500 steps then stop.
    python -m main --paper --steps 500

    # Generate sample history for the replay/backtest harness.
    python -m main --generate data/sample.jsonl --steps 3000

    # Validate a strategy: replay through real pipeline + vectorized backtest + walk-forward + MC.
    python -m main --validate data/sample.jsonl

    # Live (requires MT5 terminal logged in / exchange keys in .env).
    python -m main --live
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root (where the top-level packages `core`, `analytics`, ... live)
# is importable even when launched via the installed `cryptonaute` console script from
# a different working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("CRYPTONAUTE_HOME", str(Path(_PROJECT_ROOT) / "data"))

from core.config import Config

from data_ingestion.generate_sample import generate, load_jsonl, write_jsonl
from orchestrator import Orchestrator
from validation.engine import monte_carlo, vectorized_backtest, walk_forward
from validation.replay_harness import ReplayHarness

log = logging.getLogger("main")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"
TUNED_CONFIG = Path(__file__).resolve().parent / "config" / "tuned.yaml"


def _load_cfg(path: str | None) -> Config:
    """Load config, auto-merging a gate-passing tuned config when present.

    Precedence: explicit ``--config`` > auto-loaded ``config/tuned.yaml`` (the
    calibration produced by ``--tune``, architecture §7.3) > ``config/default.yaml``.
    This closes the loop: a tuned, validated strategy deploys automatically unless a
    config is explicitly chosen.
    """
    if path and Path(path).exists():
        return Config.load(str(DEFAULT_CONFIG), path)
    if TUNED_CONFIG.exists():
        return Config.load(str(DEFAULT_CONFIG), str(TUNED_CONFIG))
    if DEFAULT_CONFIG.exists():
        return Config.load(str(DEFAULT_CONFIG))
    return Config.load()


def cmd_run(args) -> int:
    cfg = _load_cfg(args.config)
    cfg.set("mode", "live" if args.live else "paper")
    if args.live:
        cfg.set("data.market.source", "mt5")
        cfg.set("execution.broker", "mt5")
    orch = Orchestrator(cfg)
    orch.run(steps=args.steps)
    return 0


def cmd_generate(args) -> int:
    cfg = _load_cfg(args.config)
    events = generate(cfg, args.gen_steps, seed=args.seed, trend_strength=args.trendy,
                     mean_reversion=args.reversion)
    n = write_jsonl(events, args.generate)
    log.info("Generated %d events -> %s", n, args.generate)
    return 0


def cmd_validate(args) -> int:
    cfg = _load_cfg(args.config)
    events = load_jsonl(args.validate)
    # 1) Replay harness through the REAL pipeline (paper broker). This is the authoritative
    #    expectancy/Drawdown measurement — the strategy actually trades the history.
    harness = ReplayHarness(cfg, events)
    harness.run()
    state = harness.account_state()
    net = state["balance"] - cfg.get("validation.backtest.initial_capital", 10000.0)
    dd = state.get("drawdown_pct", 0.0)
    print("\n[1/3] Replay through real pipeline (paper broker):")
    print(f"      net P&L = {net:+.2f}  drawdown = {dd:.2f}%  "
          f"open positions = {len(harness.open_positions())}")

    # 2) Vectorized backtest on close prices of the first symbol (reference sanity check).
    sym = cfg.get("symbols", ["EURUSD"])[0]
    closes = [e.payload["close"] for e in events if e.type == "candle" and e.payload.get("symbol") == sym]
    gate_pass = False
    if closes:
        import numpy as np

        arr = np.array(closes)
        signal = np.zeros_like(arr)
        ma_fast = np.concatenate([[arr[0]], (arr[1:] + arr[:-1]) / 2])
        signal[1:] = np.sign(arr[1:] - ma_fast[1:]) * 0.5
        bt = vectorized_backtest(arr, signal, initial_capital=float(cfg.get("validation.backtest.initial_capital", 10000.0)))
        wf = walk_forward(arr, lambda tr, te: np.sign(te[1:] - te[:-1]) * 0.5)
        mc = monte_carlo(bt.trades, n_paths=int(cfg.get("validation.monte_carlo.n_paths", 1000)))
        print("\n[2/3] Vectorized backtest (reference):")
        for k, v in bt.summary().items():
            print(f"      {k:18s}: {v}")
        print("\n[3/3] Monte Carlo (tail risk):")
        for k, v in mc.items():
            print(f"      {k:18s}: {round(v, 2)}")
        avg_wr = sum(r.win_rate for r in wf) / len(wf) if wf else 0.0
        print(f"\n      walk_forward folds = {len(wf)}  avg OOS win_rate = {avg_wr:.3f}")

        # Architecture §7.3 gate: positive expectancy + acceptable drawdown before live.
        max_dd = float(cfg.get("risk.max_daily_drawdown_pct", 2.0))
        gate_pass = (net > 0.0) and (dd <= max_dd)
        verdict = "PASS" if gate_pass else "FAIL"
        print(f"\nVALIDATION GATE (architecture §7.3): require positive expectancy + acceptable DD "
              f"before live.")
        print(f"  net expectancy = {net:+.2f}  (need > 0)")
        print(f"  max drawdown  = {dd:.2f}%  (limit {max_dd:.2f}%)")
        print(f"  RESULT: {verdict}")
    else:
        print("No candles found in recorded history for backtest.")
    return 0 if gate_pass else 1


def cmd_tune(args) -> int:
    cfg = _load_cfg(args.config)
    events = load_jsonl(args.tune)
    from validation.tuning import tune, best_passing, write_tuned_config

    results = tune(events, cfg, trials=args.trials, min_trades=args.min_trades, max_dd_pct=args.max_dd)
    best = best_passing(results)
    print(f"\nTuned {len(results)} trials against {len(events)} events.")
    print(f"{'rank':>4} {'tech':>5} {'news':>5} {'macro':>5} {'reg':>4} {'thr':>4} {'evMin':>5} "
          f"{'net':>10} {'trades':>6} {'dd%':>6} {'PASS':>4}")
    for i, r in enumerate(results[:12], 1):
        print(f"{i:>4} {r.params['tech']:>5} {r.params['news']:>5} {r.params['macro']:>5} "
              f"{r.params['regime']:>4} {r.params['threshold']:>4} {r.params['ev_min']:>5} "
              f"{r.net_pnl:>10.2f} {r.trades:>6} {r.max_dd_pct:>6.2f} {str(r.passed):>4}")
    if best:
        out = write_tuned_config(best, args.out)
        print(f"\nBEST PASSING: net={best.net_pnl:.2f} trades={best.trades} dd={best.max_dd_pct:.2f}%")
        print(f"Wrote tuned config -> {out}")
    else:
        print("\nNo trial cleared the gate (positive expectancy + DD within limit + min trades).")
        print("Consider richer/trending history, enabling order-flow, or widening the grid.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cryptonaute", description="Cryptonaute HFT Quant Platform")
    ap.add_argument("--config", default=None)
    ap.add_argument("--paper", action="store_true", help="Paper-trading simulation (default)")
    ap.add_argument("--live", action="store_true", help="Live trading (requires broker/keys)")
    ap.add_argument("--steps", type=int, default=None, help="Bound run length (tests/quick runs)")
    ap.add_argument("--generate", metavar="OUT", default=None, help="Generate sample history JSONL")
    ap.add_argument("--gen-steps", dest="gen_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trendy", dest="trendy", type=float, default=0.0,
                    help="Inject persistent trends into generated history (e.g. 0.003)")
    ap.add_argument("--reversion", dest="reversion", type=float, default=0.01,
                    help="OU mean-reversion speed (lower = stronger, more persistent trends)")
    ap.add_argument("--validate", metavar="HISTORY", default=None, help="Validate a recorded history file")
    ap.add_argument("--tune", metavar="HISTORY", default=None, help="Grid-search fusion params vs replay")
    ap.add_argument("--trials", type=int, default=None, help="Max tuning trials (random subsample)")
    ap.add_argument("--min-trades", dest="min_trades", type=int, default=5)
    ap.add_argument("--max-dd", dest="max_dd", type=float, default=2.0)
    ap.add_argument("--out", default="config/tuned.yaml", help="Where to write best tuned config")
    return ap


def main(argv: list | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.generate:
        return cmd_generate(args)
    if args.validate:
        return cmd_validate(args)
    if args.tune:
        return cmd_tune(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
