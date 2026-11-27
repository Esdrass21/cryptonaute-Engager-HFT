# validation/tuning.py
"""
Strategy tuning harness (architecture §7.3 / §8).

Phase 3 design
--------------
* Tune on an **in-sample** portion of the history only (`is_frac`, default 0.8).
* Select the best IS config by net PnL subject to min-trades + max-DD guards.
* Compute Deflated Sharpe Ratio (DSR) from the per-trial Sharpes on IS.
* Compute Probability of Backtest Overfitting (PBO) from the per-trial
  per-recorded-event return streams on IS.
* Re-run the winner on the **out-of-sample** tail for a clean, honest number.
* Persist everything to ``config/tuned.yaml`` + ``config/tuned.yaml.meta.json``.

The validate command reads the meta sidecar so the gate applies DSR + PBO
without re-running the grid.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import Config
from validation.metrics import (
    DSRResult,
    SharpeStats,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from validation.replay_harness import ReplayHarness

log = logging.getLogger("validation.tuning")

STARTING_CAPITAL = 10_000.0


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class TrialResult:
    params: dict[str, float]
    net_pnl: float
    trades: int
    max_dd_pct: float
    win_rate: float
    passed: bool
    score: float
    # Phase 3: real return streams
    trade_records: list[tuple] = field(default_factory=list)   # [(exit_ts, pnl), ...]
    equity_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    is_sharpe: float = 0.0

    # Backwards-compat alias for old callers.
    @property
    def oos_returns(self) -> np.ndarray:
        return np.array([p for _, p in self.trade_records], dtype=float)


@dataclass
class TuneReport:
    results: list[TrialResult]
    best: TrialResult | None
    n_trials: int
    is_frac: float
    # IS-side statistics
    is_sharpe: float
    sr0: float
    dsr: float
    psr_vs_zero: float
    pbo: float
    # OOS-side statistics for the winner
    oos_net: float | None
    oos_dd: float | None
    oos_sharpe: float | None
    oos_trades: int | None
    # Config used for the OOS run
    oos_events: int


# --------------------------------------------------------------------------- #
# Grid                                                                         #
# --------------------------------------------------------------------------- #

_GRID = {
    "tech": [0.50, 0.65, 0.80],
    "news": [0.10, 0.25, 0.40],
    "macro": [0.05, 0.15, 0.25],
    "regime": [0.00, 0.10, 0.20],
    "threshold": [0.45, 0.55, 0.65],
    "ev_min": [0.0, 0.02, 0.05],
    "exit_flip": [0.10, 0.25, 0.40],
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {k: round(v / total, 4) for k, v in weights.items()}


def _apply_params(cfg: Config, params: dict[str, float]) -> Config:
    """Return a fresh Config with `params` applied. Never mutate `cfg` in place."""
    new = Config(copy.deepcopy(cfg.data))
    norm = _normalize_weights({
        "tech": params["tech"], "news": params["news"],
        "macro": params["macro"], "regime": params["regime"],
    })
    new.set("strategy.composite.tech_weight", norm["tech"])
    new.set("strategy.composite.news_weight", norm["news"])
    new.set("strategy.composite.macro_weight", norm["macro"])
    new.set("strategy.composite.regime_weight", norm["regime"])
    new.set("strategy.composite.threshold", params["threshold"])
    new.set("strategy.ev.min_expectancy", params["ev_min"])
    new.set("strategy.exit_flip_conviction", params["exit_flip"])
    return new


# --------------------------------------------------------------------------- #
# Trial runner                                                                 #
# --------------------------------------------------------------------------- #

def _run_trial(
    base_cfg: Config,
    events,
    params: dict[str, float],
    min_trades: int,
    max_dd_pct: float,
) -> TrialResult:
    cfg = _apply_params(base_cfg, params)
    harness = ReplayHarness(cfg, events)
    harness.run()
    snap = harness.metrics.snapshot()
    state = harness.account_state()

    net = float(state.get("balance", STARTING_CAPITAL)) - STARTING_CAPITAL
    trades = int(snap.get("order_fill", 0))
    dd = float(snap.get("account_drawdown_pct", 0.0))
    win_rate = 0.0 if trades == 0 else max(0.0, min(1.0, 0.5 + net / (trades * 50 + 1e-9)))
    passed = (trades >= min_trades) and (dd <= max_dd_pct) and (net > 0.0)

    score = net
    if dd > max_dd_pct:
        score -= 1000.0
    if trades < min_trades:
        score -= 500.0

    # Phase 3: real return streams
    trade_records = harness.trade_records()
    eq_returns = harness.equity_returns()
    if len(eq_returns) >= 2:
        is_sharpe = SharpeStats.from_returns(eq_returns).sharpe
    else:
        is_sharpe = 0.0

    harness.stop_all()

    return TrialResult(
        params=params, net_pnl=net, trades=trades, max_dd_pct=dd,
        win_rate=win_rate, passed=passed, score=score,
        trade_records=trade_records, equity_returns=eq_returns,
        is_sharpe=is_sharpe,
    )


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def trial_sharpes(results: list[TrialResult]) -> list[float]:
    return [r.is_sharpe for r in results]


def pbo_from_results(
    results: list[TrialResult],
    n_chunks: int = 8,
    embargo_pct: float = 0.01,
) -> float:
    """
    PBO across the tuning grid using per-recorded-event equity returns.

    All trials share the same recorded events, so equity curves are aligned by
    construction. We truncate to the shortest common length for safety.
    """
    from validation.cv import CombinatorialPurgedKFold

    cols = [r.equity_returns for r in results if len(r.equity_returns) > 0]
    if len(cols) < 2:
        return float("nan")
    T = min(len(c) for c in cols)
    if T < n_chunks * 2:
        return float("nan")
    R = np.column_stack([c[:T] for c in cols])
    cv = CombinatorialPurgedKFold(
        n_chunks=n_chunks, embargo_pct=embargo_pct, max_splits=200,
    )
    return probability_of_backtest_overfitting(R, cv).pbo


# --------------------------------------------------------------------------- #
# Top-level tune                                                               #
# --------------------------------------------------------------------------- #

def tune(
    events,
    base_cfg: Config | None = None,
    *,
    trials: int | None = None,
    seed: int = 7,
    min_trades: int = 5,
    max_dd_pct: float = 2.0,
    is_frac: float = 0.8,
    n_chunks_pbo: int = 8,
    embargo_pct: float = 0.01,
    run_oos: bool = True,
) -> TuneReport:
    """
    Run the grid on IS, rank, then re-run the winner on OOS.

    Returns a TuneReport with all trials, the DSR/PBO statistics, and the
    winner's OOS metrics (if `run_oos` and the history is long enough).
    """
    base_cfg = base_cfg or Config.load()
    if not 0.5 <= is_frac < 1.0:
        raise ValueError("is_frac must be in [0.5, 1.0)")

    # --- split -------------------------------------------------------- #
    n = len(events)
    is_end = max(1, int(n * is_frac))
    is_events = events[:is_end]
    oos_events = events[is_end:]
    log.info("tune: IS=%d events, OOS=%d events (is_frac=%.2f)",
             len(is_events), len(oos_events), is_frac)

    # --- enumerate grid ---------------------------------------------- #
    keys = list(_GRID)
    full = [dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(_GRID[k] for k in keys))]
    if trials is not None and trials < len(full):
        import random
        rng = random.Random(seed)
        rng.shuffle(full)
        full = full[:trials]

    # --- run trials on IS -------------------------------------------- #
    results: list[TrialResult] = []
    for params in full:
        results.append(_run_trial(base_cfg, is_events, params, min_trades, max_dd_pct))
    results.sort(key=lambda r: r.score, reverse=True)

    # --- IS statistics ----------------------------------------------- #
    best = _best_passing(results)
    sharpes = trial_sharpes(results)
    is_sharpe = best.is_sharpe if best is not None else 0.0

    # DSR: deflate the winner's IS Sharpe against the trial-Sharpe distribution.
    if best is not None and len(sharpes) >= 2:
        dsr_res: DSRResult = deflated_sharpe_ratio(best.equity_returns, sharpes)
        dsr, sr0, psr0 = dsr_res.dsr, dsr_res.sr0, dsr_res.psr_vs_zero
    else:
        dsr, sr0, psr0 = 0.0, 0.0, 0.0

    pbo = pbo_from_results(results, n_chunks=n_chunks_pbo, embargo_pct=embargo_pct)

    # --- OOS re-run of the winner ------------------------------------ #
    oos_net = oos_dd = oos_sharpe = None
    oos_trades = None
    if run_oos and best is not None and len(oos_events) > 10:
        cfg_oos = _apply_params(base_cfg, best.params)
        h_oos = ReplayHarness(cfg_oos, oos_events)
        h_oos.run()
        snap_oos = h_oos.metrics.snapshot()
        st_oos = h_oos.account_state()
        oos_net = float(st_oos.get("balance", STARTING_CAPITAL)) - STARTING_CAPITAL
        oos_dd = float(snap_oos.get("account_drawdown_pct", 0.0))
        oos_trades = int(snap_oos.get("order_fill", 0))
        eq_oos = h_oos.equity_returns()
        oos_sharpe = (
            SharpeStats.from_returns(eq_oos).sharpe if len(eq_oos) >= 2 else 0.0
        )
        h_oos.stop_all()

    return TuneReport(
        results=results, best=best, n_trials=len(results), is_frac=is_frac,
        is_sharpe=is_sharpe, sr0=sr0, dsr=dsr, psr_vs_zero=psr0, pbo=pbo,
        oos_net=oos_net, oos_dd=oos_dd, oos_sharpe=oos_sharpe, oos_trades=oos_trades,
        oos_events=len(oos_events),
    )


def _best_passing(results: list[TrialResult]) -> TrialResult | None:
    for r in results:
        if r.passed:
            return r
    return None


best_passing = _best_passing  # public alias


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #

def write_tuned_config(
    result: TrialResult,
    out: str = "config/tuned.yaml",
    *,
    strict: bool = False,
    dsr: float = float("nan"),
    pbo: float = float("nan"),
    min_dsr: float = 0.0,
    max_pbo: float = 0.5,
) -> str:
    """
    Write the winning config as YAML.

    `strict=True` raises if DSR/PBO breach the thresholds. Default is now
    `strict=False` — the gate lives at validate time, so the researcher can
    always inspect the winner even if it fails the deflation test.
    """
    import yaml as _yaml

    if strict:
        if dsr == dsr and dsr < min_dsr:
            raise RuntimeError(f"tuned config rejected: DSR={dsr:.3f} < {min_dsr}")
        if pbo == pbo and pbo > max_pbo:
            raise RuntimeError(f"tuned config rejected: PBO={pbo:.3f} > {max_pbo}")

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
            "exit_flip_conviction": result.params["exit_flip"],
        }
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        _yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)
    return str(path)


def _trial_to_jsonable(r: TrialResult) -> dict:
    return {
        "params": r.params,
        "net_pnl": r.net_pnl,
        "trades": r.trades,
        "max_dd_pct": r.max_dd_pct,
        "win_rate": r.win_rate,
        "passed": r.passed,
        "score": r.score,
        "is_sharpe": r.is_sharpe,
        # trade_records kept (small); equity_returns dropped (large + derivable).
    }


def write_meta(report: TuneReport, out: str = "config/tuned.yaml.meta.json") -> str:
    """Persist the TuneReport (minus the bulky equity arrays) as JSON."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "n_trials": report.n_trials,
        "is_frac": report.is_frac,
        "is_sharpe": report.is_sharpe,
        "sr0": report.sr0,
        "dsr": report.dsr,
        "psr_vs_zero": report.psr_vs_zero,
        "pbo": report.pbo,
        "oos_net": report.oos_net,
        "oos_dd": report.oos_dd,
        "oos_sharpe": report.oos_sharpe,
        "oos_trades": report.oos_trades,
        "oos_events": report.oos_events,
        "trial_sharpes": trial_sharpes(report.results),
        "best_params": report.best.params if report.best else None,
        "results": [_trial_to_jsonable(r) for r in report.results],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, default=str)
    return str(path)


@dataclass
class TuneMeta:
    """Lightweight view of a written meta file."""
    n_trials: int
    is_frac: float
    is_sharpe: float
    sr0: float
    dsr: float
    psr_vs_zero: float
    pbo: float
    oos_net: float | None
    oos_dd: float | None
    oos_sharpe: float | None
    oos_trades: int | None
    oos_events: int
    best_params: dict | None


def load_meta(path: str | Path) -> TuneMeta:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    return TuneMeta(
        n_trials=int(d.get("n_trials", 0)),
        is_frac=float(d.get("is_frac", 0.0)),
        is_sharpe=float(d.get("is_sharpe", 0.0)),
        sr0=float(d.get("sr0", 0.0)),
        dsr=float(d.get("dsr", 0.0)),
        psr_vs_zero=float(d.get("psr_vs_zero", 0.0)),
        pbo=float(d.get("pbo", float("nan"))),
        oos_net=d.get("oos_net"),
        oos_dd=d.get("oos_dd"),
        oos_sharpe=d.get("oos_sharpe"),
        oos_trades=d.get("oos_trades"),
        oos_events=int(d.get("oos_events", 0)),
        best_params=d.get("best_params"),
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/sample.jsonl")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--max-dd", type=float, default=2.0)
    ap.add_argument("--is-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="config/tuned.yaml")
    args = ap.parse_args()

    from data_ingestion.generate_sample import load_jsonl

    events = load_jsonl(args.history)
    report = tune(
        events,
        trials=args.trials,
        min_trades=args.min_trades,
        max_dd_pct=args.max_dd,
        is_frac=args.is_frac,
        seed=args.seed,
    )

    print(f"\nTuned {report.n_trials} trials against {len(events)} events "
          f"(IS frac={report.is_frac:.2f}, OOS events={report.oos_events}).")
    print(f"{'rank':>4} {'tech':>5} {'news':>5} {'macro':>5} {'reg':>4} "
          f"{'thr':>4} {'evMin':>5} {'net':>10} {'trades':>6} {'dd%':>6} {'PASS':>4}")
    for i, r in enumerate(report.results[:12], 1):
        print(f"{i:>4} {r.params['tech']:>5} {r.params['news']:>5} "
              f"{r.params['macro']:>5} {r.params['regime']:>4} "
              f"{r.params['threshold']:>4} {r.params['ev_min']:>5} "
              f"{r.net_pnl:>10.2f} {r.trades:>6} {r.max_dd_pct:>6.2f} "
              f"{str(r.passed):>4}")

    if report.best is not None:
        print(f"\nBEST (IS): net={report.best.net_pnl:.2f} "
              f"trades={report.best.trades} dd={report.best.max_dd_pct:.2f}%")
        out = write_tuned_config(report.best, args.out)
        meta_path = write_meta(report, args.out + ".meta.json")
        print(f"Wrote tuned config -> {out}")
        print(f"Wrote tuning meta  -> {meta_path}")
    else:
        print("\nNo trial cleared the IS gate (positive expectancy + DD within limit + min trades).")

    print(f"\n  IS Sharpe   : {report.is_sharpe:.4f}")
    print(f"  SR0 (null)  : {report.sr0:.4f}  (expected max of {report.n_trials} trials)")
    print(f"  DSR         : {report.dsr:.4f}")
    print(f"  PBO         : {report.pbo:.4f}")
    if report.oos_net is not None:
        print(f"  OOS net     : {report.oos_net:+.2f}  dd={report.oos_dd:.2f}%  "
              f"sharpe={report.oos_sharpe:.4f}  trades={report.oos_trades}")


if __name__ == "__main__":
    main()
