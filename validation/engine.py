# validation/engine.py
"""
Validation & research harness (architecture §7.3 + §8).

Before ANY strategy goes live it must pass, in order:
  1. Backtest on historical data
  2. Walk-forward validation
  3. Monte Carlo stress testing
  4. Paper trading (the live paper broker)
  5. Small live capital deployment

Phase 3 additions:
  - Purged, embargoed cross-validation (López de Prado, AFML ch. 7)
  - Deflated Sharpe Ratio + PBO (Bailey & López de Prado)
  - Execution-aware backtest with deterministic latency, fat-tailed slippage,
    spread, commission, and probabilistic partial fills.

This module provides deterministic, dependency-light implementations of (1)-(3) plus the
replay harness (§8) that re-plays exact recorded history (ticks, candles, news) through the
real strategy engine to compare expected vs actual behaviour and measure slippage.

The harness reuses the *actual* pipeline components (QuantAnalytics, SignalFusion, EV engine,
RiskVeto) by wiring them onto a replay-mode MessageBus — so validation tests the real code,
not a reimplementation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from validation.execution_model import ExecutionModel, build_execution_model

log = logging.getLogger("validation.engine")


@dataclass
class Trade:
    entry_t: float
    exit_t: float
    side: int           # +1 long, -1 short
    entry: float
    exit: float
    pnl: float
    reason: str = ""


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    n: int = 0

    def summary(self) -> Dict[str, float]:
        return {
            "n_trades": self.n,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 3),
        }


@dataclass
class Fold:
    """One walk-forward fold result."""
    win_rate: float
    n_trades: int
    returns: np.ndarray


@dataclass
class FoldSummary:
    """Aggregated across folds."""
    n_folds: int
    avg_oos_win_rate: float
    oos_returns: np.ndarray


def summarize_folds(folds: List[Fold]) -> FoldSummary:
    """Aggregate fold results into a single summary."""
    if not folds:
        return FoldSummary(0, 0.0, np.array([]))
    wr = float(np.mean([f.win_rate for f in folds]))
    oos = np.concatenate([np.asarray(f.returns, dtype=float) for f in folds]) if folds else np.array([])
    return FoldSummary(len(folds), wr, oos)


def vectorized_backtest(
    prices: np.ndarray,
    signal: np.ndarray,
    *,
    initial_capital: float = 10_000.0,
    fee_pct: float = 0.0005,
    position_pct: float = 2.0,
    exec_model: Optional[ExecutionModel] = None,
    symbol: str = "X",
) -> BacktestResult:
    """Simple long/flat/short backtest driven by a precomputed signal in [-1,1].

    signal[i] > 0 -> long, < 0 -> short, ~0 -> flat. Position size = position_pct of equity.
    This is the *vectorized* reference used to calibrate the EV engine and to sanity-check the
    event-driven engine output against (architecture §8: "vectorized backtesting on existing
    history" first).

    Phase 3: applies exec_model (latency, slippage, spread, commission, partial fills)
    to each fill if provided.
    """
    equity = initial_capital
    peak = equity
    eq_curve: List[float] = [equity]
    trades: List[Trade] = []
    pos = 0
    entry_price = 0.0
    entry_t = 0.0
    n = min(len(prices), len(signal) + 1) if len(signal) else len(prices)

    rep = None
    for i in range(1, n):
        s = signal[i] if i < len(signal) else 0.0
        # close prior position at signal flip
        if pos != 0 and np.sign(s) != np.sign(pos) and s != 0:
            exit_px = prices[i]
            if exec_model is not None:
                rep = exec_model.apply(
                    symbol=symbol, side="sell" if pos > 0 else "buy",
                    mid_price=prices[i], qty=1.0, ts=float(i),
                )
                exit_px = rep.fill_price
            ret = (exit_px - entry_price) / entry_price * pos
            size = equity * (position_pct / 100.0)
            if exec_model is not None:
                size *= rep.filled_qty  # partial fill scaling
            pnl = size * (ret - fee_pct * 2) - (rep.commission if exec_model else 0.0)
            equity += pnl
            trades.append(Trade(entry_t, float(i), int(pos), entry_price, exit_px, pnl, "flip"))
            peak = max(peak, equity)
            pos = 0
        # open new position
        if s != 0 and pos == 0:
            fill_px = prices[i]
            if exec_model is not None:
                rep = exec_model.apply(
                    symbol=symbol, side="buy" if s > 0 else "sell",
                    mid_price=prices[i], qty=1.0, ts=float(i),
                )
                fill_px = rep.fill_price
            pos = int(np.sign(s))
            entry_price = fill_px
            entry_t = float(i)
        eq_curve.append(equity)
    # close at end
    if pos != 0:
        exit_px = prices[-1]
        commission = 0.0
        if exec_model is not None:
            rep = exec_model.apply(
                symbol=symbol, side="sell" if pos > 0 else "buy",
                mid_price=prices[-1], qty=1.0, ts=float(len(prices) - 1),
            )
            exit_px = rep.fill_price
            commission = rep.commission
        ret = (exit_px - entry_price) / entry_price * pos
        size = equity * (position_pct / 100.0)
        if exec_model is not None:
            size *= rep.filled_qty
        pnl = size * (ret - fee_pct * 2) - commission
        equity += pnl
        trades.append(Trade(entry_t, float(len(prices) - 1), int(pos), entry_price, exit_px, pnl, "eod"))
    eq = np.array(eq_curve)
    rets = np.diff(eq) / np.where(eq[:-1] == 0, 1e-9, eq[:-1])
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100.0) if len(eq) else 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return BacktestResult(
        trades=trades, equity_curve=eq_curve, total_pnl=equity - initial_capital,
        win_rate=(wins / len(trades)) if trades else 0.0, max_drawdown_pct=dd,
        sharpe=sharpe, n=len(trades),
    )


def walk_forward(
    prices,
    strategy_fn,
    *,
    n_splits: int = 5,
    train_pct: float = 0.5,
    test_pct: float = 0.1,
    step_pct: float = 0.1,
    embargo_pct: float = 0.01,
    horizon: int = 1,
    initial_capital: float = 10_000.0,
    exec_model: Optional[ExecutionModel] = None,
    label_end_idx=None,
) -> List[Fold]:
    """
    Purged, embargoed walk-forward.

    Returns one Fold per split, each with .win_rate, .n_trades, .returns.

    ``strategy_fn(train_arr, test_arr)`` returns a signal array aligned to ``test_arr``.
    Uses PurgedWalkForward internally so no label leaks across the train/test boundary.
    """
    from validation.cv import PurgedWalkForward

    arr = np.asarray(prices, dtype=float)
    T = len(arr)
    splitter = PurgedWalkForward(
        n_splits=n_splits, train_pct=train_pct, test_pct=test_pct,
        step_pct=step_pct, embargo_pct=embargo_pct, horizon=horizon,
        label_end_idx=label_end_idx,
    )
    folds: List[Fold] = []
    for split in splitter.split(T):
        tr, te = split.train_idx, split.test_idx
        sig = strategy_fn(arr[tr], arr[te])
        sig = np.asarray(sig, dtype=float)
        if len(sig) < 2:
            continue
        # Apply execution model to the test segment if available
        bt = vectorized_backtest(arr[te], sig, initial_capital=initial_capital,
                                  exec_model=exec_model, symbol="X")
        rets = np.diff(np.asarray(bt.equity_curve, dtype=float)) if len(bt.equity_curve) > 1 else np.array([0.0])
        wins = (rets > 0).sum()
        folds.append(Fold(win_rate=wins / max(1, len(rets)), n_trades=bt.n, returns=rets))
    return folds


def monte_carlo(
    trades: List[Trade],
    *,
    n_paths: int = 1000,
    block_size: int = 10,
    initial_capital: float = 10_000.0,
    seed: int = 123,
) -> Dict[str, float]:
    """Block-bootstrap the trade P&L distribution to estimate tail risk (architecture §7.3.3).

    Returns percentile bands of final equity and max drawdown across ``n_paths`` resamples.
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades]) if trades else np.array([0.0])
    paths_eq: List[float] = []
    paths_dd: List[float] = []
    for _ in range(n_paths):
        eq = initial_capital
        peak = eq
        max_dd = 0.0
        # block bootstrap
        idx = rng.integers(0, len(pnls), size=max(1, len(pnls) // block_size + 1))
        flat = pnls[idx].flatten()[: len(pnls)] if len(pnls) else np.array([0.0])
        for p in flat:
            eq += p
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        paths_eq.append(eq)
        paths_dd.append(max_dd)
    eq_arr = np.array(paths_eq)
    dd_arr = np.array(paths_dd)
    return {
        "eq_p5": float(np.percentile(eq_arr, 5)),
        "eq_p50": float(np.percentile(eq_arr, 50)),
        "eq_p95": float(np.percentile(eq_arr, 95)),
        "dd_p95": float(np.percentile(dd_arr, 95)),
        "ruin_prob": float((eq_arr <= initial_capital * 0.5).mean()),
    }
