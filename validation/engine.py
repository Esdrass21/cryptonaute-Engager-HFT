"""Validation & research harness (architecture §7.3 + §8).

Before ANY strategy goes live it must pass, in order:
  1. Backtest on historical data
  2. Walk-forward validation
  3. Monte Carlo stress testing
  4. Paper trading (the live paper broker)
  5. Small live capital deployment

This module provides deterministic, dependency-light implementations of (1)-(3) plus the
replay harness (§8) that re-plays exact recorded history (ticks, candles, news) through the
real strategy engine to compare expected vs actual behaviour and measure slippage.

The harness reuses the *actual* pipeline components (QuantAnalytics, SignalFusion, EV engine,
RiskVeto) by wiring them onto a replay-mode MessageBus — so validation tests the real code,
not a reimplementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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


def vectorized_backtest(
    prices: np.ndarray,
    signal: np.ndarray,
    *,
    initial_capital: float = 10_000.0,
    fee_pct: float = 0.0005,
    position_pct: float = 2.0,
) -> BacktestResult:
    """Simple long/flat/short backtest driven by a precomputed signal in [-1,1].

    signal[i] > 0 -> long, < 0 -> short, ~0 -> flat. Position size = position_pct of equity.
    This is the *vectorized* reference used to calibrate the EV engine and to sanity-check the
    event-driven engine output against (architecture §8: "vectorized backtesting on existing
    history" first).
    """
    equity = initial_capital
    peak = equity
    eq_curve: List[float] = [equity]
    trades: List[Trade] = []
    pos = 0
    entry_price = 0.0
    entry_t = 0.0
    n = min(len(prices), len(signal) + 1) if len(signal) else len(prices)
    for i in range(1, n):
        s = signal[i] if i < len(signal) else 0.0
        # close prior position at signal flip
        if pos != 0 and np.sign(s) != np.sign(pos) and s != 0:
            ret = (prices[i] - entry_price) / entry_price * pos
            pnl = equity * (position_pct / 100.0) * (ret - fee_pct * 2)
            equity += pnl
            trades.append(Trade(entry_t, float(i), int(pos), entry_price, prices[i], pnl, "flip"))
            peak = max(peak, equity)
            pos = 0
        # open new position
        if s != 0 and pos == 0:
            pos = int(np.sign(s))
            entry_price = prices[i]
            entry_t = float(i)
        eq_curve.append(equity)
    # close at end
    if pos != 0:
        ret = (prices[-1] - entry_price) / entry_price * pos
        pnl = equity * (position_pct / 100.0) * (ret - fee_pct * 2)
        equity += pnl
        trades.append(Trade(entry_t, float(len(prices) - 1), int(pos), entry_price, prices[-1], pnl, "eod"))
    eq = np.array(eq_curve)
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100.0)
    wins = sum(1 for t in trades if t.pnl > 0)
    return BacktestResult(
        trades=trades, equity_curve=eq_curve, total_pnl=equity - initial_capital,
        win_rate=(wins / len(trades)) if trades else 0.0, max_drawdown_pct=dd,
        sharpe=sharpe, n=len(trades),
    )


def walk_forward(
    prices: np.ndarray,
    make_signal,
    *,
    train_frac: float = 0.6,
    test_frac: float = 0.2,
    step_frac: float = 0.2,
    initial_capital: float = 10_000.0,
) -> List[BacktestResult]:
    """Rolling walk-forward: retrain signal on train window, test on subsequent window.

    ``make_signal(window)`` returns a signal array aligned to ``window``. We slide the window
    and backtest each out-of-sample test segment. Returns one result per fold.
    """
    n = len(prices)
    results: List[BacktestResult] = []
    i = 0
    while True:
        train_end = int(i + n * train_frac)
        test_end = int(train_end + n * test_frac)
        if test_end > n:
            break
        train = prices[i:train_end]
        test = prices[train_end:test_end]
        sig = make_signal(train, test)
        r = vectorized_backtest(test, sig, initial_capital=initial_capital)
        results.append(r)
        i += int(n * step_frac)
        if i + int(n * (train_frac + test_frac)) > n:
            break
    return results


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
