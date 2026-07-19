# validation/metrics.py
"""
Risk-adjusted performance metrics with multiple-testing correction.

Implements:
- Sharpe ratio (annualized)
- Probabilistic Sharpe Ratio (PSR) vs a benchmark SR*
- Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014)
- Probability of Backtest Overfitting (PBO) — Bailey, Borwein, López de Prado, Zhu

No scipy dependency: uses math.erf and Acklam's inverse-normal-CDF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Normal CDF / PPF (no scipy)                                                  #
# --------------------------------------------------------------------------- #

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF. Max relative error ~1.15e-9."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


# --------------------------------------------------------------------------- #
# Sharpe / PSR / DSR                                                           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SharpeStats:
    sharpe: float          # per-observation (not annualized)
    skew: float
    kurtosis: float        # raw kurtosis (3.0 = normal)
    n: int
    annualized: float      # sharpe * sqrt(periods_per_year) if caller passes it

    @classmethod
    def from_returns(cls, returns: Sequence[float], periods_per_year: int = 252) -> "SharpeStats":
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 2:
            return cls(0.0, 0.0, 3.0, n, 0.0)
        mu = r.mean()
        sd = r.std(ddof=1)
        if sd == 0:
            return cls(0.0, 0.0, 3.0, n, 0.0)
        sr = mu / sd
        m2 = ((r - mu) ** 2).mean()
        m3 = ((r - mu) ** 3).mean()
        m4 = ((r - mu) ** 4).mean()
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
        kurt = m4 / (m2 ** 2) if m2 > 0 else 3.0
        return cls(sr, float(skew), float(kurt), n,
                   float(sr * math.sqrt(periods_per_year)))


def probabilistic_sharpe_ratio(
    sr_hat: float, sr_benchmark: float, n: int, skew: float, kurt: float,
) -> float:
    """
    PSR = Φ[ (SR_hat - SR*) * sqrt(N-1) / sqrt(1 - γ3·SR_hat + (γ4-1)/4·SR_hat²) ]
    Returns P(true SR > SR*) under non-normality.
    """
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
    if denom_sq <= 0:
        return 0.0
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def expected_max_sharpe_under_null(n_trials: int, trials_sharpe_std: float) -> float:
    """
    E[max SR] under the null of zero true SR across `n_trials` independent trials
    (Bailey & López de Prado 2014, eq. 8).

        SR_0 = sqrt(V[SR]) · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    """
    if n_trials <= 1 or trials_sharpe_std <= 0:
        return 0.0
    N = float(n_trials)
    z1 = _norm_ppf(1.0 - 1.0 / N)
    z2 = _norm_ppf(1.0 - 1.0 / (N * math.e))
    return trials_sharpe_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


@dataclass(frozen=True)
class DSRResult:
    dsr: float                  # the deflated Sharpe (probability)
    sr_hat: float               # observed per-obs Sharpe
    sr0: float                  # expected max SR under null
    psr_vs_zero: float          # PSR against SR*=0
    n: int
    n_trials: int


def deflated_sharpe_ratio(
    returns: Sequence[float],
    trials_sharpes: Sequence[float],
    periods_per_year: int = 252,
) -> DSRResult:
    """
    DSR = PSR evaluated with SR* = E[max SR | N trials].

    `trials_sharpes` are the per-observation Sharpes of every configuration you
    tried during tuning (including the losers). Passing them in is how we
    account for multiple testing; omitting them under-reports the deflation.
    """
    stats = SharpeStats.from_returns(returns, periods_per_year)
    ts = np.asarray(trials_sharpes, dtype=float)
    ts = ts[np.isfinite(ts)]
    n_trials = max(1, len(ts))
    trials_std = float(ts.std(ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe_under_null(n_trials, trials_std)
    dsr = probabilistic_sharpe_ratio(stats.sharpe, sr0, stats.n, stats.skew, stats.kurtosis)
    psr0 = probabilistic_sharpe_ratio(stats.sharpe, 0.0, stats.n, stats.skew, stats.kurtosis)
    return DSRResult(dsr=dsr, sr_hat=stats.sharpe, sr0=sr0,
                     psr_vs_zero=psr0, n=stats.n, n_trials=n_trials)


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (PBO)                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PBOResult:
    pbo: float             # in [0, 1]; lower is better
    n_combinations: int
    n_strategies: int


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    cv,
    metric: str = "sharpe",
) -> PBOResult:
    """
    PBO via Combinatorial Symmetric Cross-Validation.

    `returns_matrix` is (T x N): one column per strategy/config, one row per
    observation. `cv` is a CombinatorialPurgedKFold instance.

    For each train/test partition:
      - rank strategies by metric on train (IS)
      - identify the IS-best
      - compute its rank on test (OOS) — as a percentile in [0, 1]
    PBO = fraction of partitions where the IS-best lands *below median* OOS.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    T, N = R.shape
    if N < 2:
        raise ValueError("need at least 2 strategies for PBO")

    def _score(col: np.ndarray) -> float:
        col = col[np.isfinite(col)]
        if len(col) < 2 or col.std(ddof=1) == 0:
            return -math.inf
        if metric == "sharpe":
            return col.mean() / col.std(ddof=1)
        if metric == "mean":
            return col.mean()
        raise ValueError(f"unknown metric {metric!r}")

    below_median = 0
    total = 0
    for split in cv.split(T):
        tr, te = split.train_idx, split.test_idx
        if len(tr) < 2 or len(te) < 2:
            continue
        is_scores = np.array([_score(R[tr, j]) for j in range(N)])
        oos_scores = np.array([_score(R[te, j]) for j in range(N)])
        if not np.isfinite(is_scores).any() or not np.isfinite(oos_scores).any():
            continue
        best_is = int(np.nanargmax(is_scores))
        # percentile rank of OOS performance across all strategies (0=worst, 1=best)
        order = np.argsort(np.argsort(oos_scores))  # 0..N-1
        rank_pct = order[best_is] / (N - 1)
        if rank_pct < 0.5:
            below_median += 1
        total += 1

    pbo = below_median / total if total else float("nan")
    return PBOResult(pbo=float(pbo), n_combinations=total, n_strategies=N)
