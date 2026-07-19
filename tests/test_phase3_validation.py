# tests/test_phase3_validation.py
"""Phase 3 validation tests: CV splitters, DSR/PBO, execution model."""
import numpy as np
import pytest

from validation.cv import PurgedWalkForward, CombinatorialPurgedKFold
from validation.metrics import (
    SharpeStats, probabilistic_sharpe_ratio, expected_max_sharpe_under_null,
    deflated_sharpe_ratio, probability_of_backtest_overfitting,
)
from validation.execution_model import ExecutionModel


# ---------------- CV splitters --------------------------------------------- #


def test_purged_walk_forward_no_overlap_and_embargo():
    T = 1000
    splitter = PurgedWalkForward(n_splits=4, train_pct=0.5, test_pct=0.1,
                                 step_pct=0.1, embargo_pct=0.02, horizon=5)
    splits = list(splitter.split(T))
    assert len(splits) == 4
    for s in splits:
        # No train index falls inside the test window
        assert not np.intersect1d(s.train_idx, s.test_idx).size
        # Embargo: no train index in (test_max, test_max + embargo)
        embargo = int(np.floor(0.02 * T))
        if embargo:
            bad = ((s.train_idx > s.test_idx.max()) &
                   (s.train_idx <= s.test_idx.max() + embargo))
            assert not bad.any()


def test_purged_walk_forward_label_horizon_removes_adjacent_train():
    T = 500
    splitter = PurgedWalkForward(n_splits=3, train_pct=0.6, test_pct=0.1,
                                 step_pct=0.1, embargo_pct=0.0, horizon=10)
    for s in splitter.split(T):
        # Any train index whose label window reaches into test must be gone.
        train_max = s.train_idx.max()
        assert train_max < s.test_idx.min() - 10


def test_combinatorial_cv_yields_expected_count():
    T = 800
    cv = CombinatorialPurgedKFold(n_chunks=8, embargo_pct=0.01, horizon=1)
    splits = list(cv.split(T))
    # C(8, 4) = 70 partitions; some are dropped by purge, so expect fewer
    assert 40 <= len(splits) <= 70


# ---------------- Metrics -------------------------------------------------- #


def test_sharpe_stats_on_known_series():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=1000)
    st = SharpeStats.from_returns(r)
    assert 0.05 < st.sharpe < 0.15
    assert st.n == 1000


def test_psr_increases_with_sample_size():
    # Same Sharpe, larger N -> PSR should be closer to 1
    psr_small = probabilistic_sharpe_ratio(0.05, 0.0, n=100, skew=0.0, kurt=3.0)
    psr_large = probabilistic_sharpe_ratio(0.05, 0.0, n=2000, skew=0.0, kurt=3.0)
    assert psr_large > psr_small


def test_expected_max_sharpe_grows_with_trials():
    s1 = expected_max_sharpe_under_null(1, 0.05)
    s10 = expected_max_sharpe_under_null(10, 0.05)
    s100 = expected_max_sharpe_under_null(100, 0.05)
    assert s1 == 0.0
    assert s10 > 0.0
    assert s100 > s10


def test_dsr_deflates_for_many_trials():
    rng = np.random.default_rng(1)
    # Null strategy: zero-mean returns
    r = rng.normal(0.0, 0.01, size=500)
    trials_few = [0.0, 0.0]                       # low trial std, few trials
    trials_many = [0.05] * 200                    # many trials -> higher SR0
    d1 = deflated_sharpe_ratio(r, trials_few)
    d2 = deflated_sharpe_ratio(r, trials_many)
    assert d2.dsr <= d1.dsr + 1e-9


# ---------------- PBO ------------------------------------------------------ #


def test_pbo_is_high_for_pure_noise():
    rng = np.random.default_rng(2)
    # 40 noise strategies, 600 obs: IS-best has no OOS edge.
    R = rng.normal(0.0, 0.01, size=(600, 40))
    cv = CombinatorialPurgedKFold(n_chunks=8, embargo_pct=0.0, horizon=1, max_splits=80)
    res = probability_of_backtest_overfitting(R, cv)
    assert res.n_combinations >= 40
    assert res.pbo >= 0.4       # expected ~0.5


def test_pbo_is_low_for_real_persistent_edge():
    rng = np.random.default_rng(3)
    T, N = 600, 40
    R = rng.normal(0.0, 0.01, size=(T, N))
    # give strategy 0 a genuine persistent edge
    R[:, 0] += 0.002
    cv = CombinatorialPurgedKFold(n_chunks=8, embargo_pct=0.0, horizon=1, max_splits=80)
    res = probability_of_backtest_overfitting(R, cv)
    assert res.pbo < 0.25


# ---------------- Execution model ----------------------------------------- #


def test_execution_model_is_deterministic():
    em = ExecutionModel(seed=42)
    a = em.apply(symbol="EURUSD", side="buy", mid_price=1.10, qty=1.0, ts=100.0)
    b = em.apply(symbol="EURUSD", side="buy", mid_price=1.10, qty=1.0, ts=100.0)
    assert a.fill_price == b.fill_price
    assert a.filled_qty == b.filled_qty


def test_execution_model_buy_pays_up_sell_pays_down():
    em = ExecutionModel(seed=7, slippage_bps=0.0, spread_bps=2.0,
                        latency_jitter_ms=0.0, partial_fill_prob=0.0)
    buy = em.apply(symbol="X", side="buy", mid_price=100.0, qty=1.0, ts=1.0)
    sell = em.apply(symbol="X", side="sell", mid_price=100.0, qty=1.0, ts=1.0)
    assert buy.fill_price > 100.0 > sell.fill_price


def test_execution_model_commission_is_positive():
    em = ExecutionModel(seed=8)
    r = em.apply(symbol="X", side="buy", mid_price=100.0, qty=10.0, ts=1.0)
    assert r.commission > 0.0


def test_execution_model_large_orders_more_likely_partial():
    em = ExecutionModel(seed=9, reference_notional=1000.0, partial_fill_prob=0.05)
    small_partial = sum(
        em.apply(symbol="X", side="buy", mid_price=100.0, qty=0.5, ts=float(i)).partial
        for i in range(200)
    )
    large_partial = sum(
        em.apply(symbol="X", side="buy", mid_price=100.0, qty=50.0, ts=float(i)).partial
        for i in range(200)
    )
    assert large_partial > small_partial
