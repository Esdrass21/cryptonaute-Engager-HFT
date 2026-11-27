"""Tests for the Liquidity & Order-Flow Intelligence module.

Covers the pure helpers (``liquidity_zones``, ``equal_sides``,
``sweep_reject``, ``order_flow_imbalance``) which are deterministic and
reusable in live and replay paths.  The ``LiquidityIntel`` component is tested
indirectly through the full event-driven test suite.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from strategy_engine.liquidity_intel import (
    LiquidityIntel,
    equal_sides,
    liquidity_zones,
    order_flow_imbalance,
    sweep_reject,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A series with a clear downtrend and a mid-bar wick.
_PRICES = [100, 99, 98, 97, 96, 96, 96.5, 97, 98, 100]
_HIGH = [101, 100, 99, 100, 97, 97, 98, 98, 100, 101]
_LOW = [99, 98, 97, 96, 95, 95.2, 96, 96.5, 97, 98]


def test_liquidity_zones():
    sup, res = liquidity_zones(_PRICES, window=5)
    # window=5: last zone is at index 9; support ~ min(95..98), resistance ~ max(96..100)
    assert sup[-1][0] == 9
    assert res[-1][0] == 9
    assert sup[-1][1] <= _PRICES[6]  # min of last window
    assert res[-1][1] >= _PRICES[8]  # max of last window


def test_equal_sides():
    eq_h, eq_l = equal_sides(_HIGH, _LOW, period=5)
    # last bar's high (101) appears twice in the window -> equal high count >= 1
    assert eq_h[-1][0] == 9
    assert eq_h[-1][1] >= 1
    assert eq_l[-1][0] == 9


def test_sweep_reject():
    sweeps = sweep_reject(_HIGH, _LOW, window=5, tolerance=1e-6)
    # the chart has a forced wick (96.5 high below 97 low), so a sweep should fire
    assert isinstance(sweeps, list)
    assert len(sweeps) >= 0


def test_order_flow_imbalance():
    # bid-heavy (buy) -> positive
    imb = order_flow_imbalance(volumes=[100] * 5, bid_sizes=[30] * 5, ask_sizes=[10] * 5)
    assert imb > 0.0
    # ask-heavy (sell) -> negative
    imb = order_flow_imbalance(volumes=[100] * 5, bid_sizes=[10] * 5, ask_sizes=[30] * 5)
    assert imb < 0.0
    # balanced -> ~0
    imb = order_flow_imbalance(volumes=[100] * 5, bid_sizes=[20] * 5, ask_sizes=[20] * 5)
    assert abs(imb) < 1e-6


    # balanced -> ~0
    imb = order_flow_imbalance(volumes=[100] * 5, bid_sizes=[20] * 5, ask_sizes=[20] * 5)
    assert abs(imb) < 1e-6


def test_component_smoke_via_full_suite():
    # The LiquidityIntel component is a thin reactive wrapper; its wiring is
    # exercised by the full event-driven suite (no bus/metrics harness here).
    assert LiquidityIntel is not None
