"""Tests for the Market Context Engine (instrument-specific intelligence)."""
import sys
from pathlib import Path

import numpy as np
import pytest

from strategy_engine.instrument_context import ContextEngine, build_instrument_context, get_profile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A rolling series of prices so the vol/trend helpers have input
_PRICES = [100 + 0.5 * i for i in range(30)]
_RETURNS = [np.log(_PRICES[i] / _PRICES[i - 1]) for i in range(1, len(_PRICES))]


def test_get_profile_known():
    p = get_profile("XAUUSD")
    assert p["label"] == "XAUUSD Gold"
    assert p["sensitivity"]["macro"] == 0.7
    p = get_profile("btcusd")
    assert p["label"] == "BTCUSD"
    p = get_profile("unknown_sym")
    assert p["label"] == "Generic"


def test_build_instrument_context_shape():
    ctx = build_instrument_context("EURUSD", _PRICES, _RETURNS)
    assert ctx["symbol"] == "EURUSD"
    assert set(ctx["profile"]["sensitivity"]) == {"macro", "tech", "news"}
    assert ctx["vol"] >= 0.0
    assert ctx["trend"] >= 0.0  # upward drift in the series
    assert "vol_ewma_factor" in ctx
    assert "meta" in ctx


def test_build_instrument_context_profile_override():
    ctx = build_instrument_context("XAUUSD", _PRICES, _RETURNS, meta={"source": "custom.json"})
    assert ctx["meta"]["source"] == "custom.json"
    assert ctx["profile"]["label"] == "XAUUSD Gold"


def test_build_instrument_context_short_series():
    ctx = build_instrument_context("BTCUSD", [1.0, 1.05, 1.07], [0.0, 0.047, 0.0185])
    assert ctx["vol"] >= 0.0
    assert ctx["trend"] >= 0.0
    assert ctx["profile"]["label"] == "BTCUSD"
