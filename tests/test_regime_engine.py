"""Unit tests for the Adaptive Market Regime Intelligence Engine.

Covers the pure classification function ``classify_regime`` and the
``implications`` lookup.  The engine is deterministic and pure-friendly (no bus
required) so it can be unit-tested cheaply; the component subclass
``RegimeEngine`` is the reactor on the event bus, tested later via the full
test suite (regime emits REGIME events consumed by fusion).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from strategy_engine.regime_engine import classify_regime, implications

# A flat, low-vol, no-signal series (ranging).
_FLAT = [0.0] * 10 + [0.001] * 20


def _defaults(**kw):
    base = dict(
        returns=_FLAT,
        atr_pct=0.6,
        vol_pct=0.6,
        trend_strength=0.001,
        adx=2.0,
        macd=0.0,
        zone_flux=0.0,
        macro_shock=0.0,
        of_imbalance=0.0,
        news_intensity=0.0,
    )
    base.update(kw)
    return base


def test_implications_shape():
    for key, spec in {
        "strong_bullish": {"momentum_enabled": 1, "mean_reversion_allowed": 0},
        "disorderly": {"position_size_factor": 0.4},
    }.items():
        got = implications(key)
        assert set(spec.items()) <= set(got.items()), (key, got)


def test_regime_classifications():
    # ranging (flat + mid vol) -> RANGE_BOUND
    key, conf, reasons = classify_regime(**_defaults(vol_pct=1.0, trend_strength=0.001))
    assert key == "range_bound"
    assert 0.0 <= conf <= 1.0
    assert "ranging" in reasons

    # strong trend (ATR in decimal scale, e.g. 0.6%) -> STRONG_BULLISH / STRONG_BEARISH
    key, conf, reasons = classify_regime(**_defaults(vol_pct=0.6, trend_strength=0.015, adx=4.0, atr_pct=0.006))
    assert key in ("strong_bullish", "strong_bearish")
    assert conf >= 0.8

    # high vol -> HIGH_VOL_EXPANSION
    key, conf, _ = classify_regime(**_defaults(vol_pct=3.0, trend_strength=0.0))
    assert key == "high_vol_expansion"
    assert conf <= 1.0

    # macro shock -> MACRO_EVENT_DRIVEN (no low-vol clamp in between)
    key, conf, _ = classify_regime(**_defaults(vol_pct=1.0, macro_shock=0.9))
    assert key == "macro_event_driven"
    assert conf > 0.6

    # liquidity zone flux (no strong trend) -> LIQUIDITY_DRIVEN
    key, conf, _ = classify_regime(**_defaults(vol_pct=1.0, trend_strength=0.003, zone_flux=0.6))
    assert key == "liquidity_driven"
    assert conf <= 1.0

    # strong news -> MACRO_EVENT_DRIVEN
    key, conf, _ = classify_regime(**_defaults(news_intensity=0.8))
    assert key == "macro_event_driven"
    assert conf >= 0.6

    # order-flow imbalance -> DISORDERLY (extreme) / RANGE_BOUND
    key, conf, _ = classify_regime(**_defaults(of_imbalance=0.85))
    assert key == "disorderly"
    assert conf <= 1.0

    # ATR-normalised drift sustained (>=2x ATR) -> keeps strong trend
    key, conf, _ = classify_regime(**_defaults(vol_pct=0.6, trend_strength=0.015, atr_pct=0.006))
    assert key in ("strong_bullish", "strong_bearish") and conf >= 0.8

    # ATR-normalised drift well below 2x range -> downgrade to range
    key, _, _ = classify_regime(**_defaults(vol_pct=0.6, trend_strength=0.005, atr_pct=0.006))
    assert key == "range_bound", key
