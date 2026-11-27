"""Adaptive Market Regime Intelligence Engine (Layer 5 — §7).

Extends the lightweight ``analytics.quant.regime`` *feature* into a full adaptive
regime intelligence engine:

* Multi-regime classification (strong trend / range / high/low vol expansion,
  liquidity-driven, macro-driven, risk-on/off, disorderly).
* Confidence + regime strength.
* Trading implications used to gate strategy behaviour (momentum, mean-reversion,
  position sizing, news confirmation) — per the architecture's hard rule that the
  risk engine (L6) plus this intelligence layer, never the LLM, decide.

The engine stays deterministic and fast (no LLM). It emits ``EventType.REGIME``
events consumed by ``strategy_engine.signal_fusion`` (which already wires a
``regime`` weight slot) and by the (not-yet-implemented) validation/report layer.

Regime classes
--------------
STRONG_BULLISH, STRONG_BEARISH                   trend + high trend strength
RANGE_BOUND                                      flat + low vol
HIGH_VOL_EXPANSION, LOW_VOL_COMPRESSION          vol percentile spike/crush
LIQUIDITY_DRIVEN                               retracting liquidity
MACRO_EVENT_DRIVEN                             strong macro/news event present
RISK_ON, RISK_OFF                                cross-asset sentiment
DISORDERLY                                       unstable, high churn risk

Trading implications (per-regime)
---------------------------------
momentum      "momentum_enabled"   1 / 0
mean_reversion "mean_reversion_allowed" 1 / 0
position_size  "position_size_factor" 0.5-1.0
news_conf     "news_confirmation_required" 1 / 0
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("strategy.regime_engine")

# --------------------------------------------------------------------------- #
# Regime taxonomy
# --------------------------------------------------------------------------- #
# (key, label, trading implications)
REGIMES = {
    "strong_bullish": {
        "label": "Strong Bullish",
        "implications": {
            "momentum_enabled": 1,
            "mean_reversion_allowed": 0,
            "position_size_factor": 1.0,
            "news_confirmation_required": 0,
        },
    },
    "strong_bearish": {
        "label": "Strong Bearish",
        "implications": {
            "momentum_enabled": 1,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.8,
            "news_confirmation_required": 1,
        },
    },
    "range_bound": {
        "label": "Range Bound",
        "implications": {
            "momentum_enabled": 0,
            "mean_reversion_allowed": 1,
            "position_size_factor": 0.9,
            "news_confirmation_required": 1,
        },
    },
    "high_vol_expansion": {
        "label": "High Volatility Expansion",
        "implications": {
            "momentum_enabled": 1,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.6,
            "news_confirmation_required": 1,
        },
    },
    "low_vol_compression": {
        "label": "Low Volatility Compression",
        "implications": {
            "momentum_enabled": 1,
            "mean_reversion_allowed": 1,
            "position_size_factor": 0.9,
            "news_confirmation_required": 1,
        },
    },
    "liquidity_driven": {
        "label": "Liquidity Driven",
        "implications": {
            "momentum_enabled": 0,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.5,
            "news_confirmation_required": 1,
        },
    },
    "macro_event_driven": {
        "label": "Macro-Event-Driven",
        "implications": {
            "momentum_enabled": 0,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.7,
            "news_confirmation_required": 1,
        },
    },
    "risk_on": {
        "label": "Risk-On",
        "implications": {
            "momentum_enabled": 1,
            "mean_reversion_allowed": 0,
            "position_size_factor": 1.0,
            "news_confirmation_required": 0,
        },
    },
    "risk_off": {
        "label": "Risk-Off",
        "implications": {
            "momentum_enabled": 0,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.7,
            "news_confirmation_required": 1,
        },
    },
    "disorderly": {
        "label": "Disorderly",
        "implications": {
            "momentum_enabled": 0,
            "mean_reversion_allowed": 0,
            "position_size_factor": 0.4,
            "news_confirmation_required": 1,
        },
    },
}


def classify_regime(returns, atr_pct, vol_pct, trend_strength, adx, macd, zone_flux, macro_shock, of_imbalance, news_intensity):
    """Return (key, confidence, reasons) for the current regime.

    ``atr_pct``      Average True Range as % of price (fast, 14) — drift/range proxy
    ``returns``      rolling log-return array (last up to 30)
    ``vol_pct``      annualized vol percentile (0-100)
    ``trend_strength`` % price move over the lookback (signed)
    ``adx``          +DI/-DI spread (0-10) trend strength
    ``macd``         MACD histogram (signed)
    ``zone_flux``    liquidity-zone flux (signed, liquidity_driven proxy)
    ``macro_shock``  macro-news shock (0-1)
    ``of_imbalance`` order-flow imbalance (-1..1)
    ``atr_pct``      Average True Range as % of price (fast, 14) — drift/range proxy
    ``returns``      rolling log-return array (last up to 30)
    ``vol_pct``      annualized vol percentile (0-100)
    ``trend_strength`` % price move over the lookback (signed)
    ``adx``          +DI/-DI spread (0-10) trend strength
    ``macd``         MACD histogram (signed)
    ``zone_flux``    liquidity-zone flux (signed, liquidity_driven proxy)
    ``macro_shock``  macro-news shock (0-1)
    ``of_imbalance`` order-flow imbalance (-1..1)
    ``news_intensity`` news intensity (0-1)
    """
    rets = np.asarray(returns, dtype=float)[-30:]
    vol_pct = float(vol_pct) if vol_pct else float(np.std(rets) * np.sqrt(288))

    # ---- trend strength proxy: ATR (drift/range), ADX, MACD -------------- #
    # ATR gives a raw drift magnitude (normalized below); ADX and MACD refine
    # the trend-strength read.  The three agree ~> conservative call.
    atr = float(atr_pct) if atr_pct is not None else 0.0
    trend = float(trend_strength)  # signed % move over the lookback

    # ---- trend strength proxy: ADX and MACD ----------------------------- #
    adx = float(adx) if adx is not None else 0.0
    macd = float(macd) if macd is not None else 0.0

    # ---- regime selection (deterministic rule table) --------------------- #
    # Dominant signals are checked first; ranging / low-vol are the fallback.
    macro_shock = float(macro_shock) if macro_shock is not None else 0.0
    of_imbalance = float(of_imbalance) if of_imbalance is not None else 0.0
    zone_flux = float(zone_flux) if zone_flux is not None else 0.0
    news_intensity = float(news_intensity) if news_intensity is not None else 0.0
    high_vol = vol_pct >= 2.5
    low_vol = vol_pct <= 0.8
    extreme_trend = abs(trend) >= 0.012
    strong_trend = abs(trend) >= 0.004 and adx >= 3.0
    ranging = abs(trend) < 0.002 and vol_pct < 1.2

    if macro_shock >= 0.7:
        return "macro_event_driven", min(1.0, 0.6 + macro_shock * 0.4), ["macro shock"]
    if high_vol:
        return "high_vol_expansion", min(1.0, vol_pct / 3.0), ["high vol"]
    if extreme_trend:
        # ATR-normalised drift (signal-to-noise): drift sustained at >=2x the
        # 14-period ATR is real; drift that barely exceeds a violent range is
        # likely noise and is downgraded to range_bound.
        trend_quality = abs(trend) / max(atr, 1e-9)
        if trend_quality >= 2.0:
            return ("strong_bullish" if trend > 0 else "strong_bearish"), min(1.0, abs(trend) * 100), [
                "atr-normalized drift >= 2x ATR"
            ]
        return "range_bound", 0.6, ["ATR-normalised drift < 2x range"]
    if strong_trend:
        return ("strong_bullish" if trend > 0 else "strong_bearish"), min(1.0, adx / 6.0), [
            "adx>=3 trend" if adx >= 3.0 else "trend"
        ]
    if news_intensity >= 0.6:
        return "macro_event_driven", min(1.0, news_intensity), ["news intensity"]
    if abs(of_imbalance) >= 0.6:
        return "disorderly" if abs(of_imbalance) >= 0.8 else "range_bound", min(1.0, abs(of_imbalance) / 0.8), [
            "order-flow imbalance" if abs(of_imbalance) >= 0.6 else "balanced flow"
        ]
    if abs(zone_flux) >= 0.5 and abs(of_imbalance) < 0.3:
        return "liquidity_driven", min(1.0, abs(zone_flux) / 1.0), ["liquidity zone flux"]
    if low_vol and abs(trend) < 0.003:
        return "low_vol_compression", min(1.0, 1.0 - vol_pct / 1.0), ["low vol + flat"]
    if ranging:
        return "range_bound", min(1.0, 0.6 + (1.0 - vol_pct) * 0.4), ["ranging"]
    # default: ranging
    return "range_bound", 0.6, ["default ranging"]


def implications(regime_key):
    return dict(REGIMES[regime_key]["implications"])
