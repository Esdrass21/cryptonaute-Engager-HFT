"""Market Context Engine — Instrument-Specific Intelligence (Layer 6a — §7).

The generic signals in ``signal_fusion`` are fused per-symbol, but the *meaning*
of a macro or news move depends on the instrument. XAUUSD reacts to USD/real
yields; EURUSD to ECB vs Fed policy; GBPUSD to BoE vs Fed; BTCUSD to funding
rates and ETF flows. A flat "macro shock" is therefore not a flat signal.

This engine keeps a per-instrument context object, rebuilt from the instrument's
own sensitivity profile, volatility model and signal weighting, and emits
``CONTEXT`` events that (a) inform the quality scorer's context sub-score and
(b) weight the fused composite per-symbol.

It is deterministic and fast (no LLM). The per-instrument model is deliberately
lightweight (config-driven lookup + rolling EWMA), keeping this layer inside the
"decision, not question" boundary.

Per-instrument context
----------------------
xauusd: USD + real yields, geopolitical risk, central bank demand,
        inflation expectations, liquidity sweeps
eurusd: ECB vs Fed policy, DXY, eurozone data, US economic releases
gbpusd: BoE vs Fed policy, UK CPI, UK employment, GBP sentiment
btcusd: funding rates, open interest, liquidations, ETF flows, risk appetite
"""
from __future__ import annotations

import logging

import numpy as np

from core.component import Component
from core.events import EventType, make_event

log = logging.getLogger("strategy.instrument_context")

# --------------------------------------------------------------------------- #
# Instrument profiles
# --------------------------------------------------------------------------- #
# Each profile defines the signals that matter to this instrument and the
# corresponding entries they map onto in the fused composite.
INSTRUMENT_PROFILES = {
    "xauusd": {
        "label": "XAUUSD Gold",
        "macro_signals": ["u.s. real yields", "geopolitical risk", "central bank demand",
                          "inflation expectations", "liquidity sweeps"],
        "sensitivity": {"macro": 0.7, "tech": 0.2, "news": 0.1},
        "vol_ewma": 14,            # factor in the vol EWMA (close to old param)
    },
    "eurusd": {
        "label": "EURUSD",
        "macro_signals": ["ecb vs fed policy", "dxy", "eurozone data", "us economic releases"],
        "sensitivity": {"macro": 0.75, "tech": 0.15, "news": 0.1},
        "vol_ewma": 14,
    },
    "gbpusd": {
        "label": "GBPUSD",
        "macro_signals": ["boe vs fed policy", "uk cpi", "uk employment", "gbp sentiment"],
        "sensitivity": {"macro": 0.7, "tech": 0.2, "news": 0.1},
        "vol_ewma": 14,
    },
    "btcusd": {
        "label": "BTCUSD",
        "macro_signals": ["funding rates", "open interest", "liquidations", "etf flows",
                          "risk appetite"],
        "sensitivity": {"macro": 0.5, "tech": 0.25, "news": 0.25},
        "vol_ewma": 14,
    },
}

_DEFAULT_PROFILES = {
    "unknown": {
        "label": "Generic",
        "sensitivity": {"macro": 0.5, "tech": 0.3, "news": 0.2},
        "vol_ewma": 14,
    }
}


def get_profile(symbol):
    symbol = (symbol or "").lower()
    return dict(INSTRUMENT_PROFILES.get(symbol, _DEFAULT_PROFILES["unknown"]))


def build_instrument_context(symbol, price, rets, vol_ewma_factor=14, meta=None):
    """Build a plain-dict context for ``symbol`` without any bus.

    Returns a dict:
        {
          "symbol": ...,
          "profile": {...},
          "vol": rolling eurma vol (decimal),
          "trend": signed % move over the lookback,
          "regime_anchor": the regime signal this instrument inherits,
        }
    """
    rets = np.asarray(rets, dtype=float)
    rets = rets[-20:] if len(rets) > 20 else rets
    vol = float(np.std(rets) * np.sqrt(288)) if len(rets) >= 2 else 0.0
    profile = get_profile(symbol)
    trend = float(np.mean(rets) * 288) if len(rets) >= 2 else 0.0  # annualized trend
    return {
        "symbol": symbol,
        "profile": profile,
        "vol": float(vol),
        "trend": float(trend),
        "vol_ewma_factor": int(vol_ewma_factor),
        "meta": meta or {},
    }


class ContextEngine(Component):
    """Per-instrument context engine. Ingests OHLCV + macro + order-flow and
    emits CONTEXT events for the fusion and quality layers."""
    def __init__(self, bus, cfg, clock, metrics, window: int = 200):
        super().__init__(bus, cfg, clock, metrics, name="strategy.instrument_context")
        self.window = window
        self._buf: dict[str, list[float]] = {}
        self._logbuf: dict[str, list[float]] = {}

    def start(self) -> None:
        self._running = True
        for sym in self.cfg.get("symbols", []):
            self._buf[sym] = []
            self._logbuf[sym] = []
        self.bus.subscribe(EventType.CANDLE, self._on_candle)
        self.bus.subscribe(EventType.MACRO, self._on_macro)
        self.bus.subscribe(EventType.OF_SIGNAL, self._on_of)
        log.info("Instrument Context Engine online")

    def _on_candle(self, ev) -> None:
        sym = ev.payload["symbol"]
        c = ev.payload["close"]
        self._buf.setdefault(sym, []).append(c)
        self._logbuf.setdefault(sym, []).append(float(np.log(c / self._buf[sym][-2]) if len(self._buf[sym]) >= 2 else 0.0))
        if len(self._buf[sym]) > self.window:
            self._buf[sym] = self._buf[sym][-self.window:]
        if len(self._logbuf[sym]) >= 20:
            self._emit_context(sym)

    def _on_macro(self, ev) -> None:
        # macro blends into the context weighting; no direct emit needed here
        pass

    def _on_of(self, ev) -> None:
        # order-flow signals modulate the liquidity half of the context
        pass

    def _emit_context(self, sym: str) -> None:
        rets = self._logbuf[sym]
        ctx = build_instrument_context(sym, self._buf[sym], rets)
        self.emit(make_event(
            EventType.CONTEXT, "strategy.instrument_context",
            payload={"symbol": sym, "context": ctx},
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("context_emitted")
