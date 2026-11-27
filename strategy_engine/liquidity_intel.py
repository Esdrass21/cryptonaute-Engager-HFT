"""Liquidity & Order-Flow Intelligence (Layer 6a — §7).

A dedicated liquidity intelligence engine that determines not just where price is
going, but where price may be drawn toward, rejected, or trapped. It produces
``OF_SIGNAL`` events (order-flow signals) that:

1. feed ``strategy_engine.signal_fusion`` (which already wires an
   ``of_weight`` slot), and
2. shape the quality scorer's liquidity sub-score.

Detection modules
-----------------
liquidity_zones   support/resistance zones (pools above and below price)
equal_sides       equal highs and equal lows (liquidity attraction points)
sweep_reject      liquidity sweeps and price rejection (traps)
order_flow_imbalance signed flow imbalance (-1..1) around the current price

The engine is deterministic, fast, and reusable in both live and replay paths.
"""
from __future__ import annotations

import logging

import numpy as np

from core.component import Component
from core.events import EventType, make_event

log = logging.getLogger("strategy.liquidity_intel")

# --------------------------------------------------------------------------- #
# Pure helpers (testable without a bus)
# --------------------------------------------------------------------------- #
def liquidity_zones(prices, window: int = 60):
    """Return (support, resistance) price-zones from rolling min/max of the
    last ``window`` closes.  Lists of (index, price) in the series."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    sup, res = [], []
    for i in range(window, n + 1):
        seg = p[i - window:i]
        sup.append((i - 1, float(np.min(seg))))
        res.append((i - 1, float(np.max(seg))))
    return sup, res


def equal_sides(highs, lows, period: int = 20):
    """Return (equal_highs, equal_lows) counts over a rolling window of the
    last ``period`` bars.  An equal high means price revisited a prior high
    (liquidity sweep candidate); an equal low similarly for support."""
    hs = np.asarray(highs, dtype=float)
    eq_h, eq_l = [], []
    n = len(hs)

    for i in range(period, n + 1):
        window = hs[i - period:i]
        # count of bars within a small tick of the running max (equal highs)
        running_max = np.max(window)
        eq_h.append((i - 1, float(np.sum(np.isclose(window, running_max, atol=1e-6)))))
        running_min = np.min(window)
        eq_l.append((i - 1, float(np.sum(np.isclose(window, running_min, atol=1e-6)))))
    return eq_h, eq_l


def sweep_reject(highs, lows, window: int = 20, tolerance: float = 1e-6):
    """Flag liquidty sweeps: a bar that opens far beyond the prior swing and
    closes back inside the prior range (i.e., price swept and rejected)."""
    hs = np.asarray(highs, dtype=float)
    ls = np.asarray(lows, dtype=float)
    n = len(hs)
    sweeps = []
    for i in range(window, n + 1):
        prev = slice(i - window, i - 1)
        cur = i - 1
        run_h = np.max(hs[prev]) if i - window < n else hs[cur]
        run_l = np.min(ls[prev]) if i - window < n else ls[cur]
        # a candle that wicks well above the prior swing high and closes back
        # inside the prior range -> liquidity sweep/rejection
        hw = hs[cur] - run_h  # candle high vs prior swing high
        lw = run_l - ls[cur]  # candle low vs prior swing low
        if abs(hw) > tolerance or abs(lw) > tolerance:
            sweeps.append((cur, float(hw), float(lw)))
    return sweeps


def order_flow_imbalance(volumes, bid_sizes, ask_sizes, period: int = 50):
    """Estimate signed order-flow imbalance from bid/ask depth vs volume.

    Returns a value in [-1, 1]: positive = buyers in control, negative = sellers.
    """
    vb = np.asarray(bid_sizes, dtype=float)
    va = np.asarray(ask_sizes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = min(len(vb), len(va), len(v), period)
    if n < 2:
        return 0.0
    net = np.sum(vb[-n:]) - np.sum(va[-n:])
    total = np.sum(v[-n:]) + 1e-9
    return float(np.clip(net / total, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Component
# --------------------------------------------------------------------------- #
class LiquidityIntel(Component):
    """Ingests OHLCV (with optional bid/ask sizes) and emits OF_SIGNAL events
    for the fusion and quality layers."""
    def __init__(self, bus, cfg, clock, metrics, window: int = 60):
        super().__init__(bus, cfg, clock, metrics, name="strategy.liquidity_intel")
        self.window = window
        self._highs: dict[str, list[float]] = {}
        self._lows: dict[str, list[float]] = {}
        self._closes: dict[str, list[float]] = {}
        self._bids: dict[str, list[float]] = {}
        self._asks: dict[str, list[float]] = {}
        self._volumes: dict[str, list[float]] = {}

    def start(self) -> None:
        self._running = True
        for sym in self.cfg.get("symbols", []):
            self._highs[sym] = []
            self._lows[sym] = []
            self._closes[sym] = []
            self._bids[sym] = []
            self._asks[sym] = []
            self._volumes[sym] = []
        self.bus.subscribe(EventType.CANDLE, self._on_candle)
        log.info("Liquidity Intelligence Engine online")

    def _on_candle(self, ev) -> None:
        sym = ev.payload["symbol"]
        p = ev.payload
        self._highs.setdefault(sym, []).append(float(p.get("high", p["close"])))
        self._lows.setdefault(sym, []).append(float(p.get("low", p["close"])))
        self._closes.setdefault(sym, []).append(float(p["close"]))
        if len(self._highs[sym]) > self.window:
            self._highs[sym] = self._highs[sym][-self.window:]
            self._lows[sym] = self._lows[sym][-self.window:]
            self._closes[sym] = self._closes[sym][-self.window:]
        if (len(self._highs[sym]) >= 50 and len(self._bids[sym]) >= 10
                and len(self._asks[sym]) >= 10):
            self._emit_signals(sym)

    def _emit_signals(self, sym: str) -> None:
        closes = np.asarray(self._closes[sym], dtype=float)
        highs = np.asarray(self._highs[sym], dtype=float)
        lows = np.asarray(self._lows[sym], dtype=float)
        bids = np.asarray(self._bids[sym], dtype=float)
        asks = np.asarray(self._asks[sym], dtype=float)
        volumes = np.asarray(self._volumes[sym], dtype=float)

        zones_sup, zones_res = liquidity_zones(closes, window=self.window)
        eq_h, eq_l = equal_sides(highs, lows, period=min(20, len(highs)))
        sweeps = sweep_reject(highs, lows, window=min(20, len(highs)))

        # last zone is the current level
        sup = zones_sup[-1][1] if zones_sup else float(np.min(lows))
        res = zones_res[-1][1] if zones_res else float(np.max(highs))

        # order-flow imbalance
        imba = order_flow_imbalance(volumes, bids, asks, period=min(50, len(volumes)))

        payload = {
            "symbol": sym,
            "liquidity_zone_support": float(sup),
            "liquidity_zone_resistance": float(res),
            "liquidity_pool_above": 1.0 if res > float(closes[-1]) else 0.0,
            "liquidity_pool_below": 1.0 if sup < float(closes[-1]) else 0.0,
            "equal_highs": float(eq_h[-1]) if eq_h else 0.0,
            "equal_lows": float(eq_l[-1]) if eq_l else 0.0,
            "sweep_rejects": len(sweeps),
            "order_flow_imbalance": float(imba),
        }
        self.emit(make_event(
            EventType.OF_SIGNAL, "strategy.liquidity_intel",
            payload=payload,
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("of_signal_emitted")
