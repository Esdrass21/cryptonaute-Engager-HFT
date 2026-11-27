"""Strategy / Decision engine (Layer 4 — §5).

Event-driven. Per-symbol finite-state machine drives the lifecycle:

    idle -> signal_detected -> confirmation -> position_open -> managing -> closed

Signal fusion combines technical + macro + news + regime (+ validated order-flow) into a
single composite score. The Expected-Value engine estimates reward/drawdown/win-prob/
duration/slippage *before* any order is sent; an order proposal is emitted only if EV
exceeds a calibrated threshold.

Architecture's hard rule: this layer is *deterministic* and decides. The LLM (Layer 3) only
recommends and is never in the critical path.
"""

from __future__ import annotations

import logging

from core.component import Component
from core.events import EventType, NormalizedEvent, make_event

log = logging.getLogger("strategy.fusion")


# --------------------------------------------------------------------------- #
# Signal fusion
# --------------------------------------------------------------------------- #
class SignalFusion(Component):
    """Combines incoming signals per symbol into a composite score in [-1, 1]."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="strategy.fusion")
        self.weights = {
            "tech": cfg.get("strategy.composite.tech_weight", 0.45),
            "news": cfg.get("strategy.composite.news_weight", 0.25),
            "macro": cfg.get("strategy.composite.macro_weight", 0.15),
            "regime": cfg.get("strategy.composite.regime_weight", 0.10),
            "order_flow": cfg.get("strategy.composite.of_weight", 0.05),
        }
        self.threshold = cfg.get("strategy.composite.threshold", 0.55)
        self._pending: dict[str, dict[str, float]] = {}

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TECH_SIGNAL, lambda e: self._ingest(e, "tech"))
        self.bus.subscribe(EventType.NEWS_SCORE, self._ingest_news)
        self.bus.subscribe(EventType.MACRO, lambda e: self._ingest(e, "macro"))
        self.bus.subscribe(EventType.REGIME, lambda e: self._ingest(e, "regime"))
        self.bus.subscribe(EventType.OF_SIGNAL, lambda e: self._ingest(e, "order_flow"))
        log.info("Signal fusion online (weights=%s threshold=%.2f)", self.weights, self.threshold)

    def _ingest_news(self, ev: NormalizedEvent) -> None:
        for sym in ev.symbol_impact:
            self._add(sym, "news", ev.payload.get("score", 0.0) * ev.payload.get("confidence", 0.0))

    def _ingest(self, ev: NormalizedEvent, kind: str) -> None:
        sym = ev.payload.get("symbol")
        if not sym:
            return
        self._add(sym, kind, self._extract(ev, kind))

    @staticmethod
    def _extract(ev: NormalizedEvent, kind: str) -> float:
        if kind == "tech":
            # The quant layer already emits a signed conviction in [-1, 1]; use it directly.
            return float(ev.payload.get("composite", 0.0))
        if kind == "regime":
            reg = ev.payload.get("regime")
            if reg == "trending":
                return 0.5 * ev.payload.get("confidence", 0.0)
            if reg == "volatile":
                return -0.3 * ev.payload.get("confidence", 0.0)
            return 0.0
        if kind == "macro":
            return float(ev.payload.get("surprise_index", 0.0)) * float(ev.payload.get("impact_weight", 0.0))
        if kind == "order_flow":
            return float(ev.payload.get("order_book_imbalance", 0.0))
        return 0.0

    def _add(self, sym: str, kind: str, val: float) -> None:
        bucket = self._pending.setdefault(sym, {})
        bucket[kind] = val
        if "tech" in bucket:  # technical is the primary driver
            self._combine(sym, bucket)

    def _combine(self, sym: str, bucket: dict[str, float]) -> None:
        # Weighted sum, normalized by the total weight actually present so a strong
        # single driver (e.g. a decisive tech signal) can still cross the threshold.
        total_w = sum(self.weights.get(k, 0.0) for k in bucket) or 1.0
        raw = sum(self.weights.get(k, 0.0) * v for k, v in bucket.items())
        score = max(-1.0, min(1.0, raw / total_w))
        conf = min(1.0, abs(score) + 0.05)
        self.emit(make_event(
            EventType.COMPOSITE_SIGNAL, "strategy.fusion",
            payload={
                "symbol": sym,
                "composite": float(score),
                "confidence": float(conf),
                "components": dict(bucket),
                "above_threshold": bool(abs(score) >= self.threshold),
            },
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("composite_signal_emitted")
        bucket.clear()  # decay stale signals
