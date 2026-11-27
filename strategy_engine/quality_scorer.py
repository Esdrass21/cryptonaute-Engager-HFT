"""Trade Quality Scoring (Phase 6 — item 3).

Pre-trade, deterministic quality assessment of every proposed setup, before the
expected-value gate and the risk veto.  Instead of treating a BUY/SELL as a single
unweighted verdict, every setup receives:

score | grade | meaning
------|-------|----------
90-100| A+   | Exceptional setup, eligible for execution
80-89 | A    | High-quality setup
70-79 | B    | Valid, selective execution
60-69 | C    | Watchlist / confirmation required
40-59 | D    | Weak setup
0-39  | F    | Reject

The quality grade is a *presentation* of already-known confluence (that is what
the review called "a re-weighted presentation of signals you already fuse").  It
never overrides the risk engine: a setup scoring A+ can still be vetoed by
risk_management.veto_service on hard-cap, drawdown or correlation.

Sub-scores (each in [-1, 1], higher = better)
--------------------------------------------
* tech_conformity   — how cleanly the tech signal points one way (use the fused
                      tech conviction, |composite| high + confidence high).
* regime_quality    — regime alignment with the proposed direction (trending +
                      up-trend -> high; ranging -> medium; volatile -> watch).
* macro_alignment   — macro surprise magnitude * impact, filtered by sign.
* liquidity         — order-book imbalance (|imbalance| high = deep book).
* news_confirmation — news score weighted by confidence, sign-aligned with the
                      proposed direction.

Reasons are emitted as short human-readable strings so the setup can be audited
("tech strong; news contradict; etc.").
"""

from __future__ import annotations

import logging
import time

from core.component import Component
from core.events import EventType, NormalizedEvent, make_event

log = logging.getLogger("strategy.quality")


class TradeQualityScorer(Component):
    """Per-symbol, pre-fusion trade quality assessment."""

    GRADES = [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (40, "D"), (0, "F")]

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="strategy.quality")
        self._pending: dict[str, dict[str, float]] = {}
        self._t0 = time.monotonic()

    def start(self) -> None:
        self._running = True
        # raw inputs already fused by SignalFusion; recompute quality here so it
        # is decoupled from the fusion weights and can be refreshed on a short
        # cadence without re-fusing from scratch.
        self.bus.subscribe(EventType.TECH_SIGNAL, self._ingest)
        self.bus.subscribe(EventType.NEWS_SCORE, self._ingest)
        self.bus.subscribe(EventType.MACRO, self._ingest)
        self.bus.subscribe(EventType.REGIME, self._ingest)
        self.bus.subscribe(EventType.OF_SIGNAL, self._ingest)
        self.bus.subscribe(EventType.COMPOSITE_SIGNAL, self._ingest)
        log.info("Trade quality scorer online (cadence 500ms)")

    # ---- ingestion ------------------------------------------------------
    def _ingest(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        if not sym:
            return
        kind = ev.type
        val = self._extract(ev)
        bucket = self._pending.setdefault(sym, {"__t0": time.monotonic()})
        bucket[kind] = val

    @staticmethod
    def _extract(ev: NormalizedEvent) -> float:
        if ev.type == EventType.TECH_SIGNAL:
            return float(ev.payload.get("composite", 0.0))
        if ev.type == EventType.NEWS_SCORE:
            return (float(ev.payload.get("score", 0.0))
                    * float(ev.payload.get("confidence", 0.0)))
        if ev.type == EventType.MACRO:
            return (float(ev.payload.get("surprise_index", 0.0))
                    * float(ev.payload.get("impact_weight", 0.0)))
        if ev.type == EventType.REGIME:
            reg = ev.payload.get("regime")
            if reg == "trending":
                return 0.5 * float(ev.payload.get("confidence", 0.0))
            if reg == "volatile":
                return -0.3 * float(ev.payload.get("confidence", 0.0))
            return 0.0
        if ev.type == EventType.OF_SIGNAL:
            return float(ev.payload.get("order_book_imbalance", 0.0))
        if ev.type == EventType.COMPOSITE_SIGNAL:
            return float(ev.payload.get("composite", 0.0))
        return 0.0

    # ---- scoring ----------------------------------------------------------
    def tick(self) -> None:
        # called by the orchestration loop every ~500ms (mirrors SignalFusion).
        now = time.monotonic()
        for sym, bucket in list(self._pending.items()):
            if now - bucket.get("__t0", now) > 0.5:
                self._grade(sym, bucket)
                del self._pending[sym]

    def _grade(self, sym: str, bucket: dict[str, float]) -> None:
        tech = bucket.get("tech_signal", 0.0)
        news = bucket.get("news_score", 0.0)
        macro = bucket.get("macro", 0.0)
        regime = bucket.get("regime", 0.0)
        of = bucket.get("of_signal", 0.0)
        composite = bucket.get("composite_signal", 0.0)
        conf = min(1.0, abs(composite) + 0.05)

        tech_conformity = max(-1.0, min(1.0, tech))
        regime_quality = self._regime_quality(regime, composite > 0)
        macro_alignment = self._sign_align(macro, composite > 0)
        liquidity = max(-1.0, min(1.0, of))
        news_confirmation = self._sign_align(news, composite > 0)

        q = (0.35 * tech_conformity
             + 0.20 * regime_quality
             + 0.15 * macro_alignment
             + 0.15 * liquidity
             + 0.15 * news_confirmation)
        q = max(-1.0, min(1.0, q))
        grade = self._to_grade(q * conf)

        reasons = self._reasons(tech_conformity, regime_quality, macro_alignment,
                                liquidity, news_confirmation)
        self.emit(make_event(
            EventType.QUALITY_SCORE, "strategy.quality",
            payload={
                "symbol": sym,
                "quality_score": round(q, 4),
                "grade": grade,
                "reasons": reasons,
                "inputs": {
                    "tech": round(tech_conformity, 3),
                    "regime": round(regime_quality, 3),
                    "macro": round(macro_alignment, 3),
                    "liquidity": round(liquidity, 3),
                    "news": round(news_confirmation, 3),
                },
            },
        ))

    @staticmethod
    def _regime_quality(regime: float, direction: bool) -> float:
        # regime is already in [-1,1]; reward alignment with the proposed direction.
        if regime > 0 and direction:
            return 0.6 * regime + 0.4
        if regime < 0 and not direction:
            return 0.6 * abs(regime) + 0.4
        return 0.2 * (1.0 - abs(regime))  # range / mismatch -> lower

    @staticmethod
    def _sign_align(val: float, direction: bool) -> float:
        if not direction:
            val = -val
        # map [-1,1] -> [0,1] confidence alignment; 0 = neutral
        return 0.5 * val + 0.5

    @staticmethod
    def _to_grade(q: float) -> str:
        for thr, grade in TradeQualityScorer.GRADES:
            if q >= thr:
                return grade
        return "F"

    @staticmethod
    def _reasons(*subs: float) -> list[str]:
        out: list[str] = []
        tech, regime, macro, liq, news = subs
        if tech > 0.6:
            out.append("tech_conformante")
        elif tech < -0.2:
            out.append("technique faible")
        if regime > 0.5:
            out.append("régime favorable")
        elif regime < 0.3:
            out.append("régime non-confirmant")
        if macro > 0.5:
            out.append("macro alignée")
        elif macro < 0.3:
            out.append("macro non-alignée")
        if liq > 0.6:
            out.append("liquidité confirmée")
        if news > 0.55 and news < 0.9:
            out.append("news en litige")
        return out
