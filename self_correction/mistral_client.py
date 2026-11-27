"""Self-correction loop (Layer 3 — §4).

STRICTLY ASYNCHRONOUS, OFF the critical execution path. An LLM call is 200ms–several
seconds — incompatible with real-time decisions. This layer does:

1. Post-trade review (HuggingFace) — structured JSON diagnosis of a closed trade.
2. Language-based anomaly detection from logs.
3. News scoring delegation (optional).
4. Parameter auto-tuning within pre-fixed bounds (Layer 6 keeps the veto).

All results are *recommendations*. None are auto-applied to live trading. Layer 6 (risk
governance) must approve any parameter change that affects execution.

The HuggingFace client has a ``stub`` backend (default) that produces a deterministic
structured response with zero network — so the whole platform runs and the post-trade
review path executes end-to-end in tests/paper. Set ``self_correction.hf.token_env`` to
``HF_TOKEN`` (free tier API key) to use the real inference API.
"""

from __future__ import annotations

import json
import logging

from core.component import Component
from core.config import Config
from core.events import EventType, NormalizedEvent, make_event
from .hf_client import HFClient as _HFClient

log = logging.getLogger("self_correction.self")


class Provider(Component):
    """Unified interface to the HuggingFace Inference API (fp32/bf16).

    The underlying HFClient is a pure Python HTTP client with a stub fallback:
    no local GPU, no heavy deps. Free tier: hf.co/conversation.
    """

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.provider")
        self.client = _HFClient(cfg)
        self.enabled = cfg.get("self_correction.enabled", True)
        log.info("HF client enabled=%s backend=%s model=%s",
                 self.enabled, self.client.backend, self.client.model)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.info("Self-correction disabled")
            return

    def complete(self, prompt: str, *, system: str = "") -> str:
        return self.client.complete(prompt, system=system)


class PostTradeReviewer(Component):
    """Consumes ORDER_FILL (close) events, asks HF for a structured review."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.review")
        self.provider = Provider(bus, cfg, clock, metrics)
        self.enabled = cfg.get("self_correction.enabled", True)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.info("Self-correction disabled")
            return
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        log.info("Post-trade reviewer online (HF backend)")

    def on_fill(self, ev: NormalizedEvent) -> None:
        if not ev.payload.get("closed", False):
            return
        ctx = {
            "symbol": ev.payload.get("symbol"),
            "side": ev.payload.get("side"),
            "entry": ev.payload.get("price_entry"),
            "exit": ev.payload.get("price"),
            "pnl": ev.payload.get("pnl"),
            "expected_pnl": ev.payload.get("expected_pnl"),
            "slippage": ev.payload.get("slippage"),
        }
        prompt = (
            "Review this closed trade for probable cause and calibration recommendation. "
            f"{json.dumps(ctx)} "
            "Reply JSON: {cause, slippage_ok, recommendation, auto_tune}."
        )
        raw = self.provider.complete(prompt)
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            review = {"raw": raw}
        self.emit(
            make_event(
                EventType.POST_TRADE_REVIEW,
                "self_correction.review",
                payload={"context": ctx, "review": review},
                symbol_impact=[ctx["symbol"]] if ctx.get("symbol") else [],
                timestamp=self.clock.now(),
            )
        )
        self.metrics.inc("post_trade_review_emitted")


class AnomalyDetector(Component):
    """Language-based anomaly detection from log lines / metrics breaches.

    Subscribes to METRIC + RISK_BREACH; produces ANOMALY events with a recommendation.
    Never auto-applies fixes (architecture §4.2 — validation by Layer 6 required).
    """

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.anomaly")
        self.provider = Provider(bus, cfg, clock, metrics)
        self.enabled = cfg.get("self_correction.enabled", True)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            return
        self.bus.subscribe(EventType.RISK_BREACH, self.on_breach)
        self.bus.subscribe(EventType.METRIC, self.on_metric)
        log.info("Anomaly detector online (HF backend)")

    def on_breach(self, ev: NormalizedEvent) -> None:
        raw = self.provider.complete(
            "Anomaly: " + json.dumps(ev.payload) + "\nDiagnose briefly. JSON only."
        )
        try:
            diag = json.loads(raw)
        except json.JSONDecodeError:
            diag = {"raw": raw}
        self.emit(
            make_event(
                EventType.ANOMALY,
                "self_correction.anomaly",
                payload={"source_event": ev.to_dict(), "diagnosis": diag, "auto_fix": False},
                symbol_impact=ev.symbol_impact,
                timestamp=self.clock.now(),
            )
        )
        self.metrics.inc("anomaly_emitted")

    def on_metric(self, ev: NormalizedEvent) -> None:
        name = ev.payload.get("name", "")
        val = ev.payload.get("value", 0.0)
        if "reject_rate" in name and val > 0.2:
            self.emit(
                make_event(
                    EventType.ANOMALY,
                    "self_correction.anomaly",
                    payload={
                        "trigger": name,
                        "value": val,
                        "diagnosis": {"severity": "high"},
                        "auto_fix": False,
                    },
                    timestamp=self.clock.now(),
                )
            )
            self.metrics.inc("anomaly_emitted")
