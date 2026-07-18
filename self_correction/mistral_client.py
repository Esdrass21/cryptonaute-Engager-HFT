"""Self-correction loop (Layer 3 — §4).

STRICTLY ASYNCHRONOUS, OFF the critical execution path. An LLM call is 200ms–several
seconds — incompatible with real-time decisions. This layer does:

1. Post-trade review (Mistral) — structured JSON diagnosis of a closed trade.
2. Language-based anomaly detection from logs.
3. News scoring delegation (optional).
4. Parameter auto-tuning within pre-fixed bounds (Layer 6 keeps the veto).

All results are *recommendations*. None are auto-applied to live trading. Layer 6 (risk
governance) must approve any parameter change that affects execution.

The Mistral client has a ``stub`` backend (default) that produces a deterministic
structured response with zero network — so the whole platform runs and the post-trade
review path executes end-to-end in tests/paper. Set ``self_correction.mistral.backend:
mistral`` + ``MISTRAL_API_KEY`` to use the real API.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

log = logging.getLogger("self_correction.mistral")


class MistralClient:
    """Thin wrapper around the Mistral API with a deterministic stub backend."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.get("self_correction.mistral.backend", "stub")
        self.model = cfg.get("self_correction.mistral.model", "mistral-large-latest")
        self.timeout = float(cfg.get("self_correction.mistral.timeout_s", 20.0))
        self._client = None

    def complete(self, prompt: str, *, system: str = "") -> str:
        if self.backend == "stub":
            return self._stub(prompt)
        return self._live(prompt, system)

    def _stub(self, prompt: str) -> str:
        # Deterministic structured stub: echoes a plausible diagnosis without network.
        low = prompt.lower()
        if "anomaly" in low:
            return json.dumps({
                "diagnosis": "stub-anomaly-check",
                "severity": "low",
                "recommendation": "monitor",
                "auto_fix": False,
            })
        # post-trade default
        return json.dumps({
            "cause": "stub: signal-vs-fill slippage within tolerance",
            "slippage_ok": True,
            "recommendation": "keep-params",
            "auto_tune": {},
        })

    def _live(self, prompt: str, system: str) -> str:  # pragma: no cover - network
        if self._client is None:
            from mistralai import Mistral

            self._client = Mistral(api_key=self.cfg.secret("MISTRAL_API_KEY", ""))
        resp = self._client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a trading post-mortem analyst. Reply JSON only."},
                {"role": "user", "content": prompt},
            ],
            timeout_ms=int(self.timeout * 1000),
        )
        return resp.choices[0].message.content


class PostTradeReviewer(Component):
    """Consumes ORDER_FILL (close) events, asks Mistral for a structured review."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.review")
        self.client = MistralClient(cfg)
        self.enabled = cfg.get("self_correction.enabled", True)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.info("Self-correction disabled")
            return
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        log.info("Post-trade reviewer online")

    def on_fill(self, ev: NormalizedEvent) -> None:
        if not ev.payload.get("closed", False):
            return  # only review on position close
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
            "Review this closed trade for probable cause and calibration recommendation.\n"
            f"{json.dumps(ctx)}\nReply JSON: {{cause, slippage_ok, recommendation, auto_tune}}."
        )
        raw = self.client.complete(prompt)
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            review = {"raw": raw}
        self.emit(make_event(
            EventType.POST_TRADE_REVIEW, "self_correction.review",
            payload={"context": ctx, "review": review},
            symbol_impact=[ctx["symbol"]] if ctx.get("symbol") else [],
            timestamp=self.clock.now(),
        ))
        self.metrics.inc("post_trade_review_emitted")


class AnomalyDetector(Component):
    """Language-based anomaly detection from log lines / metrics breaches.

    Subscribes to METRIC + RISK_BREACH; produces ANOMALY events with a recommendation.
    Never auto-applies fixes (architecture §4.2 — validation by Layer 6 required).
    """

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.anomaly")
        self.client = MistralClient(cfg)
        self.enabled = cfg.get("self_correction.enabled", True)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            return
        self.bus.subscribe(EventType.RISK_BREACH, self.on_breach)
        self.bus.subscribe(EventType.METRIC, self.on_metric)
        log.info("Anomaly detector online")

    def on_breach(self, ev: NormalizedEvent) -> None:
        raw = self.client.complete(f"Anomaly: {json.dumps(ev.payload)}\nDiagnose briefly. JSON only.")
        try:
            diag = json.loads(raw)
        except json.JSONDecodeError:
            diag = {"raw": raw}
        self.emit(make_event(
            EventType.ANOMALY, "self_correction.anomaly",
            payload={"source_event": ev.to_dict(), "diagnosis": diag, "auto_fix": False},
            symbol_impact=ev.symbol_impact, timestamp=self.clock.now(),
        ))
        self.metrics.inc("anomaly_emitted")

    def on_metric(self, ev: NormalizedEvent) -> None:
        # Simple threshold-based language trigger (no LLM needed for the obvious ones).
        name = ev.payload.get("name", "")
        val = ev.payload.get("value", 0.0)
        if "reject_rate" in name and val > 0.2:
            self.emit(make_event(
                EventType.ANOMALY, "self_correction.anomaly",
                payload={"trigger": name, "value": val, "diagnosis": {"severity": "high"},
                         "auto_fix": False},
                timestamp=self.clock.now(),
            ))
            self.metrics.inc("anomaly_emitted")
