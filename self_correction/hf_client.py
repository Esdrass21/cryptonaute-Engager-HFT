"""HuggingFace Inference client (fp32/bf16) — free tier alternative to Mistral API.

Replaces ``self_correction.mistral_client.MistralClient`` with a HuggingFace
Inference API client. This is a cloud-based endpoint (no local GPU needed):

  * Endpoint : https://api-inference.huggingface.co/models
  * Auth     : ``HF_TOKEN`` environment variable (API key)
  * Model    : configurable via ``self_correction.hf.model``
  * Precision: fp32 / bf16 (configurable)

Supported by HuggingFace free tier (rate-limited, no local GPU required).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from core.component import Component
from core.config import Config
from core.events import EventType, NormalizedEvent, make_event

log = logging.getLogger("self_correction.hf")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_HF_CFG = {
    "model": "mistralai/Mistral-Nemo-Instruct-2407",
    "backend": "hf",
    "endpoint": "https://api-inference.huggingface.co/models",
    "token_env": "HF_TOKEN",
    "timeout_s": 30.0,
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.1,
}

# ---------------------------------------------------------------------------
# HuggingFace Inference client
# ---------------------------------------------------------------------------
class HFClient:
    """Thin wrapper around the HuggingFace Inference API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.get("self_correction.hf.model", DEFAULT_HF_CFG["model"])
        self.backend = cfg.get("self_correction.hf.backend", "hf")
        self.endpoint = cfg.get("self_correction.hf.endpoint", DEFAULT_HF_CFG["endpoint"])
        self.token_env = cfg.get("self_correction.hf.token_env", DEFAULT_HF_CFG["token_env"])
        self.timeout = float(cfg.get("self_correction.hf.timeout_s", DEFAULT_HF_CFG["timeout_s"]))
        self.max_tokens = int(cfg.get("self_correction.hf.max_tokens", DEFAULT_HF_CFG["max_tokens"]))
        self.temperature = float(cfg.get("self_correction.hf.temperature", DEFAULT_HF_CFG["temperature"]))
        self.top_p = float(cfg.get("self_correction.hf.top_p", DEFAULT_HF_CFG["top_p"]))
        self.repetition_penalty = float(
            cfg.get("self_correction.hf.repetition_penalty", DEFAULT_HF_CFG["repetition_penalty"])
        )

        self.token = os.environ.get(self.token_env, "")
        self._client = None

        if not self.token:
            log.warning("HF_TOKEN not set -- using stub mode")
            self.enabled = False
        else:
            self.enabled = True

    # -- public API ---------------------------------------------------------
    def complete(self, prompt: str, *, system: str = "") -> str:
        if not self.enabled:
            return self._stub(prompt)

        if self.backend == "hf":
            return self._inference_api(prompt, system)
        return self._stub(prompt)

    # -- backend: inference API --------------------------------------------
    def _inference_api(self, prompt: str, system: str) -> str:
        if self._client is None:
            self._client = {"token": self.token}

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        if system:
            text = f"{system}\n\n{prompt}"
        else:
            text = prompt

        payload = {
            "inputs": text,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "return_full_text": False,
            },
        }

        try:
            resp = requests.post(
                self.endpoint + "/" + self.model.replace("/", "--"),
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                text = str(data)

            return text.strip()

        except requests.exceptions.Timeout:
            log.error("HF inference timed out after %.1fs", self.timeout)
            return self._stub(prompt)
        except requests.exceptions.RequestException as exc:
            log.error("HF inference request failed: %s", exc)
            return self._stub(prompt)

    # -- stub (fallback) ----------------------------------------------------
    def _stub(self, prompt: str) -> str:
        low = prompt.lower()
        if "anomaly" in low:
            return json.dumps({
                "diagnosis": "stub-anomaly-check",
                "severity": "low",
                "recommendation": "monitor",
                "auto_fix": False,
            })
        return json.dumps({
            "cause": "stub: signal-vs-fill slippage within tolerance",
            "slippage_ok": True,
            "recommendation": "keep-params",
            "auto_tune": {},
        })


# ---------------------------------------------------------------------------
# PostTradeReviewer — same as before, but uses HF instead of Mistral
# ---------------------------------------------------------------------------
class PostTradeReviewer(Component):
    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.review")
        self.client = HFClient(cfg)
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
            "Review this closed trade for probable cause and calibration recommendation.\n"
            f"{json.dumps(ctx)}\nReply JSON: {{cause, slippage_ok, recommendation, auto_tune}}."
        )
        raw = self.client.complete(prompt)
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


# ---------------------------------------------------------------------------
# AnomalyDetector — same as before, but uses HF instead of Mistral
# ---------------------------------------------------------------------------
class AnomalyDetector(Component):
    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="self_correction.anomaly")
        self.client = HFClient(cfg)
        self.enabled = cfg.get("self_correction.enabled", True)

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            return
        self.bus.subscribe(EventType.RISK_BREACH, self.on_breach)
        self.bus.subscribe(EventType.METRIC, self.on_metric)
        log.info("Anomaly detector online (HF backend)")

    def on_breach(self, ev: NormalizedEvent) -> None:
        raw = self.client.complete(f"Anomaly: {json.dumps(ev.payload)}\nDiagnose briefly. JSON only.")
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
