"""NLP news pipeline (Layer 2 — §3.3).

Steps (per architecture): collect -> relevance filter -> sentiment scoring
(FinBERT local / lexicon fallback / Mistral) -> fusion into Layer 4 as a feature only.

Backends selected by ``cfg.nlp.backend``:
  * ``lexicon``  — deterministic keyword scorer (default; zero deps, always available).
  * ``finbert``  — local FinBERT (ProsusAI/finbert) via transformers + torch.
  * ``mistral``  — richer scoring delegated to the Mistral API (Layer 3), async.

Architecture: news is *never* a standalone decision — only a feature fed to fusion (§3.3).
"""

from __future__ import annotations

import logging
import re

from core.component import Component
from core.events import EventType, NormalizedEvent
from data_ingestion.normalizer import normalize_news_score

log = logging.getLogger("analytics.nlp")

_BULLISH = {"rally": 0.6, "rallies": 0.6, "surge": 0.7, "gains": 0.5, "rises": 0.5, "breakout": 0.6,
            "bid": 0.4, "lifts": 0.5, "improves": 0.5, "momentum": 0.3, "hawkish": 0.2,
            "safe-haven": 0.4, "breaks resistance": 0.6}
_BEARISH = {"selloff": -0.7, "slump": -0.6, "falls": -0.5, "drops": -0.5, "retreats": -0.4,
            "pressure": -0.4, "dovish": -0.3, "compresses": -0.1, "spike": -0.2, "uncertainty": -0.3}
_SYMBOL_HINTS = {"EURUSD": ["eur", "euro", "eurusd"], "XAUUSD": ["gold", "xau", "safe-haven", "bullion"],
                 "BTCUSDT": ["btc", "bitcoin", "crypto"]}


class NewsScorer(Component):
    """Scores news relevance + sentiment and rebroadcasts NEWS_SCORE events."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="nlp.scorer")
        self.backend = cfg.get("nlp.backend", "lexicon")
        self.symbols = cfg.get("symbols", ["EURUSD", "XAUUSD", "BTCUSDT"])
        self._model = None
        if self.backend == "finbert":
            self._load_finbert()

    def _load_finbert(self):
        try:
            from transformers import pipeline

            self._model = pipeline("sentiment-analysis",
                                   model=self.cfg.get("nlp.finbert_model", "ProsusAI/finbert"))
            log.info("FinBERT loaded (%s)", self.cfg.get("nlp.finbert_model"))
        except Exception as exc:
            log.warning("FinBERT unavailable, falling back to lexicon: %s", exc)
            self.backend = "lexicon"

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.NEWS, self.on_news)
        log.info("NLP scorer online (backend=%s)", self.backend)

    def on_news(self, ev: NormalizedEvent) -> None:
        text = f"{ev.payload.get('title','')} {ev.payload.get('summary','')}"
        impact = self._relevance(text)
        if not impact:
            return
        score, conf, expl = self._sentiment(text)
        self.emit(normalize_news_score(
            f"nlp.{self.backend}", headline=ev.payload.get("title", ""),
            symbol_impact=impact, score=score, confidence=conf, explanation=expl, time=self.clock.now()))
        self.metrics.inc("nlp_news_scored")

    def _relevance(self, text: str) -> list[str]:
        low = text.lower()
        return [sym for sym in self.symbols if any(h in low for h in _SYMBOL_HINTS.get(sym, []))]

    def _sentiment(self, text: str):
        if self.backend == "finbert" and self._model is not None:
            return self._sentiment_finbert(text)
        if self.backend == "mistral":
            return 0.0, 0.0, "deferred-to-mistral"
        return self._sentiment_lexicon(text)

    def _sentiment_lexicon(self, text: str):
        low = text.lower()
        score = 0.0
        matched = []
        for term, w in _BULLISH.items():
            if re.search(r"\b" + re.escape(term) + r"\b", low):
                score += w
                matched.append(term)
        for term, w in _BEARISH.items():
            if re.search(r"\b" + re.escape(term) + r"\b", low):
                score += w
                matched.append(term)
        score = max(-1.0, min(1.0, score))
        conf = min(1.0, abs(score) + 0.1)
        return score, conf, "lexicon:" + ",".join(matched[:4])

    def _sentiment_finbert(self, text: str):
        out = self._model(text[:512])[0]
        label = out["label"].lower()
        s = out["score"]
        score = s if label == "positive" else (-s if label == "negative" else 0.0)
        return score, float(s), f"finbert:{label}"
