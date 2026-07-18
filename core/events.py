"""Layer-agnostic event contracts.

Per architecture §2.3 every ingested item is normalized into a single JSON schema
before it enters the stream: ``(timestamp, source, type, payload, symbol_impact[])``.

We encode this as a typed, immutable-ish dataclass so downstream layers can rely on
a stable shape without re-parsing free-form JSON. ``to_dict()`` / ``from_dict()``
round-trip for wire transport (Redis streams, replay harness, logging).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

# Canonical event types flowing through the bus, grouped by layer.
class EventType(str, Enum):
    # Layer 1 — market
    TICK = "tick"                       # broker/aggregator raw tick
    CANDLE = "candle"                   # OHLCV bar (closed)
    ORDER_BOOK = "order_book"           # L2 snapshot / delta
    # Layer 1 — news / macro
    NEWS = "news"                       # scored news item
    MACRO = "macro"                     # macro release / calendar event
    # Layer 2 — analytics outputs
    TECH_SIGNAL = "tech_signal"         # indicator-derived signal
    REGIME = "regime"                   # market-regime classification
    OF_SIGNAL = "of_signal"             # order-flow hypothesis signal (validated)
    NEWS_SCORE = "news_score"           # FinBERT/Mistral sentiment score
    # Layer 3 — self-correction
    POST_TRADE_REVIEW = "post_trade_review"
    ANOMALY = "anomaly"
    # Layer 4 — decision
    COMPOSITE_SIGNAL = "composite_signal"
    ORDER_PROPOSAL = "order_proposal"   # proposed trade from strategy engine
    # Layer 5 — execution
    ORDER_SENT = "order_sent"
    ORDER_FILL = "order_fill"
    ORDER_REJECT = "order_reject"
    ORDER_CANCEL = "order_cancel"
    # Layer 6 — risk
    RISK_VETO = "risk_veto"             # signed allow/deny
    RISK_BREACH = "risk_breach"         # circuit-breaker trip
    # Lifecycle / control
    CONTROL = "control"                 # start/stop/pause per symbol
    METRIC = "metric"                   # observability sample

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


@dataclass
class NormalizedEvent:
    """The single schema every layer emits/consumes.

    Attributes
    ----------
    event_id : str
        Globally unique id (uuid4) — used for replay dedup and audit.
    timestamp : float
        Unix epoch seconds (UTC). ``event_time`` is when the underlying thing
        happened; ``ingested_at`` is when we normalized it.
    source : str
        Origin identifier, e.g. ``mt5``, ``binance``, ``rss.forexlive``,
        ``strategy.ma_cross``, ``risk.veto``.
    type : str
        One of :class:`EventType` values.
    payload : Dict[str, Any]
        Structured body. Callers should treat unknown keys as forward-compatible.
    symbol_impact : List[str]
        Symbols this event may affect (e.g. ``["EURUSD", "XAUUSD"]``).
    ingested_at : float
        Set by the producer at normalization time.
    sequence : int
        Per-source monotonic sequence (helps detect gaps / desync).
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    type: str = EventType.TICK.value
    payload: Dict[str, Any] = field(default_factory=dict)
    symbol_impact: List[str] = field(default_factory=list)
    ingested_at: float = field(default_factory=time.time)
    sequence: int = 0

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedEvent":
        # Tolerate legacy keys gracefully.
        d = dict(d)
        d.pop("ingested_at", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ---- helpers -------------------------------------------------------
    def with_timestamp(self, ts: float) -> "NormalizedEvent":
        """Return a copy stamped at ``ts`` (replay harness uses this)."""
        return NormalizedEvent(
            event_id=self.event_id,
            timestamp=ts,
            source=self.source,
            type=self.type,
            payload=self.payload,
            symbol_impact=list(self.symbol_impact),
            sequence=self.sequence,
        )


def make_event(
    type: EventType | str,
    source: str,
    payload: Dict[str, Any],
    *,
    symbol_impact: Optional[List[str]] = None,
    timestamp: Optional[float] = None,
    event_id: Optional[str] = None,
    sequence: int = 0,
) -> NormalizedEvent:
    """Ergonomic constructor used by producers across all layers."""
    t = type.value if isinstance(type, EventType) else type
    return NormalizedEvent(
        event_id=event_id or uuid.uuid4().hex,
        timestamp=timestamp if timestamp is not None else time.time(),
        source=source,
        type=t,
        payload=payload,
        symbol_impact=list(symbol_impact or []),
        sequence=sequence,
    )
