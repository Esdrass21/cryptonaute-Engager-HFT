# core/schema_registry.py
"""
Versioned event schema registry.

Every event published on the bus is validated against a registered schema.
Schemas are keyed by (event_type, version). Migrations chain forward so that
replaying an old log against a newer codebase is safe and deterministic.

Phase 1 validates the *structure* of the event envelope: the top-level
NormalizedEvent fields (event_id, timestamp, source, type, payload,
symbol_impact, sequence) and that the payload is a dict. Individual layers
can register stricter payload-field schemas later without breaking existing
emitters — the default is forward-compatible structural validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


class SchemaError(ValueError):
    """Raised when an event fails validation or has no registered schema."""


@dataclass(frozen=True)
class EventSchema:
    type: str
    version: int
    validate: Callable[[Dict[str, Any]], None]
    migrate: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: Dict[tuple[str, int], EventSchema] = {}
        self._latest: Dict[str, int] = {}

    def is_initialized(self) -> bool:
        return len(self._schemas) > 0

    def register(self, schema: EventSchema) -> None:
        key = (schema.type, schema.version)
        if key in self._schemas:
            raise SchemaError(f"Schema already registered: {key}")
        self._schemas[key] = schema
        self._latest[schema.type] = max(self._latest.get(schema.type, 0), schema.version)

    def get(self, event_type: str, version: int) -> EventSchema:
        key = (event_type, version)
        if key not in self._schemas:
            raise SchemaError(f"No schema for {event_type} v{version}")
        return self._schemas[key]

    def latest_version(self, event_type: str) -> int:
        if event_type not in self._latest:
            raise SchemaError(f"No schema for {event_type}")
        return self._latest[event_type]

    def has(self, event_type: str, version: int | None = None) -> bool:
        if version is None:
            return event_type in self._latest
        return (event_type, version) in self._schemas

    def validate(self, event: Dict[str, Any]) -> None:
        if "type" not in event:
            raise SchemaError("Event missing 'type' field")
        event_type = event["type"]
        # Buses built outside the orchestrator (tests, replay harness) auto-register
        # on first publish — idempotent, safe to call every time.
        if not registry.is_initialized():
            register_default_schemas()
        version = int(event.get("version", 1))
        self.get(event_type, version).validate(event)

    def migrate_to_latest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event_type = event["type"]
        version = int(event.get("version", 1))
        latest = self.latest_version(event_type)

        while version < latest:
            schema = self.get(event_type, version)
            if schema.migrate is None:
                raise SchemaError(f"No migration from {event_type} v{version} -> v{version + 1}")
            event = schema.migrate(event, version)
            version = int(event.get("version", version + 1))
        return event


# Module-level singleton — import and use directly.
registry = SchemaRegistry()


def _require_fields(*fields: str) -> Callable[[Dict[str, Any]], None]:
    def _validate(e: Dict[str, Any]) -> None:
        for f in fields:
            if f not in e:
                raise SchemaError(f"Missing field {f!r} in {e.get('type')} v{e.get('version', 1)}")
    return _validate


def _base_validate(e: Dict[str, Any]) -> None:
    """Validate the NormalizedEvent top-level structure."""
    _require_fields("event_id", "timestamp", "source", "type", "payload", "symbol_impact", "sequence")(e)


def _structural_validate(e: Dict[str, Any]) -> None:
    """
    Default structural validation: top-level shape + payload is a dict.
    Does NOT enforce specific payload keys — that's forward-compatible.
    """
    _base_validate(e)
    if not isinstance(e.get("payload"), dict):
        raise SchemaError(f"Missing or invalid 'payload' in {e.get('type')} v{e.get('version', 1)}")


def register_default_schemas() -> None:
    """
    Register the base event schemas shipped with the platform.
    Idempotent: safe to call more than once.
    """
    all_types = [
        "tick", "candle", "order_book", "news", "macro",
        "tech_signal", "regime", "of_signal", "news_score",
        "post_trade_review", "anomaly",
        "composite_signal", "order_proposal",
        "order_sent", "order_fill", "order_reject", "order_cancel", "order_exit",
        "risk_veto", "risk_breach",
        "nowcast", "release_expected", "release_actual",
        "control", "metric",
    ]

    for event_type in all_types:
        if registry.has(event_type, 1):
            continue
        registry.register(EventSchema(event_type, 1, _structural_validate))
