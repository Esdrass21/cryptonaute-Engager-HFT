# tests/test_event_log.py
"""Tests for the durable event log and schema registry (Phase 1)."""
import json
from pathlib import Path
import pytest

from core.event_log import FileEventLog, InMemoryEventLog
from core.schema_registry import registry, register_default_schemas, SchemaError, EventSchema, SchemaRegistry


def test_file_event_log_roundtrip(tmp_path: Path):
    log_path = tmp_path / "events.jsonl"
    log = FileEventLog(log_path, fsync=False)
    log.append({"type": "tick", "version": 1, "payload": {"symbol": "BTC", "price": 100.0, "ts": 1},
                "event_id": "a" * 32, "timestamp": 1.0, "source": "sim",
                "symbol_impact": [], "sequence": 0})
    log.append({"type": "tick", "version": 1, "payload": {"symbol": "BTC", "price": 101.0, "ts": 2},
                "event_id": "b" * 32, "timestamp": 2.0, "source": "sim",
                "symbol_impact": [], "sequence": 1})
    log.close()

    replayed = list(FileEventLog(log_path, fsync=False).replay())
    assert len(replayed) == 2
    assert replayed[0]["payload"]["price"] == 100.0
    assert replayed[1]["payload"]["price"] == 101.0


def test_in_memory_event_log_order():
    log = InMemoryEventLog()
    for i in range(5):
        log.append({"type": "tick", "version": 1, "payload": {"symbol": "ETH", "price": i, "ts": i},
                    "event_id": f"id{i}", "timestamp": float(i), "source": "sim",
                    "symbol_impact": [], "sequence": i})
    prices = [e["payload"]["price"] for e in log.replay()]
    assert prices == [0, 1, 2, 3, 4]


def test_file_log_is_deterministic(tmp_path: Path):
    """Same events -> byte-identical file."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for p in (a, b):
        log = FileEventLog(p, fsync=False)
        log.append({"type": "tick", "version": 1, "payload": {"symbol": "BTC", "price": 1.0, "ts": 1},
                    "event_id": "a" * 32, "timestamp": 1.0, "source": "sim",
                    "symbol_impact": [], "sequence": 0})
        log.append({"type": "tick", "version": 1, "payload": {"symbol": "BTC", "price": 2.0, "ts": 2},
                    "event_id": "b" * 32, "timestamp": 2.0, "source": "sim",
                    "symbol_impact": [], "sequence": 1})
        log.close()
    assert a.read_bytes() == b.read_bytes()


def test_schema_rejects_missing_top_level_field():
    register_default_schemas()
    with pytest.raises(SchemaError):
        # missing 'source' field
        registry.validate({"type": "tick", "version": 1,
                           "event_id": "x" * 32, "timestamp": 1.0,
                           "symbol_impact": [], "sequence": 0,
                           "payload": {"symbol": "BTC"}})


def test_schema_rejects_missing_payload():
    register_default_schemas()
    with pytest.raises(SchemaError):
        registry.validate({"type": "tick", "version": 1,
                           "event_id": "x" * 32, "timestamp": 1.0, "source": "sim",
                           "symbol_impact": [], "sequence": 0})


def test_schema_accepts_structural_event():
    register_default_schemas()
    # This should pass — structural validation only requires top-level fields + payload is dict
    registry.validate({"type": "tick", "version": 1,
                       "event_id": "x" * 32, "timestamp": 1.0, "source": "sim",
                       "symbol_impact": ["BTC"], "sequence": 0,
                       "payload": {"symbol": "BTC", "price": 100.0}})


def test_schema_rejects_unknown_type():
    register_default_schemas()
    with pytest.raises(SchemaError):
        registry.validate({"type": "nope", "version": 1,
                           "event_id": "x" * 32, "timestamp": 1.0, "source": "sim",
                           "symbol_impact": [], "sequence": 0, "payload": {}})


def test_schema_migration_chain_forward():
    """Old version migrates forward when a newer version is registered."""
    r = SchemaRegistry()

    def v1_validate(e): pass
    def v1_to_v2(e, _):
        return {**e, "version": 2, "extra": "added"}

    def v2_validate(e): assert "extra" in e

    r.register(EventSchema("X", 1, v1_validate, migrate=v1_to_v2))
    r.register(EventSchema("X", 2, v2_validate))

    out = r.migrate_to_latest({"type": "X", "version": 1})
    assert out["version"] == 2 and out["extra"] == "added"


def test_register_default_schemas_is_idempotent():
    register_default_schemas()
    register_default_schemas()  # must not raise


def test_build_file_log_from_cfg(tmp_path: Path):
    from core.event_log import build_event_log
    cfg = {"enabled": True, "backend": "file", "path": str(tmp_path / "test.jsonl"), "fsync": False}
    log = build_event_log(cfg)
    assert isinstance(log, FileEventLog)
    log.close()


def test_build_memory_log_from_cfg():
    from core.event_log import build_event_log
    cfg = {"enabled": True, "backend": "memory"}
    log = build_event_log(cfg)
    assert isinstance(log, InMemoryEventLog)


def test_build_log_returns_none_when_disabled():
    from core.event_log import build_event_log
    assert build_event_log(None) is None
    assert build_event_log({"enabled": False}) is None
