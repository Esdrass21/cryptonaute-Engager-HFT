# tests/test_bus_event_log_integration.py
"""Integration tests: bus publishes -> event log persists."""
import pytest
from core.bus import MessageBus
from core.event_log import InMemoryEventLog
from core.schema_registry import register_default_schemas
from core.events import make_event, EventType


def test_bus_persists_before_dispatch():
    register_default_schemas()
    log = InMemoryEventLog()
    bus = MessageBus(backend="memory", event_log=log)

    ev = make_event(EventType.TICK, "sim", {"symbol": "BTC", "price": 1.0, "ts": 1})
    bus.publish(ev)

    persisted = list(log.replay())
    assert len(persisted) == 1
    assert persisted[0]["type"] == "tick"
    assert persisted[0]["payload"]["symbol"] == "BTC"


def test_bus_rejects_invalid_event_before_persist():
    """An event missing required top-level fields is rejected and not persisted."""
    register_default_schemas()
    log = InMemoryEventLog()
    bus = MessageBus(backend="memory", event_log=log)

    # Create an event and break its to_dict to omit a required field
    ev = make_event(EventType.TICK, "sim", {"symbol": "BTC"})
    original_to_dict = ev.to_dict

    def broken_to_dict():
        d = original_to_dict()
        del d["source"]  # required field missing
        return d

    ev.to_dict = broken_to_dict

    with pytest.raises(Exception):
        bus.publish(ev)

    assert len(list(log.replay())) == 0


def test_bus_replay_log_migrates():
    register_default_schemas()
    log = InMemoryEventLog()
    log.append({"type": "tick", "version": 1, "event_id": "old", "timestamp": 1.0,
                "source": "sim", "symbol_impact": [], "sequence": 0,
                "payload": {"symbol": "BTC", "price": 1.0, "ts": 1}})

    bus = MessageBus(backend="memory", event_log=None)
    events = list(bus.replay_log(log))
    assert len(events) == 1
    assert events[0]["type"] == "tick"
