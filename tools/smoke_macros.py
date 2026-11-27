"""Smoke test for the Macro surprise engine.

Verifies the engine subscribes to RELEASE_EXPECTED/RELEASE_ACTUAL and emits
MACRO events with surprise / direction / impact.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.macro_surprise import MacroSurpriseEngine
from core.bus import MessageBus
from core.events import EventType, make_event


def _make_engine():
    from core.config import Clock
    from core.event_log import InMemoryEventLog
    from core.metrics import Metrics
    bus = MessageBus(event_log=InMemoryEventLog())
    return bus, MacroSurpriseEngine(bus, None, Clock(), Metrics(), window=10)


def test_emits_macro_on_release_expected(caplog):
    bus, engine = _make_engine()
    engine.start()
    ev = make_event(
        EventType.RELEASE_EXPECTED,
        "data.bls",
        payload={"release_id": "CPI_YOY", "consensus": 0.30, "impact": "High",
                 "symbol_impact": ["EURUSD"], "scheduled_at": 1730000000.0},
        timestamp=0.0,
    )
    bus.publish(ev)
    assert engine._schedule["CPI_YOY"]["consensus"] == 0.30


def test_emits_macro_on_release_actual(caplog):
    bus, engine = _make_engine()
    engine.start()
    # publish the schedule (consensus 0.30) then the actual
    bus.publish(make_event(
        EventType.RELEASE_EXPECTED, "data.bls",
        payload={"release_id": "CPI_YOY", "consensus": 0.30,
                 "symbol_impact": ["EURUSD"], "scheduled_at": 1730000000.0},
        timestamp=0.0,
    ))
    bus.publish(make_event(
        EventType.RELEASE_ACTUAL, "data.bls",
        payload={"release_id": "CPI_YOY", "actual": 0.45, "consensus": 0.30,
                 "surprise": 0.15, "surprise_std": 0.15, "symbol_impact": ["EURUSD"]},
        timestamp=1700000000.0,
    ))
    assert engine._surprises["CPI_YOY"], "no surprises recorded"
    mac = [e for e in bus._event_log.events if e["type"] == EventType.MACRO]
    assert mac, "no MACRO event emitted"
    p = mac[0]["payload"]
    assert p["surprise"] == pytest.approx(0.15), p
    assert p["surprise_z"] == pytest.approx(1.0), p
    assert p["impact"] == "Medium", p
    assert p["direction"] == 1, p


def test_direction_bias(caplog):
    bus, engine = _make_engine()
    engine.start()
    bus.publish(make_event(
        EventType.RELEASE_ACTUAL, "data.bls",
        payload={"release_id": "NFP", "actual": 190.0, "consensus": 150.0,
                 "surprise": 40.0, "surprise_std": 70.0, "symbol_impact": ["USD"]},
        timestamp=0.0,
    ))
    bus.publish(make_event(
        EventType.RELEASE_ACTUAL, "data.bls",
        payload={"release_id": "NFP", "actual": 120.0, "consensus": 150.0,
                 "surprise": -30.0, "surprise_std": 70.0, "symbol_impact": ["USD"]},
        timestamp=0.0,
    ))
    assert engine._direction_count["NFP"][1] == 1
    assert engine._direction_count["NFP"][-1] == 1
