"""Tests for the bounded MT5 paper-execution smoke harness (no live MT5 needed)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.events import EventType
from tools import mt5_paper_smoke as smoke

PROJECT = Path(__file__).resolve().parents[1]


class FakeBus:
    def __init__(self) -> None:
        self.handlers = []

    def subscribe(self, event_type, handler) -> None:
        self.handlers.append(handler)

    def emit(self, event_type: str, payload: dict) -> None:
        for handler in self.handlers:
            handler(SimpleNamespace(type=event_type, payload=payload))


def test_monitor_counts_the_signal_to_fill_path():
    bus = FakeBus()
    mon = smoke.SessionMonitor(bus)
    assert bus.handlers == [mon.on_event]
    bus.emit(EventType.TICK.value, {})
    bus.emit(EventType.CANDLE.value, {})
    bus.emit(EventType.ORDER_PROPOSAL.value, {"symbol": "EURUSD"})
    bus.emit(EventType.RISK_VETO.value, {"decision": "allow"})
    bus.emit(EventType.RISK_VETO.value, {"decision": "deny"})
    bus.emit(EventType.ORDER_FILL.value, {"symbol": "EURUSD", "closed": False})
    bus.emit(EventType.ORDER_FILL.value, {"symbol": "EURUSD", "closed": True})
    bus.emit(EventType.ORDER_REJECT.value, {})
    s = mon.summary()
    assert s["ticks"] == 1 and s["candles"] == 1
    assert s["proposals"] == 1 and s["veto_allow"] == 1 and s["veto_deny"] == 1
    assert s["fills"] == 1 and s["exits"] == 1 and s["rejects"] == 1
    assert s["proposal_symbols"] == ["EURUSD"]
    assert s["fill_symbols"] == ["EURUSD"]


class FakeOrchestrator:
    def __init__(self, emit: bool = True) -> None:
        self.bus = FakeBus()
        self.emit = emit
        self.started = False
        self.stopped = False
        self._connected = True

    def start(self) -> None:
        self.started = True
        if self.emit:
            self.bus.emit(EventType.CANDLE.value, {"symbol": "EURUSD"})
            self.bus.emit(EventType.ORDER_PROPOSAL.value, {"symbol": "EURUSD"})
            self.bus.emit(EventType.RISK_VETO.value, {"decision": "allow"})
            self.bus.emit(EventType.ORDER_FILL.value, {"symbol": "EURUSD", "closed": False})

    def stop(self) -> None:
        self.stopped = True


def _cfg():
    return smoke.build_smoke_config(str(PROJECT / "config" / "default.yaml"),
                                    symbols=["EURUSD"])


def test_build_smoke_config_forces_safe_paper_mt5():
    cfg = _cfg()
    assert cfg.get("mode") == "paper"
    assert cfg.get("execution.broker") == "paper"
    assert cfg.get("data.market.source") == "mt5"
    assert cfg.get("event_log.enabled") is False
    assert cfg.get("notifications.telegram.enabled") is False
    assert cfg.get("data.news.source") == "sim"
    assert cfg.get("symbols") == ["EURUSD"]
    assert cfg.get("strategy.tsmom.symbols") == ["EURUSD"]
    assert cfg.get("strategy.noise_band.symbols") == ["EURUSD"]


def test_run_paper_session_stops_at_max_candles_and_reports():
    orch = FakeOrchestrator()
    report = smoke.run_paper_session(_cfg(), seconds=5.0, max_candles=1,
                                     orchestrator=orch, poll_s=0.0)
    assert orch.started and orch.stopped
    assert report["candles"] == 1
    assert report["proposals"] == 1
    assert report["veto_allow"] == 1
    assert report["fills"] == 1
    assert report["connected"] is True
    assert report["mode"] == "paper" and report["broker"] == "paper"


def test_run_paper_session_times_out_without_events():
    orch = FakeOrchestrator(emit=False)
    orch._connected = False
    report = smoke.run_paper_session(_cfg(), seconds=0.01, max_candles=None,
                                     orchestrator=orch, poll_s=0.005)
    assert orch.stopped
    assert report["ticks"] == 0 and report["connected"] is False
    assert report["duration_s"] >= 0.0


class _FakeTick:
    bid = 1.1000
    ask = 1.1002


class _FakeMT5:
    initialized = True

    @staticmethod
    def initialize():
        return _FakeMT5.initialized

    @staticmethod
    def last_error():
        return (0, "ok")

    @staticmethod
    def symbol_select(symbol, enable):
        return True

    @staticmethod
    def symbol_info_tick(symbol):
        return _FakeTick()

    @staticmethod
    def shutdown():
        return None


def test_preflight_ok_and_symbol_mapping():
    report = smoke.run_preflight(_FakeMT5, ["EURUSD"], {"EURUSD": "EURUSDm"})
    assert report["ok"] is True
    row = report["symbols"][0]
    assert row["broker_symbol"] == "EURUSDm"
    assert row["bid"] > 0 and row["ask"] > 0


def test_preflight_reports_init_failure():
    class Down(_FakeMT5):
        @staticmethod
        def initialize():
            return False

    report = smoke.run_preflight(Down, ["EURUSD"], {})
    assert report["ok"] is False
    assert "initialize failed" in report["reason"]
