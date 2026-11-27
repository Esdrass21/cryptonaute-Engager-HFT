"""Smoke test for the Portfolio VaR component.

Verifies the component subscribes to CANDLE events and emits PORTFOLIO_VaR
events with a parametric 95% and 99% VaR estimate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bus import MessageBus
from core.config import Clock
from core.events import EventType, make_event
from analytics.portfolio_var import PortfolioVaR


def _make_engine():
    from core.event_log import InMemoryEventLog
    from core.metrics import Metrics
    bus = MessageBus(event_log=InMemoryEventLog())
    return bus, PortfolioVaR(
        bus, None, Clock(), Metrics(),
        window=252, confidence=0.95, report_s=0.01,
        weights={"EURUSD": 0.5, "BTCUSD": 0.5},
    )


def test_emits_portfolio_var(caplog):
    bus, engine = _make_engine()
    engine.start()

    # portfolio: 50% EURUSD, 50% BTCUSD
    engine._weights = {'EURUSD': 0.5, 'BTCUSD': 0.5}

    # simulate a steady uptrend for both symbols (~0.25% gain per candle)
    close = 1.0
    for i in range(20):
        close *= 1.0025
        ts = i * 1.0
        for sym in engine._weights:
            bus.publish(make_event(
                EventType.CANDLE, 'market',
                payload={'symbol': sym, 'timeframe': 'M5', 'open': close - 0.01,
                         'high': close + 0.01, 'low': close - 0.01,
                         'close': close, 'volume': 1000, 'closed': True},
                symbol_impact=[sym], timestamp=ts,
            ))
        engine.tick()

    mac = [e for e in bus._event_log.events if e['type'] == EventType.PORTFOLIO_VaR]
    assert len(mac) == 1, f'expected 1 PORTFOLIO_VaR event, got {len(mac)}'
    p = mac[0]['payload']
    assert p['confidence'] == 0.95, p
    assert p['var95'] < 0.0, p          # rising market -> negative VaR (gain)
    assert p['var99'] < 0.0, p
    assert p['volatility'] > 0.0, p

    # the component emits once it has >= 20 returns (first emit).  With a real
    # clock the loop above is fast, so switch to virtual time and advance it well
    # past report_s to flush a second report reflecting the full 38-return stream
    # (19 returns per symbol).
    engine.clock.set_virtual(engine.clock.now() + 1.0)
    engine.tick()
    mac = [e for e in bus._event_log.events if e['type'] == EventType.PORTFOLIO_VaR]
    assert len(mac) == 2, f'expected 2 PORTFOLIO_VaR events, got {len(mac)}'
    p = mac[-1]['payload']
    assert p['n'] == 19 * 2, p          # 19 returns/symbol, skip 1st candle
