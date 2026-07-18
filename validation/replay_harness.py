"""Deterministic replay harness (architecture §8).

Replays an exact recorded sequence of events (ticks, candles, news) through the *real*
pipeline components wired onto a replay-mode :class:`core.bus.MessageBus`. Because the bus
dispatches subscribers in insertion order, the strategy engine, EV engine, risk veto and
execution all run exactly as they would live — but with a virtual clock and a paper broker.

This lets us:
  * compare expected vs actual strategy behaviour on historical data,
  * measure realized slippage vs the budgeted tolerance,
  * validate any strategy/parameter change before deployment (§7.3 step 4: paper trading).

The harness is also what the test-suite uses to exercise the full 6-layer flow with zero
external services.
"""

from __future__ import annotations

import logging
from typing import List

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import NormalizedEvent
from core.metrics import Metrics

from analytics.classic.macro import MacroAnalytics
from analytics.nlp.news_scorer import NewsScorer
from analytics.order_flow.orderbook import OrderFlowAnalytics
from analytics.quant.indicators import QuantAnalytics
from analytics.quant.regime import RegimeDetector
from execution.broker_adapter import build_broker
from execution.order_manager import OrderManager
from risk_management.circuit_breaker import CircuitBreaker
from risk_management.portfolio_optimizer import PortfolioOptimizer
from risk_management.veto_service import RiskVetoService
from self_correction.mistral_client import AnomalyDetector, PostTradeReviewer
from strategy_engine.expected_value import ExpectedValueEngine
from strategy_engine.signal_fusion import SignalFusion
from strategy_engine.state_machine import StrategyOrchestrator

log = logging.getLogger("validation.replay")


class ReplayHarness:
    """Wires the full platform onto a replay bus and replays a recorded event list."""

    def __init__(self, cfg: Config, events: List[NormalizedEvent]):
        self.cfg = cfg
        self.clock = Clock()
        self.clock.set_virtual(events[0].timestamp if events else 0.0)
        self.metrics = Metrics()
        # replay-mode bus: buffers, dispatched in order by run_replay()
        self.bus = MessageBus(backend="memory", replay=True)
        self.events = events
        self.components = []
        self._wire()

    def _wire(self) -> None:
        c = self.cfg
        self.market_quant = QuantAnalytics(self.bus, c, self.clock, self.metrics)
        self.regime = RegimeDetector(self.bus, c, self.clock, self.metrics)
        self.macro = MacroAnalytics(self.bus, c, self.clock, self.metrics)
        self.of = OrderFlowAnalytics(self.bus, c, self.clock, self.metrics,
                                     enabled=c.get("analytics.order_flow.enabled", False))
        self.nlp = NewsScorer(self.bus, c, self.clock, self.metrics)
        self.fusion = SignalFusion(self.bus, c, self.clock, self.metrics)
        self.ev = ExpectedValueEngine(self.bus, c, self.clock, self.metrics)
        self.fsm = StrategyOrchestrator(self.bus, c, self.clock, self.metrics)
        self.veto = RiskVetoService(self.bus, c, self.clock, self.metrics)
        self.portfolio = PortfolioOptimizer(self.bus, c, self.clock, self.metrics)
        self.circuit = CircuitBreaker(self.bus, c, self.clock, self.metrics)
        self.broker = build_broker(c, self.clock)
        self.execution = OrderManager(self.bus, c, self.clock, self.metrics, broker=self.broker)
        self.review = PostTradeReviewer(self.bus, c, self.clock, self.metrics)
        self.anomaly = AnomalyDetector(self.bus, c, self.clock, self.metrics)
        self.components = [self.market_quant, self.regime, self.macro, self.of, self.nlp,
                           self.fusion, self.ev, self.fsm, self.veto, self.portfolio,
                           self.circuit, self.execution, self.review, self.anomaly]
        for comp in self.components:
            comp.start()

    def queue(self) -> None:
        for ev in self.events:
            self.bus.enqueue(ev)

    def run(self) -> int:
        self.queue()
        return self.bus.run_replay()

    def step(self) -> None:
        self.bus.step_replay()

    def account_state(self) -> dict:
        return self.broker.get_account_state()

    def open_positions(self) -> List[dict]:
        return self.broker.get_open_positions()

    def stop_all(self) -> None:
        for comp in self.components:
            comp.stop()
