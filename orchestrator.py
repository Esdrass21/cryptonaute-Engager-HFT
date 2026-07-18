"""Platform orchestrator.

Wires all six layers onto one :class:`core.bus.MessageBus` and runs them. In ``paper`` mode
(architecture §8: mandatory before any real capital) the platform is fully functional with
the simulated market feed + paper broker. Switch ``data.market.source`` / ``execution.broker``
to ``mt5`` / ``crypto`` for live data and real execution.

The orchestrator is the single place that knows the layer wiring order. Validation (replay
harness) reuses the same component constructors, so live and validated paths share code.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import List, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config, setup_logging
from core.metrics import Metrics

from analytics.classic.macro import MacroAnalytics
from analytics.nlp.news_scorer import NewsScorer
from analytics.order_flow.orderbook import OrderFlowAnalytics
from analytics.quant.indicators import QuantAnalytics
from analytics.quant.regime import RegimeDetector
from data_ingestion.market_feeds.feeds import build_market_feed
from data_ingestion.news_feeds.feeds import build_news_feed
from execution.order_manager import OrderManager
from risk_management.circuit_breaker import CircuitBreaker
from risk_management.portfolio_optimizer import PortfolioOptimizer
from risk_management.veto_service import RiskVetoService
from self_correction.mistral_client import AnomalyDetector, PostTradeReviewer
from strategy_engine.expected_value import ExpectedValueEngine
from strategy_engine.signal_fusion import SignalFusion
from strategy_engine.state_machine import StrategyOrchestrator

log = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        setup_logging(cfg.get("observability.log_level", "INFO"))
        self.clock = Clock()
        self.bus = MessageBus(
            backend=cfg.get("bus.backend", "memory"),
            redis_url=cfg.get("bus.redis_url"),
        )
        self.metrics = Metrics(
            enable_prometheus=cfg.get("observability.prometheus.enabled", False),
            port=cfg.get("observability.prometheus.port", 9100),
        )
        self.components: List[Component] = []
        self._wire()

    def _wire(self) -> None:
        c, bus, clk, m = self.cfg, self.bus, self.clock, self.metrics
        # Layer 1
        self.market = build_market_feed(bus, c, clk, m)
        self.news = build_news_feed(bus, c, clk, m)
        # Layer 2
        self.quant = QuantAnalytics(bus, c, clk, m)
        self.regime = RegimeDetector(bus, c, clk, m)
        self.macro = MacroAnalytics(bus, c, clk, m)
        self.order_flow = OrderFlowAnalytics(bus, c, clk, m, enabled=c.get("analytics.order_flow.enabled", False))
        self.nlp = NewsScorer(bus, c, clk, m)
        # Layer 3
        self.review = PostTradeReviewer(bus, c, clk, m)
        self.anomaly = AnomalyDetector(bus, c, clk, m)
        # Layer 4
        self.fusion = SignalFusion(bus, c, clk, m)
        self.ev = ExpectedValueEngine(bus, c, clk, m)
        self.fsm = StrategyOrchestrator(bus, c, clk, m)
        # Layer 5
        self.execution = OrderManager(bus, c, clk, m)
        # Layer 6
        self.veto = RiskVetoService(bus, c, clk, m)
        self.portfolio = PortfolioOptimizer(bus, c, clk, m)
        self.circuit = CircuitBreaker(bus, c, clk, m)

        self.components = [
            self.market, self.news, self.quant, self.regime, self.macro,
            self.order_flow, self.nlp, self.review, self.anomaly,
            self.fusion, self.ev, self.fsm, self.execution,
            self.veto, self.portfolio, self.circuit,
        ]

    def start(self) -> None:
        for comp in self.components:
            try:
                comp.start()
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to start %s: %s", comp.name, exc)
        log.info("=== Cryptonaute HFT platform ONLINE (mode=%s) ===", self.cfg.get("mode"))
        self._emit_account_metric()

    def _emit_account_metric(self) -> None:
        try:
            state = self.execution.broker.get_account_state()
            self.metrics.set("account_drawdown_pct", state.get("drawdown_pct", 0.0))
            self.metrics.set("account_balance", state.get("balance", 0.0))
        except Exception:  # noqa: BLE001
            pass

    def run(self, steps: Optional[int] = None) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        self.start()
        try:
            if hasattr(self.market, "run_realtime"):
                self.market.run_realtime(steps=steps)
            else:
                while self.market.running:
                    time.sleep(1.0)
        finally:
            self.stop()

    def stop(self) -> None:
        for comp in self.components:
            comp.stop()
        log.info("Platform stopped. Final metrics: %s", self.metrics.snapshot())

    def snapshot(self) -> dict:
        return self.metrics.snapshot()
