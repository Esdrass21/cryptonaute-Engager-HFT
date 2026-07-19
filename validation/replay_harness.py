# validation/replay_harness.py
"""Deterministic replay harness (architecture §8).

Phase 3 additions
-----------------
* One shared ``KeyManager`` — L4/L5/L6 verify against identical key material.
* Per-recorded-event equity sampling → a fixed-length return stream that aligns
  across trials. This is what PBO needs.
* Per-trade PnL collected from exit fills → a real return stream for DSR.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.bus import MessageBus
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent
from core.keys import KeyManager, build_key_provider
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
        self.bus = MessageBus(backend="memory", replay=True)
        self.events = events
        self.components = []

        # Shared key material — one provider, one manager, injected everywhere.
        provider = build_key_provider(cfg.get("core.keys", None))
        self.km = KeyManager(provider, mode=cfg.get("mode", "paper"))

        # Return / trade streams.
        self._trade_records: List[Tuple[float, float]] = []       # (exit_ts, pnl)
        self._equity_curve: List[Tuple[float, float]] = []        # (ts, balance)

        self._wire()

    # ------------------------------------------------------------------ #
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

        # L4/L5/L6 all share the same KeyManager.
        self.fsm = StrategyOrchestrator(self.bus, c, self.clock, self.metrics,
                                        key_manager=self.km)
        self.veto = RiskVetoService(self.bus, c, self.clock, self.metrics,
                                    key_manager=self.km)
        self.portfolio = PortfolioOptimizer(self.bus, c, self.clock, self.metrics)
        self.circuit = CircuitBreaker(self.bus, c, self.clock, self.metrics)

        self.broker = build_broker(c, self.clock)
        self.execution = OrderManager(self.bus, c, self.clock, self.metrics,
                                      broker=self.broker, key_manager=self.km)

        self.review = PostTradeReviewer(self.bus, c, self.clock, self.metrics)
        self.anomaly = AnomalyDetector(self.bus, c, self.clock, self.metrics)

        self.components = [
            self.market_quant, self.regime, self.macro, self.of, self.nlp,
            self.fusion, self.ev, self.fsm, self.veto, self.portfolio,
            self.circuit, self.execution, self.review, self.anomaly,
        ]

        # Trade recorder — must come *after* execution so it sees the fills.
        self.bus.subscribe(EventType.ORDER_FILL, self._on_fill)

        for comp in self.components:
            comp.start()

    # ------------------------------------------------------------------ #
    def _on_fill(self, ev: NormalizedEvent) -> None:
        # Only exit fills carry realized PnL.
        if ev.payload.get("closed", False):
            ts = float(getattr(ev, "timestamp", self.clock.now()))
            pnl = float(ev.payload.get("pnl", 0.0))
            self._trade_records.append((ts, pnl))

    # ------------------------------------------------------------------ #
    def queue(self) -> None:
        for ev in self.events:
            self.bus.enqueue(ev)

    def run(self) -> int:
        """
        Replay the recorded events, sampling broker equity once per *recorded*
        event (reaction events are filtered by identity). The result is a
        fixed-length equity curve that aligns across trials.
        """
        self.queue()
        recorded_ids = {id(e) for e in self.events}

        processed = 0
        while self.bus.replay_size() > 0:
            ev = self.bus.step_replay()
            if ev is None:
                break
            processed += 1
            if id(ev) in recorded_ids:
                bal = float(self.broker.get_account_state().get("balance", 0.0))
                self._equity_curve.append((float(ev.timestamp), bal))

        # Realize P&L at end of recording.
        self.execution.close_all()
        if self.events:
            self._equity_curve.append((
                float(self.events[-1].timestamp),
                float(self.broker.get_account_state().get("balance", 0.0)),
            ))
        return processed

    def step(self) -> None:
        self.bus.step_replay()

    # ------------------------------------------------------------------ #
    def account_state(self) -> dict:
        return self.broker.get_account_state()

    def open_positions(self) -> List[dict]:
        return self.broker.get_open_positions()

    # ---- Phase 3 accessors ------------------------------------------- #
    def trade_records(self) -> List[Tuple[float, float]]:
        """List of ``(exit_ts, pnl)`` for every closed trade."""
        return list(self._trade_records)

    def trade_pnls(self) -> np.ndarray:
        """Per-trade realized PnL, as an array."""
        return np.array([p for _, p in self._trade_records], dtype=float)

    def equity_curve(self) -> List[Tuple[float, float]]:
        """``(ts, balance)`` sampled once per recorded event."""
        return list(self._equity_curve)

    def equity_returns(self) -> np.ndarray:
        """Simple per-recorded-event returns from the equity curve."""
        if len(self._equity_curve) < 2:
            return np.array([], dtype=float)
        eq = np.array([b for _, b in self._equity_curve], dtype=float)
        prev = np.where(eq[:-1] == 0.0, 1e-9, eq[:-1])
        return np.diff(eq) / prev

    # ---- Backwards-compat alias -------------------------------------- #
    def oos_returns(self) -> np.ndarray:
        """Alias for `equity_returns()` — kept for any un-migrated callers."""
        return self.equity_returns()

    # ------------------------------------------------------------------ #
    def stop_all(self) -> None:
        for comp in self.components:
            comp.stop()
