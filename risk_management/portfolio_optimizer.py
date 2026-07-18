"""Portfolio optimizer (Layer 6 — §7.1).

Continuous risk allocation: correlation, covariance, VaR, risk-contribution between active
symbols/strategies. Maintains a running covariance matrix consumed by the veto service for
correlated-exposure checks.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy.stats import norm

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType
from core.metrics import Metrics

log = logging.getLogger("risk.portfolio")


class PortfolioOptimizer(Component):
    def __init__(self, bus, cfg, clock, metrics, lookback: int = 250):
        super().__init__(bus, cfg, clock, metrics, name="risk.portfolio")
        self.lookback = lookback
        self.symbols = cfg.get("symbols", [])
        self._rets: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.correlation: Dict[str, Dict[str, float]] = {}
        self.var: Dict[str, float] = {}
        self.var_conf = float(cfg.get("risk.var_confidence", 0.95))

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        log.info("Portfolio optimizer online")

    def on_candle(self, ev) -> None:
        sym = ev.payload["symbol"]
        if sym not in self._rets:
            return
        self._rets[sym].append(ev.payload["close"])
        if len(self._rets[sym]) > self.lookback:
            self._rets[sym].pop(0)
        if len(self._rets[sym]) >= 30:
            self._update()

    def _update(self) -> None:
        series = {s: np.array(self._rets[s], dtype=float) for s in self.symbols if len(self._rets[s]) >= 30}
        if len(series) < 2:
            return
        minlen = min(len(v) for v in series.values())
        names = list(series.keys())
        returns = np.column_stack([np.diff(np.log(series[s][-minlen:])) for s in names])
        corr = np.corrcoef(returns.T)
        self.correlation = {names[i]: {names[j]: float(corr[i][j]) for j in range(len(names))}
                            for i in range(len(names))}
        z = norm.ppf(self.var_conf)
        for i, s in enumerate(names):
            vol = float(np.std(returns[:, i]) * np.sqrt(288))
            self.var[s] = z * vol
        self.metrics.set("portfolio_var_max", max(self.var.values()) if self.var else 0.0)

    def max_correlation_with(self, symbol: str, open_symbols: List[str]) -> float:
        best = 0.0
        for o in open_symbols:
            if o == symbol:
                continue
            best = max(best, abs(self.correlation.get(symbol, {}).get(o, 0.0)))
        return best
