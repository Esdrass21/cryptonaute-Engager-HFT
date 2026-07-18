"""Market-regime classification (Layer 2 — §3.5).

Lightweight statistical regime detector: trending / ranging / volatile, plus a confidence.
Used as a *context feature* in Layer 4 fusion (architecture insists it is a feature, never a
sole trigger). Synchronous + fast so it can live near the critical path.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event
from core.metrics import Metrics

log = logging.getLogger("analytics.regime")


class RegimeDetector(Component):
    def __init__(self, bus, cfg, clock, metrics, lookback: int = 100):
        super().__init__(bus, cfg, clock, metrics, name="analytics.regime")
        self.lookback = lookback
        self._closes: Dict[str, list] = {}

    def start(self) -> None:
        self._running = True
        for s in self.cfg.get("symbols", []):
            self._closes[s] = []
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        log.info("Regime detector online")

    def on_candle(self, ev) -> None:
        sym = ev.payload["symbol"]
        self._closes.setdefault(sym, []).append(ev.payload["close"])
        if len(self._closes[sym]) > self.lookback:
            self._closes[sym] = self._closes[sym][-self.lookback:]
        if len(self._closes[sym]) >= 20:
            self._emit(sym)

    def _emit(self, sym: str) -> None:
        c = np.array(self._closes[sym])
        rets = np.diff(np.log(c))
        vol = float(np.std(rets) * np.sqrt(288))
        trend = float((c[-1] - c[0]) / (c[0] + 1e-9))
        adf_like = float(np.std(rets)) / (np.mean(np.abs(rets)) + 1e-9)
        if vol > 2.5:
            regime, conf = "volatile", min(1.0, vol / 4)
        elif adf_like < 0.6 and abs(trend) > 0.003:
            regime, conf = "trending", min(1.0, abs(trend) * 100)
        else:
            regime, conf = "ranging", 0.6
        self.emit(make_event(
            EventType.REGIME, "analytics.regime",
            payload={"symbol": sym, "regime": regime, "confidence": float(conf), "vol": vol},
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("regime_emitted")
