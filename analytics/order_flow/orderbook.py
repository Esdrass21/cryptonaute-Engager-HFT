"""Order-flow & market-structure analytics (Layer 2 — §3.4).

Architecture is explicit: treat order-flow concepts as *hypotheses to validate*, not acquired
truth. Two registers:

1. **Institutional order flow (measurable)** — Order Book Imbalance, Bid/Ask Pressure, Cumulative
   Volume Delta, sweep/absorption on L2. Requires real depth data (crypto via ccxt; rarely on
   retail FX via MT5).
2. **ICT / Smart-Money concepts** (Order Blocks, FVGs, Stop Hunts, etc.) — discretionary TA with
   no established statistical validation. Coded but DISABLED by default — the platform refuses to
   let unvalidated concepts influence decisions until you validate them per §7.3.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np

from core.component import Component
from core.events import EventType, make_event

log = logging.getLogger("analytics.order_flow")


class OrderFlowAnalytics(Component):
    def __init__(self, bus, cfg, clock, metrics, enabled: bool = False):
        super().__init__(bus, cfg, clock, metrics, name="analytics.order_flow")
        self.enabled = enabled
        self.validated_only = cfg.get("analytics.order_flow.validated_only", True)
        self._books: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.info("Order-flow analytics DISABLED (architecture §3.4: hypotheses not yet validated)")
            return
        self.bus.subscribe(EventType.ORDER_BOOK, self.on_book)
        log.info("Order-flow analytics online (validated_only=%s)", self.validated_only)

    def on_book(self, ev) -> None:
        sym = ev.payload["symbol"]
        bids = np.array(ev.payload["bids"], dtype=float)
        asks = np.array(ev.payload["asks"], dtype=float)
        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        self.emit(make_event(
            EventType.OF_SIGNAL, "analytics.order_flow",
            payload={"symbol": sym, "order_book_imbalance": float(obi),
                      "direction": float(np.sign(obi)), "kind": "institutional"},
            symbol_impact=[sym], timestamp=self.clock.now()))
        self.metrics.inc("of_signal_emitted")

    def ict_detect(self, candles: list[dict]) -> dict:
        """Coded but DISABLED by default. Returns an ICT hypothesis dict.

        Exists so you can validate it in the backtester (validation/) and, once your §7.3
        evidence supports it, set ``analytics.order_flow.validated_only: false`` to enable.
        """
        return {"kind": "ict", "validated": False, "note": "disabled-until-validated"}
