"""Trade journal (Layer 5 — execution audit trail).

A thin terminal-grade logger: every fill, close, reject, veto and circuit-breaker trip is
printed once with P&L, so the console shows the *trade flow* instead of FSM state noise.
All lines also land in ``logs/trading.log`` via the root file handler (core/config).
"""

from __future__ import annotations

import logging

from core.component import Component
from core.events import EventType, NormalizedEvent

log = logging.getLogger("execution.journal")


class TradeJournal(Component):
    """Subscribes to execution + risk events and emits one clean line per trade action."""

    def __init__(self, bus, cfg, clock, metrics, name="execution.journal"):
        super().__init__(bus, cfg, clock, metrics, name)
        self._open_pnl: dict = {}  # sym -> running realized pnl for this session

    def start(self) -> None:
        super().start()
        for ev, handler in (
            (EventType.ORDER_FILL, self.on_fill),
            (EventType.ORDER_REJECT, self.on_reject),
            (EventType.RISK_VETO, self.on_veto),
            (EventType.RISK_BREACH, self.on_breach),
        ):
            self.bus.subscribe(ev, handler)
        log.info("Trade journal online")

    # ---- handlers ------------------------------------------------------
    def on_fill(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        sym = p.get("symbol", "?")
        side = p.get("side", "?")
        px = p.get("price", 0.0)
        if p.get("closed"):
            pnl = p.get("pnl", 0.0)
            self._open_pnl[sym] = self._open_pnl.get(sym, 0.0) + pnl
            net = self._open_pnl[sym]
            log.info("CLOSE %s %s @ %.5f  pnl=%+.2f  net=%+.2f  ticket=%s",
                     sym, side, px, pnl, net, p.get("ticket"))
            self.metrics.inc("fills")
            self.metrics.set(f"pnl_{sym}", net)
        else:
            log.info("FILL  %s %s %.2f @ %.5f  ticket=%s",
                     sym, side, p.get("volume", 0.0), px, p.get("ticket"))
            self.metrics.inc("fills")

    def on_reject(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        log.warning("REJECT %s reason=%s", p.get("symbol", "?"), p.get("reason", "?"))
        self.metrics.inc("rejects")

    def on_veto(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        dec = p.get("decision", "?")
        if dec == "deny":
            log.warning("VETO  DENY %s position_pct=%.2f%%",
                        p.get("symbol", "?"), p.get("position_pct", 0.0))
        else:
            log.info("VETO  ALLOW %s", p.get("symbol", "?"))
        self.metrics.inc("vetoes")

    def on_breach(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        log.critical("BREACH %s=%.2f%% -> trading HALT",
                     p.get("name", "?"), p.get("value", 0.0))
