"""Risk, Capital & Governance (Layer 6 — §7). UNIFIED absolute veto.

This is the single authority that can block an order. The Execution Engine (Layer 5) refuses
any order not explicitly approved by this service. It has authority over all other layers,
including Mistral's recommendations (architecture §7.2).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import List, Tuple

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

from .circuit_breaker import CircuitBreaker
from .portfolio_optimizer import PortfolioOptimizer

log = logging.getLogger("risk.veto")


class RiskVetoService(Component):
    """Signs every order decision. Execution only acts on decision == 'allow'."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="risk.veto")
        r = cfg.get("risk", {})
        self.max_pos_pct = float(r.get("max_position_pct", 2.0))
        self.hard_cap_pct = float(r.get("hard_cap_pct", 5.0))          # non-negotiable
        self.max_dd_pct = float(r.get("max_daily_drawdown_pct", 2.0))
        self.max_corr = float(r.get("max_correlation", 0.7))
        self.var_conf = float(r.get("var_confidence", 0.95))
        self.secret = cfg.secret("RISK_SIGNING_KEY", "cryptonaute-local-dev-key")
        self.enabled = bool(r.get("veto_service", {}).get("enabled", True))

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.warning("Risk veto DISABLED — UNSAFE, only for tests")
            return
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self.on_proposal)
        log.info("Risk veto service online (hard_cap=%.2f%% dd=%.2f%%)",
                 self.hard_cap_pct, self.max_dd_pct)

    def on_proposal(self, ev: NormalizedEvent) -> None:
        # Only vet proposals that haven't been executed yet (the raw strategy proposal).
        if ev.payload.get("execute", False):
            return
        decision, reasons = self._evaluate(ev.payload)
        signed = self._sign(ev.payload, decision)
        self.emit(make_event(
            EventType.RISK_VETO, "risk.veto",
            payload={"symbol": ev.payload.get("symbol"), "decision": decision,
                     "reasons": reasons, "signature": signed,
                     "proposal_ref": ev.event_id},
            symbol_impact=[ev.payload.get("symbol")] if ev.payload.get("symbol") else [],
            timestamp=self.clock.now(),
        ))
        self.metrics.inc("risk_veto_emitted")

    # ---- evaluation ---------------------------------------------------
    def _evaluate(self, proposal: dict) -> Tuple[str, List[str]]:
        sym = proposal.get("symbol")
        reasons: List[str] = []
        pos_pct = abs(proposal.get("position_pct", self.max_pos_pct))
        if pos_pct > self.hard_cap_pct:
            reasons.append(f"position {pos_pct:.2f}% exceeds HARD CAP {self.hard_cap_pct:.2f}%")
        if pos_pct > self.max_pos_pct:
            reasons.append(f"position {pos_pct:.2f}% exceeds soft cap {self.max_pos_pct:.2f}%")
        dd = proposal.get("account_drawdown_pct", 0.0)
        if dd >= self.max_dd_pct:
            reasons.append(f"daily drawdown {dd:.2f}% >= limit {self.max_dd_pct:.2f}%")
        corr = proposal.get("max_open_correlation", 0.0)
        if corr > self.max_corr:
            reasons.append(f"correlated exposure {corr:.2f} > {self.max_corr:.2f}")
        decision = "deny" if reasons else "allow"
        return decision, reasons

    def _sign(self, proposal: dict, decision: str) -> str:
        blob = json.dumps({k: proposal.get(k) for k in sorted(proposal)}, sort_keys=True) + decision
        return hashlib.sha256((blob + self.secret).encode()).hexdigest()[:16]
