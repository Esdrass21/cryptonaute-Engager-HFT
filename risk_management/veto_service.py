# risk_management/veto_service.py
"""
Risk, Capital & Governance (Layer 6 -- S7). UNIFIED absolute veto.

This is the single authority that can block an order. The Execution Engine (Layer 5) refuses
any order not explicitly approved by this service. It has authority over all other layers,
including Mistral's recommendations (architecture S7.2).

Phase 2: signing is delegated to a KeyManager (pluggable provider, key rotation).
Verification still supports the legacy raw-secret path for backward-compat in tests.
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
from core.keys import KeyManager, build_key_provider

from .circuit_breaker import CircuitBreaker
from .portfolio_optimizer import PortfolioOptimizer

log = logging.getLogger("risk.veto")


class RiskVetoService(Component):
    """Signs every order decision. Execution only acts on decision == 'allow'."""

    def __init__(self, bus, cfg, clock, metrics, key_manager: KeyManager | None = None):
        super().__init__(bus, cfg, clock, metrics, name="risk.veto")
        r = cfg.get("risk", {})
        self.max_pos_pct = float(r.get("max_position_pct", 2.0))
        self.hard_cap_pct = float(r.get("hard_cap_pct", 5.0))          # non-negotiable
        self.max_dd_pct = float(r.get("max_daily_drawdown_pct", 2.0))
        self.max_corr = float(r.get("max_correlation", 0.7))
        self.var_conf = float(r.get("var_confidence", 0.95))
        self.enabled = bool(r.get("veto_service", {}).get("enabled", True))

        # Phase 2: key manager -- no dev fallback
        if key_manager is not None:
            self.km = key_manager
        else:
            self.km = KeyManager(
                build_key_provider(cfg.get("core.keys", None)),
                mode=cfg.get("mode", "paper"),
            )

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.warning("Risk veto DISABLED -- UNSAFE, only for tests")
            return
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self.on_proposal)
        log.info("Risk veto service online (hard_cap=%.2f%% dd=%.2f%%)",
                 self.hard_cap_pct, self.max_dd_pct)

    def on_proposal(self, ev: NormalizedEvent) -> None:
        # Only vet proposals that haven't been executed yet (the raw strategy proposal).
        if ev.payload.get("execute", False):
            return
        decision, reasons = self._evaluate(ev.payload)
        proposal_ref = ev.event_id
        proposal_copy = dict(ev.payload)

        # Sign via KeyManager (HMAC-SHA256 over flat canonical payload)
        from risk_management.signing import sign_veto
        signed_payload, sig_dict = sign_veto(
            self.km, ev.payload, decision,
            reason=",".join(reasons), ts=self.clock.now()
        )

        self.emit(make_event(
            EventType.RISK_VETO, "risk.veto",
            payload={"symbol": ev.payload.get("symbol"), "decision": decision,
                     "reasons": reasons, "signature": sig_dict,
                     "signed_payload": signed_payload,
                     "proposal_ref": proposal_ref, "proposal": proposal_copy},
            symbol_impact=[ev.payload.get("symbol")] if ev.payload.get("symbol") else [],
            timestamp=self.clock.now(),
        ))
        self.metrics.inc("risk_veto_emitted")

    # ---- verification (delegates to KeyManager) ------------------------
    @classmethod
    def verify(cls, secret_or_km, payload: dict) -> bool:
        """Return True iff the RISK_VETO payload carries a valid signature.

        Accepts either a KeyManager or a raw-secret string (legacy test compat).
        """
        sig = payload.get("signature")
        if not sig:
            return False
        proposal = payload.get("proposal", {}) or {}
        decision = payload.get("decision", "")

        # New path: KeyManager (verifies with key_id carried in sig)
        if isinstance(secret_or_km, KeyManager):
            from risk_management.signing import verify_veto
            return verify_veto(secret_or_km, proposal, decision, sig)

        # Legacy path: raw-secret string
        if not sig or not isinstance(sig, str):
            return False
        blob = json.dumps({k: proposal.get(k) for k in sorted(proposal)}, sort_keys=True) + decision
        expected = hashlib.sha256((blob + secret_or_km).encode()).hexdigest()[:16]
        return expected == sig

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
