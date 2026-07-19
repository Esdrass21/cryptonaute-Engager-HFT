# risk_management/signing.py
"""
Canonical signing adapter for RISK_VETO events.

Bridges KeyManager to the existing veto payload shape:
- sign_veto: signs (proposal_dict + decision + reason + ts) via KeyManager, returns (signed_payload, sig_dict)
- verify_veto: verifies the signature against (proposal_dict + decision + reason + ts)
- verify_any: unified verifier for both RISK_VETO events and forwarded execution proposals

The signature dict carries key_id so old signatures still verify after rotation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Tuple

from core.keys import KeyManager


def sign_veto(km: KeyManager, proposal: dict, decision: str, reason: str = "", ts: float = 0.0) -> Tuple[dict, dict]:
    """Sign a veto decision. Returns (signed_payload, signature_dict).

    The signed payload is a flat canonical dict (sorted by key) of proposal fields +
    decision + reason + ts — so verification checks the exact bytes that were signed.
    """
    signed_payload = {
        **{k: proposal.get(k) for k in sorted(proposal)},
        "decision": decision,
        "reason": reason,
        "ts": ts,
    }
    sig = km.sign(signed_payload)
    return signed_payload, sig.to_dict()


def verify_veto(km: KeyManager, proposal: dict, decision: str, sig: dict, reason: str = "", ts: float = 0.0) -> bool:
    """Verify a veto signature against the proposal + decision."""
    signed_payload = {
        **{k: proposal.get(k) for k in sorted(proposal)},
        "decision": decision,
        "reason": reason,
        "ts": ts,
    }
    return km.verify(signed_payload, sig)


def verify_any(km, payload: dict) -> bool:
    """
    Verify a RISK_VETO event payload OR a forwarded execution proposal.

    Preferred (Phase 2+): dict signature over the exact signed_payload.
        - RISK_VETO event    : keys  signature      / signed_payload
        - forwarded proposal : keys  veto_signature / veto_signed_payload

    Legacy (deprecated, removed in Phase 3):
        - v1 hex string under signature / veto_signature / signature_v1 / veto_signature_v1
        - recomputed over {"proposal": ...} + decision, keyed by key_bytes.hex().
    """
    # ---- preferred path: dict signature over signed_payload -------------
    for sig_key, payload_key in (
        ("signature",      "signed_payload"),
        ("veto_signature", "veto_signed_payload"),
    ):
        sig = payload.get(sig_key)
        signed = payload.get(payload_key)
        if isinstance(sig, dict) and isinstance(signed, dict):
            # A present dict signature must verify; no silent fallback (fail-closed).
            return km.verify(signed, sig)

    # ---- legacy v1 path -------------------------------------------------
    sig_v1 = payload.get("signature_v1") or payload.get("veto_signature_v1")
    if not isinstance(sig_v1, str) or not sig_v1:
        cand = payload.get("signature") or payload.get("veto_signature")
        if isinstance(cand, str):
            sig_v1 = cand
    if not isinstance(sig_v1, str) or not sig_v1:
        return False

    proposal = payload.get("proposal") or payload.get("veto_proposal") or {}
    decision = payload.get("decision") or payload.get("veto_decision") or ""
    blob = json.dumps({k: proposal.get(k) for k in sorted(proposal)}, sort_keys=True) + decision
    for kid in km.provider.list_ids():
        try:
            key = km.provider.get(kid).key_bytes
        except Exception:
            continue
        if hashlib.sha256((blob + key.hex()).encode()).hexdigest()[:16] == sig_v1:
            return True
    return False
