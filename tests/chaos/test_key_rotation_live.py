"""Chaos: rotate the key while signatures are in flight. Old signatures must
still verify; new signatures must use the new key."""
import base64, json
from pathlib import Path
import pytest
from core.keys import KeyManager, FileKeyProvider
from risk_management.signing import sign_veto, verify_any

pytestmark = pytest.mark.chaos


def test_rotation_does_not_invalidate_in_flight_signatures(tmp_path: Path):
    keyring = tmp_path / "keyring.json"
    v1 = base64.b64encode(b"1" * 32).decode()
    v2 = base64.b64encode(b"2" * 32).decode()
    keyring.write_text(json.dumps({"current": "v1", "keys": {"v1": v1}}))

    km = KeyManager(FileKeyProvider(keyring), mode="paper")
    proposal = {"proposal_id": "p1", "symbol": "X", "side": "buy", "qty": 1.0}
    signed, sig = sign_veto(km, proposal, "allow", "", 1.0)

    # Rotate: add v2 and promote it.
    keyring.write_text(json.dumps({"current": "v2", "keys": {"v1": v1, "v2": v2}}))

    # Reload is async in production; for tests, create a new KM instance
    # (simulates a reload or a new process seeing the updated keyring).
    km2 = KeyManager(FileKeyProvider(keyring), mode="paper")

    # Old signature still verifies (v1 is in the keyring).
    assert verify_any(km2, {"signature": sig, "signed_payload": signed}) is True

    # New signature uses v2.
    signed2, sig2 = sign_veto(km2, proposal, "allow", "", 2.0)
    assert sig2["key_id"] == "v2"
    assert verify_any(km2, {"signature": sig2, "signed_payload": signed2}) is True
