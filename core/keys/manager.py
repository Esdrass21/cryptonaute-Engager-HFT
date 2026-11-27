# core/keys/manager.py
"""
KeyManager — the single call site for signing and verification.

- Holds a KeyProvider.
- Refuses EnvKeyProvider in live mode unless RISK_ALLOW_ENV_KEYS=1.
- Signs payloads as: base64(hmac_sha256(key, canonical_json(payload))) + key_id.
- Verification looks up the key_id carried in the signature, so old signatures
  verify correctly after rotation.
- Rotate via provider.reload() (FileKeyProvider) — no restart needed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from core.keys.provider import (
    EnvKeyProvider,
    KeyProvider,
    KeyProviderError,
)

log = logging.getLogger("core.keys.manager")

SIG_VERSION = 1  # bump if the signature envelope format changes


@dataclass(frozen=True)
class Signature:
    key_id: str
    digest_b64: str
    v: int = SIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"key_id": self.key_id, "digest_b64": self.digest_b64, "v": self.v}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signature:
        return cls(key_id=d["key_id"], digest_b64=d["digest_b64"], v=int(d.get("v", 1)))


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


class KeyManager:
    def __init__(self, provider: KeyProvider, mode: str = "paper") -> None:
        self.provider = provider
        self.mode = mode
        if mode == "live" and isinstance(provider, EnvKeyProvider) \
                and os.environ.get("RISK_ALLOW_ENV_KEYS") != "1":
            raise KeyProviderError(
                "Refusing EnvKeyProvider in live mode. "
                "Configure core.keys.provider=file (or KMS/Vault) or set RISK_ALLOW_ENV_KEYS=1."
            )

    def sign(self, payload: dict[str, Any]) -> Signature:
        mat = self.provider.current()
        digest = hmac.new(mat.key_bytes, _canonical_json(payload), hashlib.sha256).digest()
        return Signature(key_id=mat.key_id, digest_b64=base64.b64encode(digest).decode("ascii"))

    def verify(self, payload: dict[str, Any], sig: Signature | dict[str, Any]) -> bool:
        if sig is None:
            return False
        if isinstance(sig, dict):
            try:
                sig = Signature.from_dict(sig)
            except Exception:
                return False
        if sig is None:
            return False
        try:
            mat = self.provider.get(sig.key_id)
        except KeyProviderError:
            log.warning("KeyManager: signature references unknown key_id %r", sig.key_id)
            return False
        expected = hmac.new(mat.key_bytes, _canonical_json(payload), hashlib.sha256).digest()
        try:
            given = base64.b64decode(sig.digest_b64)
        except Exception:
            return False
        return hmac.compare_digest(expected, given)

    def reload(self) -> None:
        reload = getattr(self.provider, "reload", None)
        if callable(reload):
            reload()
            log.info("KeyManager: reloaded provider, current=%s", self.provider.current().key_id)
        else:
            log.info("KeyManager: provider does not support reload")

    def current_key_id(self) -> str:
        return self.provider.current().key_id
