# core/keys/provider.py
"""
Pluggable signing-key providers.

Backends (Phase 2 ships two; AWS KMS / Vault are stubs to fill in later):
- EnvKeyProvider   - dev / CI only. Reads a base64 key from an env var.
- FileKeyProvider  - production-friendly: JSON keyring on disk, keys base64-encoded.

Every provider returns bytes (the raw HMAC key) plus a key_id the caller
embeds in signed payloads so verification survives rotation.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("core.keys")


class KeyProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyMaterial:
    key_id: str
    key_bytes: bytes


class KeyProvider(ABC):
    """A KeyProvider resolves key_id -> KeyMaterial, and names the current key."""

    @abstractmethod
    def get(self, key_id: str) -> KeyMaterial: ...

    @abstractmethod
    def current(self) -> KeyMaterial: ...

    def list_ids(self) -> list[str]:
        return [self.current().key_id]


class EnvKeyProvider(KeyProvider):
    """
    Reads the HMAC key from an environment variable. Intended for CI and local
    development only. In live mode the platform refuses to use this provider
    unless RISK_ALLOW_ENV_KEYS=1 is also set (see KeyManager).
    """

    def __init__(self, var: str = "RISK_SIGNING_KEY", key_id: str = "env:default") -> None:
        self.var = var
        self._key_id = key_id
        raw = os.environ.get(var)
        if not raw:
            raise KeyProviderError(f"{var} not set")

        # Try base64 first; if it decodes cleanly, validate length and use as-is.
        decoded = None
        try:
            decoded = base64.b64decode(raw, validate=True)
        except Exception:
            pass

        if decoded is not None:
            if len(decoded) < 32:
                raise KeyProviderError(f"{var} must decode to >= 32 bytes (got {len(decoded)})")
            key = decoded
        else:
            # Not base64 — fall back to raw bytes (dev only)
            log.warning("%s is not base64 — using raw bytes (dev only)", var)
            key = raw.encode("utf-8")
            if len(key) < 32:
                key = key + b"\x00" * (32 - len(key))

        self._material = KeyMaterial(key_id, key)

    def get(self, key_id: str) -> KeyMaterial:
        if key_id != self._key_id:
            raise KeyProviderError(f"EnvKeyProvider only knows {self._key_id!r}")
        return self._material

    def current(self) -> KeyMaterial:
        return self._material


class FileKeyProvider(KeyProvider):
    """
    JSON keyring on disk. Format:

        {
          "current": "v3",
          "keys": {
            "v2": "<base64 32+ bytes>",
            "v3": "<base64 32+ bytes>"
          }
        }

    Path is expected to be outside the repo. Read once at startup;
    reload() re-reads for rotation without restart.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise KeyProviderError(f"keyring not found: {self.path}")
        self._cache: dict[str, KeyMaterial] = {}
        self._current_id: str = ""
        self.reload()

    def reload(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise KeyProviderError(f"failed to read keyring: {exc}") from exc

        keys = data.get("keys", {})
        current = data.get("current")
        if not keys or not current:
            raise KeyProviderError("keyring missing 'keys' or 'current'")
        if current not in keys:
            raise KeyProviderError(f"current key_id {current!r} not in keys")

        new_cache: dict[str, KeyMaterial] = {}
        for kid, b64 in keys.items():
            try:
                raw = base64.b64decode(b64)
            except Exception as exc:
                raise KeyProviderError(f"key {kid!r} not base64") from exc
            if len(raw) < 32:
                raise KeyProviderError(f"key {kid!r} must be >= 32 bytes")
            new_cache[kid] = KeyMaterial(kid, raw)

        self._cache = new_cache
        self._current_id = current
        log.info("FileKeyProvider: loaded %d key(s), current=%s", len(new_cache), current)

    def get(self, key_id: str) -> KeyMaterial:
        if key_id not in self._cache:
            raise KeyProviderError(f"unknown key_id {key_id!r}")
        return self._cache[key_id]

    def current(self) -> KeyMaterial:
        return self._cache[self._current_id]

    def list_ids(self) -> list[str]:
        return list(self._cache.keys())


def build_key_provider(cfg: dict | None) -> KeyProvider:
    """
    Config shape:

        core:
          keys:
            provider: env           # env | file
            env_var: RISK_SIGNING_KEY
            keyring_path: /etc/cryptonaute/keyring.json
    """
    cfg = cfg or {}
    provider = (cfg.get("provider") or "env").lower()

    if provider == "env":
        return EnvKeyProvider(var=cfg.get("env_var", "RISK_SIGNING_KEY"),
                              key_id=cfg.get("key_id", "env:default"))
    if provider == "file":
        path = cfg.get("keyring_path")
        if not path:
            raise KeyProviderError("provider=file requires keyring_path")
        return FileKeyProvider(path)
    raise KeyProviderError(f"unknown key provider: {provider!r}")
