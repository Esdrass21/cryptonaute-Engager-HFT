import base64, json
from pathlib import Path
import pytest

from core.config import Config


@pytest.fixture
def keyring(tmp_path: Path):
    p = tmp_path / "keyring.json"
    p.write_text(json.dumps({
        "current": "v1",
        "keys": {"v1": base64.b64encode(b"1" * 32).decode()},
    }))
    return p


@pytest.fixture
def cfg(tmp_path: Path, keyring: Path) -> Config:
    c = Config.load()
    c.set("mode", "paper")
    c.set("core.keys.provider", "file")
    c.set("core.keys.keyring_path", str(keyring))
    c.set("core.audit.path", str(tmp_path / "audit.jsonl"))
    c.set("symbols", ["EURUSD"])
    c.set("execution.order_type", "market")
    return c
