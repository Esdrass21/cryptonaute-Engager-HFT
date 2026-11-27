"""Config loading regressions: duplicate YAML keys and legacy ``core.*`` paths."""
from __future__ import annotations

from pathlib import Path

from core.config import Config
from execution.financing import schedule_from_config

PROJECT = Path(__file__).resolve().parents[1]


def test_duplicate_top_level_keys_are_merged(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text(
        "validation:\n"
        "  backtest: { initial_capital: 1.0 }\n"
        "  gate: { min_dsr: 0.1 }\n"
        "strategy:\n"
        "  enabled: [tsmom]\n"
        "validation:\n"
        "  signal_log: { enabled: true }\n"
        "strategy:\n"
        "  release_drift: { enabled: false }\n",
        encoding="utf-8",
    )
    cfg = Config.load(path)
    assert cfg.get("validation.backtest.initial_capital") == 1.0
    assert cfg.get("validation.gate.min_dsr") == 0.1
    assert cfg.get("validation.signal_log.enabled") is True
    assert cfg.get("strategy.enabled") == ["tsmom"]
    assert cfg.get("strategy.release_drift.enabled") is False


def test_shipped_default_config_keeps_both_sections():
    cfg = Config.load(PROJECT / "config" / "default.yaml")
    assert cfg.get("validation.backtest.initial_capital") is not None
    assert cfg.get("validation.signal_log.enabled") is False
    assert cfg.get("strategy.enabled")
    assert cfg.get("strategy.release_drift.enabled") is True
    sched = schedule_from_config(cfg)
    assert sched.enabled is True
    assert sched.rate_for("EURUSD").long_bps_per_day == 0.35
    assert sched.rate_for("USOIL").long_bps_per_day == 1.5


def test_get_any_prefers_legacy_core_when_present():
    cfg = Config.load()
    cfg.set("core.keys.provider", "file")
    assert cfg.get_any("core.keys", "keys")["provider"] == "file"


def test_get_any_falls_back_to_flat_layout():
    cfg = Config.load()  # DEFAULTS define the flat keys/two_person/audit blocks
    assert cfg.get("core.keys") is None
    assert cfg.get_any("core.keys", "keys")["provider"] == "env"
    assert cfg.get_any("core.two_person", "two_person")["min_approvers"] == 1
    assert cfg.get_any("core.audit", "audit")["path"] == "data/audit.jsonl"


def test_get_any_returns_default_when_missing():
    cfg = Config.load()
    assert cfg.get_any("nope.a", "nope.b", default="fallback") == "fallback"
