"""Configuration + logging + a pluggable clock.

* :class:`Config` loads ``config/*.yaml`` (deep-merged over defaults) and exposes a
  dotted-path getter. Secrets are loaded from ``.env`` via ``python-dotenv`` but kept
  separate from behavioral config, exactly as the architecture's secret policy requires.
* :class:`Clock` abstracts time so the replay harness and tests can run deterministically.
* ``structlog``-style structured logging is configured here once.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

log = logging.getLogger("core.config")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (architecture: config deep-merge)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class _MergingSafeLoader(yaml.SafeLoader):
    """SafeLoader that deep-merges duplicate mapping keys instead of dropping them.

    Hand-written YAML often ends up with two top-level ``strategy:`` (or
    ``validation:``) blocks. PyYAML's default safe loader silently keeps only the
    last one, which silently discards tuned strategy parameters. Merging keeps
    both sections and matches the config deep-merge contract used for files.
    """


def _construct_merging_mapping(loader: yaml.SafeLoader, node, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        if key in mapping and isinstance(mapping[key], dict) and isinstance(value, dict):
            mapping[key] = _deep_merge(mapping[key], value)
        else:
            mapping[key] = value
    return mapping


_MergingSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: _construct_merging_mapping(loader, node, deep=True),
)


class Config:
    """Dotted-path config reader with safe defaults.

    Usage
    -----
    >>> cfg = Config.load("config/default.yaml")
    >>> cfg.get("risk.max_daily_drawdown_pct", 2.0)
    """

    DEFAULTS: dict[str, Any] = {
        "platform": {"name": "cryptonaute-hft", "version": "3.0.0"},
        "mode": "paper",                       # paper | live
        "bus": {"backend": "memory", "redis_url": None},
        "symbols": ["EURUSD", "XAUUSD", "BTCUSDT"],
        "timeframe": "M5",
        "data": {
            "market": {"source": "sim", "mt5_login": None},   # sim | mt5 | crypto
            "news": {"source": "sim", "rss_feeds": []},
        },
        "analytics": {
            "regime_model": "statistical",     # statistical | (future) ml
            "order_flow": {"enabled": False, "validated_only": True},
        },
        "nlp": {
            "backend": "lexicon",              # lexicon | finbert | mistral
            "finbert_model": "ProsusAI/finbert",
        },
        "self_correction": {
            "enabled": True,
            "mistral": {"model": "mistral-large-latest", "timeout_s": 20.0, "backend": "stub"},
        },
        "strategy": {
            "enabled": ["ma_cross", "rsi_mean_reversion"],
            "composite": {"threshold": 0.55},
            "ev": {"min_expected_value": 0.0, "calibrated": False},
        },
        "execution": {
            "broker": "paper",                 # paper | mt5 | crypto
            "max_retry": 3,
            "base_backoff_s": 0.5,
            "slippage_tolerance_pct": 0.05,
            "poll_rtt_adaptive": True,
        },
        "risk": {
            "max_position_pct": 2.0,
            "max_daily_drawdown_pct": 2.0,
            "max_correlation": 0.7,
            "hard_cap_pct": 5.0,              # non-negotiable ceiling (architecture §7.1)
            "var_confidence": 0.95,
            "veto_service": {"enabled": True},
            "circuit_breaker": {"enabled": True},
        },
        "validation": {
            "backtest": {"initial_capital": 10000.0},
            "walk_forward": {"train_pct": 0.6, "test_pct": 0.4, "step": 0.2},
            "monte_carlo": {"n_paths": 1000, "block_size": 10},
        },
        "observability": {
            "prometheus": {"enabled": False, "port": 9100},
            "log_level": "INFO",
        },
        "event_log": {
            "enabled": True,
            "backend": "file",
            "path": "data/events.jsonl",
            "fsync": True,
        },
        "keys": {
            "provider": "env",
            "env_var": "RISK_SIGNING_KEY",
            "key_id": "env:default",
        },
        "audit": {
            "path": "data/audit.jsonl",
            "fsync": True,
        },
        "two_person": {
            "min_approvers": 1,
        },
    }

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, *paths: str | Path) -> Config:
        data = copy.deepcopy(cls.DEFAULTS)
        for p in paths:
            p = Path(p)
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    loaded = yaml.load(fh, Loader=_MergingSafeLoader) or {}
                data = _deep_merge(data, loaded)
                log.info("Config: loaded %s", p)
        return cls(data)

    # ---- accessors ----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def get_any(self, *paths: str, default: Any = None) -> Any:
        """Return the first present dotted path (legacy ``core.*`` vs flat layouts)."""
        for dotted in paths:
            cur: Any = self._data
            found = True
            for part in dotted.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    found = False
                    break
            if found:
                return cur
        return default

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self._data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # ---- secrets (never logged) --------------------------------------
    def secret(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)


@dataclass
class Clock:
    """Abstraction over time so replay/tests are deterministic.

    In live mode ``now()`` returns real time. In replay mode it advances to the
    timestamp of the event currently being processed.
    """

    _virtual: float = field(default=0.0)
    _mode: str = "real"  # real | virtual

    def now(self) -> float:
        return self._virtual if self._mode == "virtual" else time.time()

    def set_virtual(self, ts: float) -> None:
        self._virtual = ts
        self._mode = "virtual"

    def advance(self, delta: float) -> None:
        self._virtual += delta

    @property
    def mode(self) -> str:
        return self._mode


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging once: console + an on-disk ``logs/trading.log``."""
    root = logging.getLogger()
    if root.handlers:
        # already configured (re-init in tests / replays) — don't stack handlers
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    # On-disk audit trail of every run (trade flow, vetoes, errors).
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "trading.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # diskless env — console logging still works
    # Quiet noisy third-party loggers in dev (incl. the FSM state-machine engine,
    # which otherwise INFO-logs every state callback and drowns the trade feed).
    for name in ("urllib3", "asyncio", "apscheduler", "ccxt", "MetaTrader5", "transitions"):
        logging.getLogger(name).setLevel(logging.WARNING)
