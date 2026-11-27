"""Overnight swap / rollover financing model for FX, metals, energy and crypto.

Brokers debit (or credit) a "swap" / rollover on positions held across the daily
financing cut. Momentum strategies hold positions for days, so a P&L that ignores
financing overstates net returns. This module models that drag deterministically
from the bar timestamps and the open position, so the same code serves the
vectorized research lab and the paper broker.

Conventions
-----------
* Rates are **basis points of position notional per night**, positive = cost
  (debit), negative = credit. Both sides are modelled independently, so a
  carry-earning side can be expressed with a negative rate.
* The default schedule is deliberately conservative: a symmetric debit per asset
  family. Supply real broker rates via ``config/swap_rates.yaml`` (or a
  ``swap`` block in a config file) for exact numbers.
* Financing is charged once per calendar day at/after ``rollover_hour_utc`` on
  the first bar at or after that hour (falling back to the first bar of the day
  when a feed has no bar in the window). Wednesday is charged ``triple`` by
  default, matching the T+2 weekend convention most FX/CFD brokers use.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # pyyaml ships with the platform; keep a graceful fallback for JSON-only use
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


@dataclass(frozen=True)
class SwapRate:
    """Per-night financing rates in bps of notional (positive = debit)."""

    long_bps_per_day: float = 0.0
    short_bps_per_day: float = 0.0


@dataclass(frozen=True)
class SwapSchedule:
    """Symbol/family financing rates plus rollover conventions."""

    rates: dict[str, SwapRate] = field(default_factory=dict)
    families: dict[str, SwapRate] = field(default_factory=dict)
    default: SwapRate = field(default_factory=SwapRate)
    rollover_hour_utc: int = 21
    triple_swap_weekday: int | None = 2  # Wednesday
    enabled: bool = True

    def rate_for(self, symbol: str) -> SwapRate:
        sym = (symbol or "").upper()
        if sym in self.rates:
            return self.rates[sym]
        return self.families.get(infer_family(sym), self.default)

    def rate_for_side(self, symbol: str, side: float) -> float:
        rate = self.rate_for(symbol)
        return rate.long_bps_per_day if side > 0 else rate.short_bps_per_day

    def multiplier_for(self, ts: Any) -> float:
        if self.triple_swap_weekday is None:
            return 1.0
        try:
            stamp = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")
        except Exception:  # pragma: no cover - defensive
            return 1.0
        return 3.0 if stamp.weekday() == int(self.triple_swap_weekday) else 1.0


def infer_family(symbol: str) -> str:
    """Classify a platform symbol into a financing family."""
    sym = (symbol or "").upper()
    if not sym:
        return "other"
    crypto_bases = (
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB", "LTC", "AVAX",
        "DOT", "LINK", "MATIC", "TRX", "XLM", "ATOM", "UNI", "AAVE",
    )
    if sym.endswith(("USDT", "USDC")) or sym.startswith(crypto_bases):
        return "crypto"
    if sym.startswith(("XAU", "XAG", "GOLD", "SILVER", "XPT", "XPD")):
        return "metal"
    if sym in ("USOIL", "UKOIL", "WTI", "BRENT") or sym.startswith(("OIL", "CL", "NG")):
        return "energy"
    if len(sym) == 6 and sym.isalpha():
        return "fx"
    return "other"


def default_schedule() -> SwapSchedule:
    """Conservative built-in debit schedule (per night, bps of notional)."""
    return SwapSchedule(
        families={
            "fx": SwapRate(0.35, 0.35),
            "metal": SwapRate(0.80, 0.80),
            "energy": SwapRate(1.50, 1.50),
            "crypto": SwapRate(3.00, 3.00),
            "other": SwapRate(0.0, 0.0),
        },
        default=SwapRate(0.0, 0.0),
        rollover_hour_utc=21,
        triple_swap_weekday=2,
        enabled=True,
    )


def _as_rate(value: Any) -> SwapRate:
    if isinstance(value, SwapRate):
        return value
    if isinstance(value, dict):
        long_bps = value.get("long_bps_per_day", value.get("long_bps", value.get("long", 0.0)))
        short_bps = value.get("short_bps_per_day", value.get("short_bps", value.get("short", 0.0)))
        return SwapRate(float(long_bps or 0.0), float(short_bps or 0.0))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return SwapRate(float(value[0]), float(value[1]))
    if isinstance(value, (int, float)):
        return SwapRate(float(value), float(value))
    return SwapRate()


def _as_weekday(value: Any) -> int | None:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("", "none", "off", "false", "no"):
            return None
        return _WEEKDAYS.get(key, _WEEKDAYS.get(key[:3], 2))
    return int(value)


def schedule_from_mapping(raw: dict[str, Any] | None) -> SwapSchedule:
    """Build a schedule from a mapping (accepts a full config or a raw swap block)."""
    if not isinstance(raw, dict):
        return default_schedule()

    data = raw
    # Unwrap config files that nest the schedule under execution.swap / validation.swap.
    for section in ("execution", "validation"):
        block = data.get(section)
        if isinstance(block, dict) and isinstance(block.get("swap"), dict):
            data = block["swap"]
            break
    if isinstance(data.get("swap"), dict):
        data = data["swap"]

    base = default_schedule()
    families = dict(base.families)
    for name, value in (data.get("families") or {}).items():
        families[str(name)] = _as_rate(value)

    rates: dict[str, SwapRate] = {}
    for name, value in (data.get("symbols") or {}).items():
        rates[str(name).upper()] = _as_rate(value)

    default = _as_rate(data["default"]) if "default" in data else base.default
    rollover = int(data.get("rollover_hour_utc", base.rollover_hour_utc)) % 24
    triple = data.get("triple_swap_weekday", base.triple_swap_weekday)

    return SwapSchedule(
        rates=rates,
        families=families,
        default=default,
        rollover_hour_utc=rollover,
        triple_swap_weekday=_as_weekday(triple),
        enabled=bool(data.get("enabled", base.enabled)),
    )


def load_swap_schedule(path: str | Path | None = None) -> SwapSchedule:
    """Load a schedule from a YAML/JSON file (or return the built-in default)."""
    if path is None:
        return default_schedule()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("pyyaml is required to load a non-JSON swap config")
        data = yaml.safe_load(text)
    return schedule_from_mapping(data)


def schedule_from_config(cfg: Any, dotted: str = "execution.swap") -> SwapSchedule:
    """Read a schedule from a ``Config``-like object (or return the default)."""
    if cfg is None:
        return default_schedule()
    raw = cfg.get(dotted) if hasattr(cfg, "get") else None
    return schedule_from_mapping(raw) if raw else default_schedule()


def utc_index(index: Iterable[Any]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def rollover_flags(index: Iterable[Any], rollover_hour_utc: int = 21) -> np.ndarray:
    """Mark the first bar at/after the daily rollover hour (one per calendar day)."""
    idx = utc_index(index)
    n = len(idx)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags

    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0
    days = idx.normalize().to_numpy()

    first_any: dict[Any, int] = {}
    first_after: dict[Any, int] = {}
    for i in range(n):
        day = days[i]
        if day not in first_any:
            first_any[day] = i
        if hours[i] >= rollover_hour_utc and day not in first_after:
            first_after[day] = i

    for day, i in first_any.items():
        flags[first_after.get(day, i)] = True
    return flags


def financing_series(
    index: Iterable[Any],
    position: Iterable[float],
    schedule: SwapSchedule | None,
    symbol: str,
) -> np.ndarray:
    """Per-bar financing cost as a fraction of notional (positive = debit).

    ``position`` is the *held* position at each bar (``simulate`` already lags
    the target signal), so the bar flagged as the daily rollover is charged with
    ``position[i]`` — the exposure actually open across the cut.
    """
    pos = np.asarray(list(position) if not isinstance(position, np.ndarray) else position, dtype=float)
    n = len(pos)
    out = np.zeros(n, dtype=float)
    if schedule is None or not schedule.enabled or n == 0:
        return out

    idx = utc_index(index)
    if len(idx) != n:
        raise ValueError(f"index/position length mismatch: {len(idx)} != {n}")

    flags = rollover_flags(idx, schedule.rollover_hour_utc)

    for i in np.flatnonzero(flags):
        side = pos[i]
        if abs(side) < 1e-12:
            continue
        bps = schedule.rate_for_side(symbol, side)
        if bps == 0.0:
            continue
        out[i] = abs(side) * bps * schedule.multiplier_for(idx[i]) / 1e4
    return out


def summarize_financing(costs: np.ndarray) -> dict[str, Any]:
    """Small audit summary for reports/tests."""
    arr = np.asarray(costs, dtype=float)
    return {
        "total_pct": round(100.0 * float(arr.sum()), 6),
        "nights_charged": int(np.count_nonzero(np.abs(arr) > 0.0)),
        "worst_night_pct": round(100.0 * float(arr.max()), 6) if arr.size else 0.0,
    }
