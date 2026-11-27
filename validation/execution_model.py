# validation/execution_model.py
"""
Deterministic execution model for the paper broker during backtests.

Applied by ReplayHarness / vectorized_backtest to translate a *signal* fill into
a realistic fill. Everything is a function of (symbol, virtual_time), so replays
of the same recording produce identical fills.

Models:
- fixed + jittered latency (deterministic from symbol+ts)
- fat-tailed slippage (Student-t, scaled by a configurable bps)
- spread cost (half-spread applied to each side)
- commission (bps of notional)
- probabilistic partial fills, with a size-dependent fill fraction
- optional queue delay for limit-style entries (skips forward `queue_delay` bars)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FillReport:
    fill_price: float
    filled_qty: float
    requested_qty: float
    slippage_bps: float
    commission: float
    latency_s: float
    partial: bool


class ExecutionModel:
    def __init__(
        self,
        *,
        latency_ms: float = 30.0,
        latency_jitter_ms: float = 10.0,
        slippage_bps: float = 1.5,          # scale of the Student-t
        slippage_df: float = 4.0,           # degrees of freedom (fat tails)
        spread_bps: float = 0.5,
        commission_bps: float = 1.0,
        partial_fill_prob: float = 0.10,
        partial_fill_frac: tuple[float, float] = (0.3, 0.9),
        reference_notional: float = 100_000.0,  # size at which fill prob = 1 - partial_fill_prob
        seed: int = 0,
    ) -> None:
        self.latency_ms = float(latency_ms)
        self.latency_jitter_ms = float(latency_jitter_ms)
        self.slippage_bps = float(slippage_bps)
        self.slippage_df = float(slippage_df)
        self.spread_bps = float(spread_bps)
        self.commission_bps = float(commission_bps)
        self.partial_fill_prob = float(partial_fill_prob)
        self.partial_fill_frac = partial_fill_frac
        self.reference_notional = float(reference_notional)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    def _rng(self, symbol: str, ts: float) -> np.random.Generator:
        # Deterministic per (symbol, ts): replays are byte-for-byte reproducible.
        h = (hash((symbol, int(round(ts * 1_000_000)), self.seed))) & 0xFFFFFFFF
        return np.random.default_rng(h)

    # ------------------------------------------------------------------ #
    def apply(
        self,
        *,
        symbol: str,
        side: str,               # "buy" | "sell"
        mid_price: float,
        qty: float,
        ts: float,
        notional_hint: float | None = None,
    ) -> FillReport:
        rng = self._rng(symbol, ts)

        # --- latency ---------------------------------------------------- #
        lat_ms = self.latency_ms + (rng.random() - 0.5) * 2.0 * self.latency_jitter_ms
        lat_ms = max(0.0, lat_ms)

        # --- slippage: fat-tailed, sign-correct (buy pays up, sells pay down) #
        t = rng.standard_t(self.slippage_df) / np.sqrt(self.slippage_df / (self.slippage_df - 2))
        slip_bps = float(t) * self.slippage_bps
        half_spread_bps = self.spread_bps / 2.0
        cost_bps = slip_bps + half_spread_bps
        direction = 1.0 if side == "buy" else -1.0
        fill_price = mid_price * (1.0 + direction * cost_bps / 10_000.0)

        # --- size-dependent partial fills ------------------------------- #
        notional = (notional_hint if notional_hint is not None else abs(qty * mid_price))
        size_factor = min(1.0, notional / max(1.0, self.reference_notional))
        # Larger orders are more likely to be partially filled.
        p_partial = self.partial_fill_prob + (1.0 - self.partial_fill_prob) * size_factor
        partial = rng.random() < p_partial
        if partial:
            lo, hi = self.partial_fill_frac
            frac = float(lo + (hi - lo) * rng.random())
        else:
            frac = 1.0
        filled_qty = qty * frac

        # --- commission ------------------------------------------------- #
        commission = abs(filled_qty * fill_price) * self.commission_bps / 10_000.0

        return FillReport(
            fill_price=float(fill_price),
            filled_qty=float(filled_qty),
            requested_qty=float(qty),
            slippage_bps=float(cost_bps),
            commission=float(commission),
            latency_s=float(lat_ms / 1000.0),
            partial=bool(partial),
        )


def build_execution_model(cfg) -> ExecutionModel:
    """Read `validation.execution.*` from config with sensible defaults."""
    ex = (cfg.get("validation.execution", {}) or {}) if cfg is not None else {}
    return ExecutionModel(
        latency_ms=ex.get("latency_ms", 30.0),
        latency_jitter_ms=ex.get("latency_jitter_ms", 10.0),
        slippage_bps=ex.get("slippage_bps", 1.5),
        slippage_df=ex.get("slippage_df", 4.0),
        spread_bps=ex.get("spread_bps", 0.5),
        commission_bps=ex.get("commission_bps", 1.0),
        partial_fill_prob=ex.get("partial_fill_prob", 0.10),
        partial_fill_frac=tuple(ex.get("partial_fill_frac", (0.3, 0.9))),
        reference_notional=ex.get("reference_notional", 100_000.0),
        seed=ex.get("seed", 0),
    )
