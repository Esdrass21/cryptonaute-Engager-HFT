# execution/book.py
"""
Synthetic + explicit order books for the paper broker.

Design
------
* Deterministic: same (symbol, ts, mid) always yields the same ladder, so replays
  are byte-for-byte reproducible.
* Two modes:
    - explicit: `OrderBook.from_levels(bids, asks)` when a feed provides depth.
    - synthesized: `synthesize_book(symbol, mid, ...)` otherwise.
* No queue position yet — that lands with limit-order support (Phase 4.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class DepthLevel:
    price: float
    size: float                       # base units (e.g. BTC, EUR)

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class WalkResult:
    vwap: float
    filled_base: float
    filled_notional: float
    remaining_notional: float
    levels_consumed: int
    top_price: float


class OrderBook:
    """Bid/ask ladder, sorted best-first. bids descending, asks ascending."""

    __slots__ = ("bids", "asks")

    def __init__(self, bids: List[DepthLevel], asks: List[DepthLevel]) -> None:
        self.bids = sorted(bids, key=lambda l: -l.price)
        self.asks = sorted(asks, key=lambda l: l.price)

    # ---- accessors ---------------------------------------------------- #
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    def mid(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb and ba:
            return (bb + ba) / 2.0
        return bb or ba or 0.0

    def spread_bps(self) -> float:
        mid = self.mid()
        if not mid:
            return 0.0
        return (self.best_ask() - self.best_bid()) / mid * 10_000.0

    # ---- construction ------------------------------------------------- #
    @classmethod
    def from_levels(
        cls,
        bids: Iterable[Tuple[float, float]],
        asks: Iterable[Tuple[float, float]],
    ) -> "OrderBook":
        return cls(
            [DepthLevel(p, s) for p, s in bids],
            [DepthLevel(p, s) for p, s in asks],
        )


# --------------------------------------------------------------------------- #
# Synthetic book                                                               #
# --------------------------------------------------------------------------- #

def _seed(symbol: str, ts: float, salt: int = 0) -> int:
    h = hash((symbol, int(round(ts * 1_000_000)), salt)) & 0xFFFFFFFF
    return h


def synthesize_book(
    symbol: str,
    mid: float,
    *,
    ts: float,
    n_levels: int = 5,
    tick_size_bps: float = 1.0,
    top_size_notional: float = 50_000.0,
    depth_growth: float = 1.6,
    size_jitter: float = 0.25,
    seed_salt: int = 0,
) -> OrderBook:
    """
    Build a symmetric ladder around `mid`. Deterministic in (symbol, ts, mid).

    - tick_size = mid * tick_size_bps / 10_000
    - half spread = one tick (bid = mid - tick, ask = mid + tick)
    - top-size per level derived from `top_size_notional / price`
    """
    if mid <= 0:
        return OrderBook(bids=[], asks=[])

    tick = mid * tick_size_bps / 10_000.0
    rng = np.random.default_rng(_seed(symbol, ts, seed_salt))

    def _level(price: float, k: int) -> DepthLevel:
        base_notional = top_size_notional * (depth_growth ** k)
        noise = 1.0 + (rng.random() - 0.5) * 2.0 * size_jitter
        notional = max(1.0, base_notional * noise)
        return DepthLevel(price=price, size=notional / price)

    bids = [_level(mid - tick * (k + 1), k) for k in range(n_levels)]
    asks = [_level(mid + tick * (k + 1), k) for k in range(n_levels)]
    return OrderBook(bids=bids, asks=asks)


# --------------------------------------------------------------------------- #
# Book walking                                                                 #
# --------------------------------------------------------------------------- #

def walk_book(
    book: OrderBook,
    side: str,
    notional: float,
) -> Optional[WalkResult]:
    """
    Cross the book to fill `notional` (in account currency) on `side`.

    - buy  crosses asks (ascending)
    - sell crosses bids (descending)

    Returns None if there is no liquidity on the crossing side.
    """
    if notional <= 0:
        return None
    side = side.lower()
    if side == "buy":
        levels = book.asks
    elif side == "sell":
        levels = book.bids
    else:
        raise ValueError(f"unknown side {side!r}")

    if not levels:
        return None

    remaining = float(notional)
    filled_notional = 0.0
    filled_base = 0.0
    consumed = 0
    top_price = levels[0].price
    for lvl in levels:
        if remaining <= 0.0:
            break
        take_notional = min(remaining, lvl.notional)
        take_base = take_notional / lvl.price
        filled_notional += take_notional
        filled_base += take_base
        remaining -= take_notional
        consumed += 1

    if filled_notional <= 0.0:
        return None
    vwap = filled_notional / filled_base if filled_base > 0 else top_price
    return WalkResult(
        vwap=vwap,
        filled_base=filled_base,
        filled_notional=filled_notional,
        remaining_notional=remaining,
        levels_consumed=consumed,
        top_price=top_price,
    )


# --------------------------------------------------------------------------- #
# Phase 4.5 — limit orders + queue position                                  #
# --------------------------------------------------------------------------- #

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RestingOrder:
    """
    A limit order resting in the book, with FIFO queue accounting.
    """
    order_id: str
    symbol: str
    side: str
    limit_price: float
    notional: float
    placed_ts: float
    ttl_s: float = 0.0
    filled_notional: float = 0.0
    queue_ahead: float = 0.0
    state: str = "resting"
    last_fill_ts: float = 0.0
    fills: list = field(default_factory=list)

    @property
    def remaining_notional(self) -> float:
        return max(0.0, self.notional - self.filled_notional)

    @property
    def is_open(self) -> bool:
        return self.state == "resting"

    def vwap(self) -> float:
        if not self.fills:
            return self.limit_price
        total = sum(n for _, n, _ in self.fills)
        if total <= 0:
            return self.limit_price
        return sum(n * p for _, n, p in self.fills) / total


@dataclass(frozen=True)
class FillEvent:
    """Emitted by the broker whenever a resting order fills (partially or fully)."""
    order_id: str
    symbol: str
    side: str
    notional: float
    price: float
    ts: float
    fully_filled: bool
    remaining_notional: float
    commission: float = 0.0
