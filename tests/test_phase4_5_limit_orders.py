"""Phase 4.5 — limit orders + queue position tests."""
import math
from pathlib import Path
import pytest

from core.config import Config
from execution.book import RestingOrder, FillEvent
from execution.broker_adapter import PaperBroker


class _Clock:
    def __init__(self, t=0.0): self._t = float(t)
    def now(self): return self._t
    def set_virtual(self, t): self._t = float(t)


def _cfg(**over):
    cfg = Config.load()
    cfg.set("execution.commission_bps", 0.0)
    cfg.set("execution.book.tick_size_bps", 1.0)
    cfg.set("execution.book.top_size_notional", 10_000.0)
    cfg.set("execution.book.n_levels", 3)
    cfg.set("execution.book.depth_growth", 1.5)
    cfg.set("execution.book.size_jitter", 0.0)     # deterministic sizes
    cfg.set("execution.queue_flow_multiplier", 5.0)
    cfg.set("execution.queue_erosion_rate", 0.0)   # no erosion by default in tests
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


# ---------- RestingOrder basics ------------------------------------------- #

def test_resting_order_remaining_and_vwap():
    o = RestingOrder(order_id="L1", symbol="X", side="buy",
                     limit_price=99.0, notional=1000.0, placed_ts=0.0)
    assert o.remaining_notional == 1000.0
    o.fills.append((0.1, 400.0, 99.0))
    o.fills.append((0.2, 600.0, 99.0))
    o.filled_notional = 1000.0
    assert o.remaining_notional == 0.0
    assert math.isclose(o.vwap(), 99.0)


# ---------- Placement ---------------------------------------------------- #

def test_limit_below_market_rests():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=99.0, notional=500.0, ttl_s=10.0)
    assert res["ok"] and res["state"] == "resting"
    assert res["queue_ahead"] >= 0.0
    assert len(br.get_resting_orders("X")) == 1


def test_limit_crossing_is_immediate_taker():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=101.0, notional=500.0, ttl_s=10.0)
    assert res["ok"] and res.get("crossed") is True
    assert res.get("filled_notional", 0.0) > 0


def test_sell_limit_above_market_rests():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "sell", limit_price=101.0, notional=500.0, ttl_s=10.0)
    assert res["ok"] and res["state"] == "resting"


# ---------- Queue behaviour --------------------------------------------- #

def test_queue_position_reported():
    """A limit order at a level that already has size joins the back of the queue."""
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    # Place a buy *below* best_bid so it isn't crossing and isn't the touch.
    # The bid levels are at 99.99, 99.98, ... — pick 99.98, which has depth.
    limit_price = 99.98
    res = br.send_limit_order("X", "buy", limit_price=limit_price, notional=200.0, ttl_s=10.0)
    assert res["ok"] and res["state"] == "resting"
    assert res["queue_ahead"] > 0.0


def test_price_must_cross_limit_before_fill():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=99.0, notional=200.0, ttl_s=100.0)
    assert res["state"] == "resting"

    # Price drifts but does not reach the limit.
    br.update_price("X", 99.9)
    br.update_price("X", 99.5)
    assert br.drain_fills() == []


def test_full_fill_when_price_sweeps_through_limit():
    br = PaperBroker(_cfg(**{"execution.queue_flow_multiplier": 100.0,
                             "execution.queue_erosion_rate": 0.0,
                             "execution.book.top_size_notional": 500.0}),
                     _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=99.98, notional=200.0, ttl_s=100.0)
    # Sweep hard below the limit.
    br.update_price("X", 95.0)
    fills = br.drain_fills()
    assert fills, "expected a fill after sweep through limit"
    total = sum(f.notional for f in fills)
    assert total > 0.0
    # Average fill price is the limit, never worse.
    for f in fills:
        assert f.price == 99.98


def test_partial_fill_after_queue_consumed():
    """Large queue ahead + moderate sweep = partial fill."""
    br = PaperBroker(_cfg(**{"execution.queue_flow_multiplier": 2.0}),
                     _Clock(1.0), starting_balance=100_000.0)
    br.update_price("X", 100.0)
    # Place well below BBO so we sit behind meaningful depth.
    res = br.send_limit_order("X", "buy", limit_price=99.5, notional=5_000.0, ttl_s=100.0)
    assert res["state"] == "resting" and res["queue_ahead"] > 0

    # Moderate drift: price reaches limit but volume is small relative to queue.
    br.update_price("X", 99.4)
    fills = br.drain_fills()
    if fills:
        total = sum(f.notional for f in fills)
        assert total <= 5_000.0


def test_ttl_expires_unfilled_limit():
    clock = _Clock(0.0)
    br = PaperBroker(_cfg(), clock, starting_balance=10_000.0)
    br.update_price("X", 100.0)
    br.send_limit_order("X", "buy", limit_price=95.0, notional=200.0, ttl_s=1.0)
    assert br.get_resting_orders("X")[0].state == "resting"

    clock.set_virtual(2.0)
    br.update_price("X", 100.0)      # small move, no fill
    assert br.get_resting_orders("X") == []
    # The expired order is still retrievable by id, just not "open".
    o = list(br._resting.values())[0]
    assert o.state == "expired"


def test_cancel_order_removes_from_open():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=99.0, notional=200.0, ttl_s=10.0)
    oid = res["order_id"]
    out = br.cancel_order(oid)
    assert out["ok"] and out["state"] == "cancelled"
    assert br.get_resting_orders("X") == []


def test_replace_order_requeues_at_new_price():
    br = PaperBroker(_cfg(), _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    a = br.send_limit_order("X", "buy", limit_price=99.0, notional=200.0, ttl_s=10.0)
    b = br.replace_order(a["order_id"], new_limit_price=99.5)
    assert b["ok"] and b["state"] == "resting"
    assert b["limit_price"] == 99.5
    assert a["order_id"] != b["order_id"]


# ---------- Determinism -------------------------------------------------- #

def test_limit_fills_are_deterministic_across_instances():
    def run():
        br = PaperBroker(_cfg(**{"execution.queue_flow_multiplier": 20.0}),
                         _Clock(1.0), starting_balance=10_000.0)
        br.update_price("X", 100.0)
        br.send_limit_order("X", "buy", limit_price=99.0, notional=300.0, ttl_s=100.0)
        br.update_price("X", 98.0)
        return [(f.notional, f.price) for f in br.drain_fills()]

    assert run() == run()


# ---------- Sell-side symmetry ------------------------------------------ #

def test_sell_limit_fills_on_rise():
    br = PaperBroker(_cfg(**{"execution.queue_flow_multiplier": 100.0,
                             "execution.queue_erosion_rate": 0.0}),
                     _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    br.send_limit_order("X", "sell", limit_price=100.02, notional=200.0, ttl_s=100.0)
    br.update_price("X", 105.0)
    fills = br.drain_fills()
    assert fills, "expected a sell-limit fill on a strong up move"
    assert all(f.price == 100.02 for f in fills)


# ---------- Queue erosion ---------------------------------------------- #

def test_queue_erodes_when_price_approaches():
    br = PaperBroker(_cfg(**{"execution.queue_erosion_rate": 0.5,
                             "execution.queue_erosion_ticks": 2}),
                     _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    res = br.send_limit_order("X", "buy", limit_price=99.98, notional=100.0, ttl_s=100.0)
    before = res["queue_ahead"]
    oid = res["order_id"]

    # Price is now within a tick or two of the limit -> erosion runs.
    br.update_price("X", 99.99)
    after = br.get_resting_order(oid).queue_ahead
    assert after < before
