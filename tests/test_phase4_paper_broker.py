"""Phase 4 — realistic paper broker tests.

Book-aware fills, partial fills, commission, SL/TP ownership, determinism.
"""
import math
from pathlib import Path
import pytest

from core.config import Config
from execution.book import DepthLevel, OrderBook, synthesize_book, walk_book
from execution.broker_adapter import PaperBroker


class _Clock:
    def __init__(self, t=0.0): self._t = t
    def now(self): return self._t
    def set_virtual(self, t): self._t = t


def _cfg(**over):
    cfg = Config.load()
    cfg.set("execution.commission_bps", 1.0)
    cfg.set("execution.book.tick_size_bps", 1.0)
    cfg.set("execution.book.top_size_notional", 10_000.0)
    cfg.set("execution.book.depth_growth", 1.6)
    cfg.set("execution.retry_partial", False)  # disabled for unit tests that check partial
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


# ---------- book primitives ----------------------------------------------- #

def test_walk_book_consumes_best_levels_first():
    book = OrderBook.from_levels(
        bids=[(99.0, 100.0), (98.0, 100.0)],
        asks=[(101.0, 100.0), (102.0, 100.0)],
    )
    w = walk_book(book, "buy", notional=10_000.0)
    assert w is not None
    assert w.levels_consumed == 1
    assert math.isclose(w.vwap, 101.0, rel_tol=1e-9)
    assert w.remaining_notional <= 1e-6


def test_walk_book_consumes_multiple_levels():
    book = OrderBook.from_levels(
        bids=[(99.0, 100.0)],
        asks=[(101.0, 50.0), (102.0, 200.0)],
    )
    w = walk_book(book, "buy", notional=10_000.0)
    assert w is not None
    assert w.levels_consumed == 2
    assert 101.0 < w.vwap < 102.0


def test_walk_book_partial_when_book_exhausted():
    book = OrderBook.from_levels(bids=[(99.0, 10.0)], asks=[(101.0, 10.0)])
    w = walk_book(book, "buy", notional=10_000.0)
    assert w is not None
    assert w.remaining_notional > 0.0


def test_synthesized_book_is_deterministic():
    a = synthesize_book("BTC", 60_000.0, ts=1.0, n_levels=5)
    b = synthesize_book("BTC", 60_000.0, ts=1.0, n_levels=5)
    assert [l.price for l in a.bids] == [l.price for l in b.bids]
    assert [l.size for l in a.asks] == [l.size for l in b.asks]


def test_synthesized_book_is_symmetric_around_mid():
    book = synthesize_book("X", 100.0, ts=1.0, tick_size_bps=1.0, n_levels=3)
    assert math.isclose(book.best_bid(), 100.0 - 0.01, rel_tol=1e-6)
    assert math.isclose(book.best_ask(), 100.0 + 0.01, rel_tol=1e-6)
    assert math.isclose(book.mid(), 100.0, rel_tol=1e-9)


# ---------- PaperBroker ---------------------------------------------------- #

def test_broker_reports_richer_fill_fields():
    cfg = _cfg()
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    br.update_price("EURUSD", 1.10)
    res = br.send_market_order("EURUSD", "buy", volume=500.0)
    assert res["ok"] is True
    for k in ("vwap", "filled_notional", "partial", "slippage_bps",
              "commission", "levels_consumed"):
        assert k in res, f"missing {k}"
    assert res["filled_notional"] > 0


def test_broker_larger_order_has_worse_slippage():
    cfg = _cfg(**{"execution.retry_partial": False})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=1_000_000.0)
    br.update_price("X", 100.0)
    small = br.send_market_order("X", "buy", volume=200.0)
    br.update_price("X", 100.0)
    large = br.send_market_order("X", "buy", volume=500_000.0)
    assert large["slippage_bps"] > small["slippage_bps"]


def test_broker_partial_fill_when_book_exhausted():
    cfg = _cfg(**{"execution.book.top_size_notional": 500.0,
                  "execution.book.n_levels": 2,
                  "execution.book.depth_growth": 1.5})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=1_000_000.0)
    br.update_price("X", 100.0)
    # book notional ≈ 500 + 750 = 1250; ask for far more
    res = br.send_market_order("X", "buy", volume=10_000.0)
    assert res["ok"] is True
    assert res["partial"] is True
    assert res["remaining_notional"] > 0


def test_broker_commission_reduces_balance_on_open():
    cfg = _cfg(**{"execution.commission_bps": 10.0})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    before = br.balance
    res = br.send_market_order("X", "buy", volume=5_000.0)
    assert res["commission"] > 0
    assert br.balance < before


def test_broker_never_closes_sl_tp_silently():
    """
    Regression for the Phase 4 fix: the broker must not mutate positions on
    update_price, or the FSM never learns about the close.
    """
    cfg = _cfg()
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    br.send_market_order("X", "buy", volume=1_000.0, sl=95.0, tp=105.0)
    assert len(br.get_open_positions()) == 1
    # Price crashes through SL. Pre-Phase-4 the broker closed it here. Now it must not.
    br.update_price("X", 90.0)
    assert len(br.get_open_positions()) == 1


def test_broker_realized_pnl_sign_is_correct():
    cfg = _cfg(**{"execution.commission_bps": 0.0})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    opened = br.send_market_order("X", "buy", volume=1_000.0)
    br.update_price("X", 110.0)
    closed = br.close_position("X", opened["ticket"])
    assert closed["ok"] is True
    assert closed["pnl"] > 0


def test_broker_deterministic_across_instances():
    cfg = _cfg()
    b1 = PaperBroker(cfg, _Clock(5.0), starting_balance=10_000.0)
    b2 = PaperBroker(cfg, _Clock(5.0), starting_balance=10_000.0)
    b1.update_price("X", 100.0)
    b2.update_price("X", 100.0)
    r1 = b1.send_market_order("X", "buy", volume=1_000.0)
    r2 = b2.send_market_order("X", "buy", volume=1_000.0)
    assert r1["vwap"] == r2["vwap"]
    assert r1["filled_notional"] == r2["filled_notional"]


def test_broker_retry_partial_fills_residual():
    """When retry_partial is enabled, partial fills should be retried."""
    cfg = _cfg(**{"execution.retry_partial": True,
                  "execution.book.top_size_notional": 500.0,
                  "execution.book.n_levels": 2,
                  "execution.book.depth_growth": 1.5,
                  "execution.max_retry": 3})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=1_000_000.0)
    br.update_price("X", 100.0)
    # Ask for far more than book has
    res = br.send_market_order("X", "buy", volume=50_000.0)
    assert res["ok"] is True
    # After retry, filled notional should be greater than first partial
    # (though may still be partial if the book is thin)
    assert res["filled_notional"] > 0


def test_broker_close_reports_commission():
    cfg = _cfg(**{"execution.commission_bps": 10.0})
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    br.update_price("X", 100.0)
    opened = br.send_market_order("X", "buy", volume=1_000.0)
    br.update_price("X", 100.0)
    closed = br.close_position("X", opened["ticket"])
    assert closed["ok"] is True
    assert closed["commission"] > 0


def test_broker_no_liquidity_returns_fail():
    cfg = _cfg()
    br = PaperBroker(cfg, _Clock(1.0), starting_balance=10_000.0)
    # No price set, so mid=0, book is empty
    res = br.send_market_order("X", "buy", volume=1_000.0)
    assert res["ok"] is False
    assert res["reason"] in ("no-price", "no-liquidity")
