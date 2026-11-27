"""Chaos: an empty book. Fills must not occur; system must not crash."""
import pytest

from execution.book import OrderBook, walk_book

pytestmark = pytest.mark.chaos


def test_walk_empty_book_returns_none():
    book = OrderBook(bids=[], asks=[])
    assert walk_book(book, "buy", notional=100.0) is None
    assert walk_book(book, "sell", notional=100.0) is None


def test_paper_broker_survives_empty_book(cfg):
    from execution.broker_adapter import PaperBroker
    class C:
        def now(self): return 1.0
    br = PaperBroker(cfg, C(), starting_balance=1_000.0)
    res = br.send_market_order("EURUSD", "buy", volume=100.0)
    assert res["ok"] is False
    assert res["reason"] == "no-price"
    assert br.get_account_state()["balance"] == 1_000.0
