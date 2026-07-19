"""Chaos: virtual clock moves backwards. TTLs must not spuriously fire."""
import pytest
from execution.broker_adapter import PaperBroker

pytestmark = pytest.mark.chaos


class _C:
    def __init__(self, t): self._t = float(t)
    def now(self): return self._t
    def set(self, t): self._t = float(t)


def test_backwards_clock_does_not_expire_resting_orders(cfg):
    clock = _C(1000.0)
    br = PaperBroker(cfg, clock, starting_balance=10_000.0)
    br.update_price("EURUSD", 1.10)
    res = br.send_limit_order("EURUSD", "buy", limit_price=1.05, notional=100.0, ttl_s=10.0)
    assert res["state"] == "resting"

    clock.set(100.0)
    br.update_price("EURUSD", 1.10)

    order = br.get_resting_order(res["order_id"])
    assert order.state in ("resting", "invalid")
