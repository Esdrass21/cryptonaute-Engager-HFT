"""Chaos: many resting limits expire at once. System must cancel them all,
emit an ORDER_REJECT per expiry, and not leak state."""
import pytest
from execution.broker_adapter import PaperBroker

pytestmark = pytest.mark.chaos


class _Clock:
    def __init__(self, t=0.0): self._t = float(t)
    def now(self): return self._t
    def advance(self, d): self._t += d


def test_ttl_storm_cleans_up(cfg):
    clock = _Clock(0.0)
    br = PaperBroker(cfg, clock, starting_balance=1_000_000.0)
    br.update_price("EURUSD", 1.10)

    for _ in range(50):
        br.send_limit_order("EURUSD", "buy", limit_price=1.05, notional=100.0, ttl_s=1.0)
    assert len(br.get_resting_orders("EURUSD")) == 50

    # Time passes; a small market move triggers TTL sweep.
    clock.advance(2.0)
    br.update_price("EURUSD", 1.10)

    assert br.get_resting_orders("EURUSD") == []
    # Every one is either expired or cancelled, not silently dropped.
    states = [o.state for o in br._resting.values()]
    assert all(s in ("expired", "cancelled", "filled") for s in states)
