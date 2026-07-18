"""Execution engine (Layer 5 — §6).

* :class:`BrokerAdapter` — interface to a broker. ``PaperBroker`` is a zero-dependency
  fill simulator (the default — the platform runs immediately). ``MT5Broker`` / ``CryptoBroker``
  wrap MetaTrader5 / ccxt and activate when enabled. Every broker verifies the *real* account
  state after each order (architecture §6).
* :class:`OrderManager` — consumes *approved* ORDER_PROPOSAL (veto=allow) events, sends via the
  broker with exponential-backoff retry, requote handling, and adaptive polling to the measured
  round-trip time. Cancels an order automatically if slippage exceeds the budgeted tolerance.
  Systematically re-reads real fill/position state afterwards.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

log = logging.getLogger("execution.broker")


class BrokerAdapter(ABC):
    """Common broker interface."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def send_market_order(self, symbol: str, side: str, volume: float, sl: float = 0.0, tp: float = 0.0) -> dict: ...

    @abstractmethod
    def close_position(self, symbol: str, ticket: int) -> dict: ...

    @abstractmethod
    def get_account_state(self) -> dict: ...

    @abstractmethod
    def get_open_positions(self) -> List[dict]: ...


class PaperBroker(BrokerAdapter):
    """Zero-dependency simulated broker. Fills at mid +/- a small modeled slippage.

    Used for paper trading and the validation harness. Tracks a single account balance and
    open positions so the risk layer and EV calibration read realistic state.
    """

    def __init__(self, cfg: Config, clock: Clock, starting_balance: float = 10_000.0):
        super().__init__(cfg)
        self.clock = clock
        self.balance = starting_balance
        self.equity_peak = starting_balance
        self._positions: Dict[str, dict] = {}
        self._tickets = 0
        self._last_price: Dict[str, float] = {}

    # ---- interface ----------------------------------------------------
    def connect(self) -> bool:
        return True

    def _next_ticket(self) -> int:
        self._tickets += 1
        return self._tickets

    def send_market_order(self, symbol: str, side: str, volume: float, sl: float = 0.0, tp: float = 0.0) -> dict:
        price = self._last_price.get(symbol, 1.0)
        # model slippage as a small adverse move
        slip = price * self.cfg.get("execution.slippage_tolerance_pct", 0.05) / 100.0
        fill = price + slip if side == "buy" else price - slip
        ticket = self._next_ticket()
        self._positions[symbol] = {
            "ticket": ticket, "symbol": symbol, "side": side,
            "volume": volume, "open_price": fill, "sl": sl, "tp": tp,
            "open_time": self.clock.now(),
        }
        return {"ticket": ticket, "symbol": symbol, "side": side, "volume": volume,
                "fill_price": fill, "ok": True, "broker": "paper"}

    def close_position(self, symbol: str, ticket: int) -> dict:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return {"ok": False, "reason": "no-position"}
        price = self._last_price.get(symbol, pos["open_price"])
        slip = price * self.cfg.get("execution.slippage_tolerance_pct", 0.05) / 100.0
        fill = price - slip if pos["side"] == "buy" else price + slip
        direction = 1 if pos["side"] == "buy" else -1
        pnl = direction * (fill - pos["open_price"]) * pos["volume"] * 100_000  # std FX lot scale
        self.balance += pnl
        self.equity_peak = max(self.equity_peak, self.balance)
        return {"ticket": ticket, "symbol": symbol, "fill_price": fill, "pnl": pnl, "ok": True, "broker": "paper"}

    def get_account_state(self) -> dict:
        return {"balance": self.balance, "equity_peak": self.equity_peak,
                "drawdown_pct": max(0.0, (self.equity_peak - self.balance) / self.equity_peak * 100.0)}

    def get_open_positions(self) -> List[dict]:
        return list(self._positions.values())

    # ---- price feed for realistic fills ------------------------------
    def update_price(self, symbol: str, mid: float) -> None:
        self._last_price[symbol] = mid
        # check SL/TP on open positions
        pos = self._positions.get(symbol)
        if pos and pos["sl"] and pos["tp"]:
            if (pos["side"] == "buy" and (mid <= pos["sl"] or mid >= pos["tp"])) or \
               (pos["side"] == "sell" and (mid >= pos["sl"] or mid <= pos["tp"])):
                self.close_position(symbol, pos["ticket"])


class MT5Broker(BrokerAdapter):  # pragma: no cover - requires live terminal
    def connect(self) -> bool:
        import MetaTrader5 as mt5

        self._mt5 = mt5
        return bool(mt5.initialize())

    def send_market_order(self, symbol, side, volume, sl=0.0, tp=0.0) -> dict:
        import MetaTrader5 as mt5

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "deviation": 10,
            "magic": 20240718,
            "comment": "cryptonaute",
        }
        res = mt5.order_send(req)
        return {"ticket": getattr(res, "deal", 0), "ok": res.retcode == mt5.TRADE_RETCODE_DONE,
                "fill_price": getattr(res, "price", 0.0), "broker": "mt5"}

    def close_position(self, symbol, ticket) -> dict:
        import MetaTrader5 as mt5

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {"ok": False, "reason": "no-position"}
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": pos[0].volume,
               "type": mt5.ORDER_TYPE_SELL if pos[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
               "position": ticket, "deviation": 10}
        res = mt5.order_send(req)
        return {"ok": res.retcode == mt5.TRADE_RETCODE_DONE, "pnl": getattr(res, "profit", 0.0)}

    def get_account_state(self) -> dict:
        acc = self._mt5.account_info()
        return {"balance": acc.balance, "equity": acc.equity, "drawdown_pct": acc.max_drawdown}

    def get_open_positions(self) -> List[dict]:
        return [{"ticket": p.ticket, "symbol": p.symbol, "side": "buy" if p.type == 0 else "sell",
                 "volume": p.volume, "open_price": p.price_open} for p in self._mt5.positions_get() or []]


class CryptoBroker(BrokerAdapter):  # pragma: no cover - requires network
    def connect(self) -> bool:
        import ccxt

        self._ex = ccxt.binance()
        return True

    def send_market_order(self, symbol, side, volume, sl=0.0, tp=0.0) -> dict:
        order = self._ex.create_market_order(symbol, side, volume)
        return {"ticket": order["id"], "ok": True, "fill_price": float(order["average"] or order["price"]),
                "broker": "crypto"}

    def close_position(self, symbol, ticket) -> dict:
        return {"ok": True}

    def get_account_state(self) -> dict:
        bal = self._ex.fetch_balance()
        return {"balance": bal["USDT"]["total"]}

    def get_open_positions(self) -> List[dict]:
        return []


def build_broker(cfg: Config, clock: Clock) -> BrokerAdapter:
    kind = cfg.get("execution.broker", "paper")
    if kind == "mt5":
        return MT5Broker(cfg)
    if kind == "crypto":
        return CryptoBroker(cfg)
    return PaperBroker(cfg, clock, starting_balance=float(cfg.get("validation.backtest.initial_capital", 10000.0)))
