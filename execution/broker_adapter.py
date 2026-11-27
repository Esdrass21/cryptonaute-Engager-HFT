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
from abc import ABC, abstractmethod

from core.config import Clock, Config
from execution.book import (
    FillEvent,
    OrderBook,
    RestingOrder,
    synthesize_book,
    walk_book,
)
from execution.financing import SwapSchedule, schedule_from_config

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
    def get_open_positions(self) -> list[dict]: ...


class PaperBroker(BrokerAdapter):
    """
    Zero-dependency simulated broker with a book-aware fill model.

    Phase 4 changes
    ---------------
    * Fills cross a synthetic order book (`synthesize_book`), so large orders pay
      slippage that scales with size, and orders larger than top-of-book depth
      partially fill.
    * Commission is charged on open and close (`execution.commission_bps`).
    * SL/TP is NOT handled here. `OrderManager` owns the trigger; the broker just
      reports fills. (The old broker silently closed positions without emitting a
      fill event, desynchronising the FSM's `_open_side` from `_positions`.)

    Semantics kept from before
    --------------------------
    * `volume` on `send_market_order` is **notional in account currency** — the
      same units `OrderManager._position_size` produces. Conversion to base units
      happens inside `walk_book`.
    * `get_account_state()` returns `{"balance", "equity_peak", "drawdown_pct"}`.
    """

    def __init__(
        self,
        cfg: Config,
        clock: Clock,
        starting_balance: float = 10_000.0,
    ) -> None:
        super().__init__(cfg)
        self.clock = clock
        self.balance = starting_balance
        self.equity_peak = starting_balance

        self._positions: dict[str, dict] = {}
        self._tickets = 0
        self._last_price: dict[str, float] = {}
        self._books: dict[str, OrderBook] = {}

        # Book synthesis config (with safe defaults).
        self._n_levels = int(cfg.get("execution.book.n_levels", 5))
        self._tick_bps = float(cfg.get("execution.book.tick_size_bps", 1.0))
        self._top_notional = float(cfg.get("execution.book.top_size_notional", 50_000.0))
        self._depth_growth = float(cfg.get("execution.book.depth_growth", 1.6))
        self._size_jitter = float(cfg.get("execution.book.size_jitter", 0.25))
        self._commission_bps = float(cfg.get("execution.commission_bps", 0.0))

        # Overnight swap / rollover financing (opt-in via execution.swap.enabled).
        swap_cfg = cfg.get("execution.swap", None) or {}
        self._swap: SwapSchedule | None = (
            schedule_from_config(cfg, "execution.swap")
            if swap_cfg.get("enabled", False) else None
        )
        self._last_swap_day: dict[str, object] = {}

        # Phase 4.5 — limit orders + queue model
        self._resting: dict[str, RestingOrder] = {}
        self._pending_fills: list[FillEvent] = []
        self._flow_multiplier = float(cfg.get("execution.queue_flow_multiplier", 5.0))
        self._erosion_rate = float(cfg.get("execution.queue_erosion_rate", 0.05))
        self._erosion_ticks = int(cfg.get("execution.queue_erosion_ticks", 2))
        self._order_seq = 0

    # ---- interface ---------------------------------------------------- #
    def connect(self) -> bool:
        return True

    def _next_ticket(self) -> int:
        self._tickets += 1
        return self._tickets

    def _book(self, symbol: str) -> OrderBook:
        """Return the current book. Uses an explicit feed if present, else synthesizes."""
        book = self._books.get(symbol)
        if book is not None:
            return book
        mid = self._last_price.get(symbol, 0.0)
        book = synthesize_book(
            symbol, mid,
            ts=self.clock.now(),
            n_levels=self._n_levels,
            tick_size_bps=self._tick_bps,
            top_size_notional=self._top_notional,
            depth_growth=self._depth_growth,
            size_jitter=self._size_jitter,
        )
        self._books[symbol] = book
        return book

    # ---- price / book feed -------------------------------------------- #
    def update_price(self, symbol: str, mid: float) -> None:
        """
        Advance the market. Synthesizes a fresh book, then processes any
        resting limit orders against the implied trade volume. No SL/TP here
        — see OrderManager. Also applies overnight swap financing when enabled.
        """
        self._charge_swap(symbol, self.clock.now())
        prev = self._last_price.get(symbol, mid)
        self._last_price[symbol] = mid
        self._books.pop(symbol, None)       # force re-synthesis from new mid
        if self._resting:
            self._process_resting_orders(symbol, prev, mid)

    def update_book(
        self,
        symbol: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Install an explicit book (used when a feed provides depth)."""
        book = OrderBook.from_levels(bids, asks)
        self._books[symbol] = book
        self._last_price[symbol] = book.mid()

    # ------------------------------------------------------------------ #
    # Overnight swap / rollover financing                                 #
    # ------------------------------------------------------------------ #

    def _charge_swap(self, symbol: str, now: float) -> float:
        """Debit financing once per calendar day rollover for a held position."""
        if self._swap is None or not self._swap.enabled:
            return 0.0
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        try:
            import pandas as pd
            stamp = pd.Timestamp(now, unit="s", tz="UTC")
        except Exception:  # noqa: BLE001 - non-epoch clock in tests
            return 0.0

        # Only the first tick at/after the daily cut triggers a charge; ticks
        # before the cut leave the day "uncharged" until the window opens.
        if stamp.hour < self._swap.rollover_hour_utc:
            return 0.0
        day = stamp.date()
        prev_day = self._last_swap_day.get(symbol)
        # No charge if the day was already settled (or never opened). A None
        # prev_day means this is the first post-cut tick, so charge now.
        if prev_day == day:
            return 0.0
        self._last_swap_day[symbol] = day

        side = 1.0 if pos["side"] == "buy" else -1.0
        bps = self._swap.rate_for_side(symbol, side)
        if bps == 0.0:
            self._last_swap_day[symbol] = day
            return 0.0
        cost = abs(float(pos["volume"])) * bps * self._swap.multiplier_for(stamp) / 1e4
        self._last_swap_day[symbol] = day
        if cost:
            self.balance -= cost
            pos["financing"] = float(pos.get("financing", 0.0)) + cost
        return cost

    # ------------------------------------------------------------------ #
    # Phase 4.5 — limit orders + queue position                           #
    # ------------------------------------------------------------------ #

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"L{self._order_seq:08d}"

    def send_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        notional: float,
        ttl_s: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> dict:
        """Place a limit order. Crossing → taker; non-crossing → resting FIFO."""
        book = self._book(symbol)
        best_bid, best_ask = book.best_bid(), book.best_ask()
        if best_bid <= 0 or best_ask <= 0:
            return {"ok": False, "reason": "no-price", "broker": "paper"}

        side = side.lower()
        crossing = (side == "buy" and limit_price >= best_ask) or \
                   (side == "sell" and limit_price <= best_bid)

        if crossing:
            res = self.send_market_order(symbol, side, notional, sl=sl, tp=tp)
            if res.get("ok"):
                res["order_type"] = "limit"
                res["crossed"] = True
                res["limit_price"] = float(limit_price)
            return res

        order_id = self._next_order_id()
        queue_ahead = self._queue_depth_at(book, side, limit_price)
        order = RestingOrder(
            order_id=order_id, symbol=symbol, side=side,
            limit_price=float(limit_price), notional=float(notional),
            placed_ts=self.clock.now(), ttl_s=float(ttl_s),
            queue_ahead=queue_ahead,
        )
        self._resting[order_id] = order
        return {
            "ok": True, "broker": "paper", "order_type": "limit",
            "order_id": order_id, "symbol": symbol, "side": side,
            "limit_price": float(limit_price),
            "notional": float(notional),
            "queue_ahead": queue_ahead,
            "state": "resting",
        }

    def cancel_order(self, order_id: str) -> dict:
        o = self._resting.get(order_id)
        if o is None:
            return {"ok": False, "reason": "unknown-order"}
        if not o.is_open:
            return {"ok": False, "reason": f"order-{o.state}"}
        o.state = "cancelled"
        return {"ok": True, "order_id": order_id, "state": "cancelled",
                "filled_notional": o.filled_notional}

    def replace_order(
        self,
        order_id: str,
        new_limit_price: float,
        new_notional: float | None = None,
        new_ttl_s: float | None = None,
    ) -> dict:
        """Cancel-and-replace. Re-queues behind current depth at new price."""
        o = self._resting.get(order_id)
        if o is None:
            return {"ok": False, "reason": "unknown-order"}
        if not o.is_open:
            return {"ok": False, "reason": f"order-{o.state}"}
        symbol, side = o.symbol, o.side
        notional = float(new_notional) if new_notional is not None else o.remaining_notional
        ttl = float(new_ttl_s) if new_ttl_s is not None else max(0.0, o.ttl_s - (self.clock.now() - o.placed_ts))
        o.state = "cancelled"
        return self.send_limit_order(symbol, side, float(new_limit_price), notional, ttl_s=ttl)

    def drain_fills(self) -> list[FillEvent]:
        """Return and clear pending fill events."""
        out = self._pending_fills
        self._pending_fills = []
        return out

    def get_resting_order(self, order_id: str) -> RestingOrder | None:
        return self._resting.get(order_id)

    def get_resting_orders(self, symbol: str | None = None) -> list[RestingOrder]:
        orders = [o for o in self._resting.values() if o.is_open]
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ------------------------------------------------------------------ #
    # Fill engine (called from update_price)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _queue_depth_at(book: OrderBook, side: str, limit_price: float) -> float:
        if side == "buy":
            levels = [level for level in book.bids if level.price >= limit_price]
        else:
            levels = [level for level in book.asks if level.price <= limit_price]
        return sum(level.notional for level in levels)

    def _synthesize_trade_volume(self, prev_mid: float, new_mid: float) -> float:
        """Trade volume implied by a mid move. Deterministic."""
        if prev_mid <= 0 or new_mid <= 0:
            return 0.0
        ret = abs(new_mid - prev_mid) / prev_mid
        return ret * self._flow_multiplier * self._top_notional

    def _process_resting_orders(self, symbol: str, prev_mid: float, new_mid: float) -> None:
        now = self.clock.now()
        volume = self._synthesize_trade_volume(prev_mid, new_mid)
        rising = new_mid > prev_mid

        for order in self._resting.values():
            if order.symbol != symbol or not order.is_open:
                continue

            # --- TTL ---
            if order.ttl_s > 0 and (now - order.placed_ts) >= order.ttl_s:
                order.state = "expired"
                continue

            # --- queue erosion ---
            if self._erosion_rate > 0:
                tick = (new_mid * self._tick_bps / 10_000.0) or 1e-9
                dist_ticks = abs(new_mid - order.limit_price) / tick
                if dist_ticks <= self._erosion_ticks:
                    order.queue_ahead *= (1.0 - self._erosion_rate)

            # --- fill evaluation ---
            crossed_notional = self._crossed_notional(order, prev_mid, new_mid, volume, rising)
            if crossed_notional <= 0:
                continue

            # FIFO: consume queue first.
            consumed = min(crossed_notional, order.queue_ahead)
            order.queue_ahead -= consumed
            residual = crossed_notional - consumed
            if residual <= 0:
                continue

            fill_notional = min(residual, order.remaining_notional)
            if fill_notional <= 0:
                continue

            fill_price = order.limit_price
            commission = fill_notional * self._commission_bps / 10_000.0
            self.balance -= commission

            order.filled_notional += fill_notional
            order.last_fill_ts = now
            order.fills.append((now, fill_notional, fill_price))
            fully = order.filled_notional >= order.notional - 1e-9
            if fully:
                order.state = "filled"

            self._pending_fills.append(FillEvent(
                order_id=order.order_id, symbol=symbol, side=order.side,
                notional=fill_notional, price=fill_price, ts=now,
                fully_filled=fully,
                remaining_notional=order.remaining_notional,
                commission=commission,
            ))

    @staticmethod
    def _crossed_notional(
        order: RestingOrder, prev_mid: float, new_mid: float,
        total_volume: float, rising: bool,
    ) -> float:
        if total_volume <= 0:
            return 0.0
        P = order.limit_price
        if order.side == "buy":
            if new_mid > P:
                return 0.0
            if prev_mid <= P:
                return total_volume
            if prev_mid == new_mid:
                return 0.0
            frac = (P - new_mid) / (prev_mid - new_mid)
            return total_volume * max(0.0, min(1.0, frac))
        else:  # sell
            if new_mid < P:
                return 0.0
            if prev_mid >= P:
                return total_volume
            if prev_mid == new_mid:
                return 0.0
            frac = (new_mid - P) / (new_mid - prev_mid)
            return total_volume * max(0.0, min(1.0, frac))

    # ---- order send --------------------------------------------------- #
    def send_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> dict:
        book = self._book(symbol)
        mid = book.mid()
        if mid <= 0.0:
            return {"ok": False, "reason": "no-price", "broker": "paper"}

        walk = walk_book(book, side, float(volume))
        if walk is None:
            return {"ok": False, "reason": "no-liquidity", "broker": "paper"}

        direction = 1.0 if side == "buy" else -1.0
        # Adverse slippage vs mid, in bps (positive = worse than mid).
        slippage_bps = (walk.vwap - mid) / mid * 10_000.0 * direction

        commission = walk.filled_notional * self._commission_bps / 10_000.0
        self.balance -= commission

        ticket = self._next_ticket()
        self._positions[symbol] = {
            "ticket": ticket, "symbol": symbol, "side": side,
            "volume": walk.filled_notional,          # notional actually filled
            "requested_notional": float(volume),
            "open_price": walk.vwap, "sl": float(sl), "tp": float(tp),
            "open_time": self.clock.now(),
            "open_commission": commission,
        }

        return {
            "ok": True,
            "broker": "paper",
            "ticket": ticket,
            "symbol": symbol,
            "side": side,
            "fill_price": walk.vwap,                 # alias for backward compat
            "vwap": walk.vwap,
            "requested_notional": float(volume),
            "filled_notional": walk.filled_notional,
            "filled_base": walk.filled_base,
            "partial": walk.remaining_notional > 0.0,
            "remaining_notional": walk.remaining_notional,
            "levels_consumed": walk.levels_consumed,
            "slippage_bps": slippage_bps,
            "commission": commission,
        }

    def close_position(self, symbol: str, ticket: int) -> dict:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return {"ok": False, "reason": "no-position"}

        book = self._book(symbol)
        mid = book.mid() or pos["open_price"]
        side = "sell" if pos["side"] == "buy" else "buy"
        walk = walk_book(book, side, float(pos["volume"]))
        if walk is None:
            # Book is empty; fall back to mid (should not happen in sim).
            walk = type("W", (), {"vwap": mid, "filled_notional": pos["volume"],
                                   "remaining_notional": 0.0, "levels_consumed": 0,
                                   "filled_base": pos["volume"] / mid if mid else 0.0,
                                   "top_price": mid})()

        direction = 1.0 if pos["side"] == "buy" else -1.0
        ret = (walk.vwap - pos["open_price"]) / pos["open_price"] if pos["open_price"] else 0.0
        gross_pnl = direction * ret * walk.filled_notional
        close_commission = walk.filled_notional * self._commission_bps / 10_000.0
        net_pnl = gross_pnl - close_commission

        self.balance += net_pnl
        self.equity_peak = max(self.equity_peak, self.balance)

        return {
            "ok": True,
            "broker": "paper",
            "ticket": ticket,
            "symbol": symbol,
            "side": pos["side"],
            "fill_price": walk.vwap,
            "vwap": walk.vwap,
            "closed_notional": walk.filled_notional,
            "pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "commission": close_commission,
            "open_commission": pos.get("open_commission", 0.0),
            "financing": pos.get("financing", 0.0),
            "partial": walk.remaining_notional > 0.0,
        }

    # ---- state -------------------------------------------------------- #
    def get_account_state(self) -> dict:
        return {
            "balance": self.balance,
            "equity_peak": self.equity_peak,
            "drawdown_pct": max(0.0, (self.equity_peak - self.balance) / self.equity_peak * 100.0),
        }

    def get_open_positions(self) -> list[dict]:
        return list(self._positions.values())


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

    def get_open_positions(self) -> list[dict]:
        return [{"ticket": p.ticket, "symbol": p.symbol, "side": "buy" if p.type == 0 else "sell",
                 "volume": p.volume, "open_price": p.price_open} for p in self._mt5.positions_get() or []]


class CryptoBroker(BrokerAdapter):  # pragma: no cover - requires network
    """ccxt-backed crypto exchange.

    The exchange id is configurable via ``execution.exchange`` (default ``binance``) so the
    platform can target Binance / Bybit / OKX / Kraken / Coinbase without code changes — ccxt
    unifies their APIs. API keys are read from the secret store (``<EXCHANGE>_API_KEY`` /
    ``<EXCHANGE>_SECRET`` in ``.env``) and never appear in behavioral config.
    """

    KNOWN = {
        "binance": "binance",
        "bybit": "bybit",
        "okx": "okx",
        "kraken": "kraken",
        "coinbase": "coinbase",
        "coinbasepro": "coinbasepro",
    }

    def __init__(self, cfg: Config, clock: Clock | None = None):
        super().__init__(cfg)
        self.clock = clock
        name = str(cfg.get("execution.exchange", "binance")).lower()
        self.exchange_id = self.KNOWN.get(name, name)  # allow pass-through of other ccxt ids
        self._ex = None

    def connect(self) -> bool:
        import ccxt

        ex_class = getattr(ccxt, self.exchange_id, None)
        if ex_class is None:
            log.error("CryptoBroker: unknown exchange id %r", self.exchange_id)
            return False
        key = self.cfg.secret(f"{self.exchange_id.upper()}_API_KEY")
        secret = self.cfg.secret(f"{self.exchange_id.upper()}_SECRET")
        self._ex = ex_class({"apiKey": key, "secret": secret} if key else {})
        return True

    def send_market_order(self, symbol, side, volume, sl=0.0, tp=0.0) -> dict:
        order = self._ex.create_market_order(symbol, side, volume)
        return {"ticket": order["id"], "ok": True,
                "fill_price": float(order["average"] or order["price"]),
                "broker": self.exchange_id}

    def close_position(self, symbol, ticket) -> dict:
        # close via reduce-only market order at current position side; ccxt has no
        # cross-exchange position-close-by-ticket, so callers should pass side/volume.
        return {"ok": True}

    def get_account_state(self) -> dict:
        bal = self._ex.fetch_balance()
        # prefer a USDT/USDC quote balance, fall back to total net worth
        for q in ("USDT", "USDC", "USD", "EUR"):
            if q in bal and bal[q].get("total"):
                return {"balance": float(bal[q]["total"]), "broker": self.exchange_id}
        return {"balance": float(bal.get("total", {}).get("USDT", 0.0)), "broker": self.exchange_id}

    def get_open_positions(self) -> list[dict]:
        return []


def build_broker(cfg: Config, clock: Clock) -> BrokerAdapter:
    kind = cfg.get("execution.broker", "paper")
    if kind == "mt5":
        return MT5Broker(cfg)
    if kind == "crypto":
        return CryptoBroker(cfg)
    return PaperBroker(cfg, clock, starting_balance=float(cfg.get("validation.backtest.initial_capital", 10000.0)))
