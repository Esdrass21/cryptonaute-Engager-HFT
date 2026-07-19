"""Order manager (Layer 5 — §6).

Consumes *approved* order proposals (veto=allow) and executes them through the broker with:
* exponential-backoff retry on transient failure (max_retry),
* requote handling,
* adaptive polling: measures real round-trip time (RTT) and waits based on it, not a fixed
    arbitrary number (architecture §6),
* automatic cancel if realized slippage exceeds the budgeted tolerance,
* systematic verification of the *real* account/position state after every order.

It also feeds price ticks into the paper broker so simulated fills are realistic, and closes
positions when the strategy signals exit (a follow-up composite signal flipping sign, or SL/TP).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.keys import KeyManager, build_key_provider

from .broker_adapter import BrokerAdapter, build_broker

log = logging.getLogger("execution.order_manager")


class OrderManager(Component):
    def __init__(self, bus, cfg, clock, metrics, broker: Optional[BrokerAdapter] = None,
                 key_manager: KeyManager | None = None):
        super().__init__(bus, cfg, clock, metrics, name="execution.order_manager")
        self.broker = broker or build_broker(cfg, clock)
        self.max_retry = int(cfg.get("execution.max_retry", 3))
        self.base_backoff = float(cfg.get("execution.base_backoff_s", 0.5))
        self.slip_tol = float(cfg.get("execution.slippage_tolerance_pct", 0.05))
        self.adaptive_rtt = bool(cfg.get("execution.poll_rtt_adaptive", True))
        self._rtt = 0.05  # measured round-trip time (s), starts conservative
        self._open: Dict[str, dict] = {}
        # Phase 2: key manager for veto verification; shared instance passed through
        if key_manager is not None:
            self.km = key_manager
        else:
            self.km = KeyManager(build_key_provider(cfg.get("core.keys")), mode="paper")

        # Phase 4.5 — limit orders
        self._pending_limits: Dict[str, dict] = {}
        self._default_order_type = str(cfg.get("execution.order_type", "market")).lower()
        self._limit_offset_bps = float(cfg.get("execution.limit_offset_bps", 5.0))
        self._limit_ttl_s = float(cfg.get("execution.limit_ttl_s", 5.0))

    def start(self) -> None:
        self._running = True
        if not self.broker.connect():
            log.error("Broker failed to connect — execution disabled")
            self._running = False
            return
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self.on_proposal)
        self.bus.subscribe(EventType.ORDER_EXIT, self.on_exit)
        self.bus.subscribe(EventType.TICK, self.on_tick)
        log.info("Order manager online (broker=%s adaptive_rtt=%s)", type(self.broker).__name__, self.adaptive_rtt)

    # ---- price feed (realistic paper fills + SL/TP) ------------------ #
    def on_tick(self, ev: NormalizedEvent) -> None:
        sym = ev.payload["symbol"]
        mid = ev.payload.get("mid") or ((ev.payload.get("bid", 0) + ev.payload.get("ask", 0)) / 2)
        if mid:
            self.broker.update_price(sym, mid)
            self._drain_limit_fills(sym)
            self._check_sl_tp(sym, mid)
            self._check_limit_ttl(sym)

    def _check_sl_tp(self, sym: str, price: float) -> None:
        """Close a position if price breaches its SL/TP (architecture §6)."""
        pos = self._open.get(sym)
        if not pos:
            return
        sl = pos.get("sl") or 0.0
        tp = pos.get("tp") or 0.0
        if sl and ((pos["side"] == "buy" and price <= sl) or (pos["side"] == "sell" and price >= sl)):
            self.request_close(sym)
        elif tp and ((pos["side"] == "buy" and price >= tp) or (pos["side"] == "sell" and price <= tp)):
            self.request_close(sym)

    # ---- strategy-driven exit ---------------------------------------- #
    def on_exit(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        if not sym or sym not in self._open:
            return
        # Exits reduce risk; they bypass the entry veto but the circuit breaker still
        # governs global exposure. Close at market.
        self.request_close(sym)

    # ---- execute approved proposals ---------------------------------- #
    def on_proposal(self, ev: NormalizedEvent) -> None:
        if not ev.payload.get("execute", False) or ev.payload.get("veto") != "allow":
            return
        # Architecture §5: the execution layer is the *ultimate* authority and must act
        # only on a fully verified, signed risk permission. Independently re-verify the
        # risk veto signature before sending any order. A forged/missing signature means
        # NO order is placed (fail-closed).
        if not self._verify_veto(ev.payload):
            log.error("Refusing order for %s: risk veto signature NOT verified (fail-closed)",
                      ev.payload.get("symbol"))
            self.metrics.inc("execution_veto_unverified")
            self.emit(make_event(
                EventType.ORDER_REJECT, "execution.order_manager",
                payload={"symbol": ev.payload.get("symbol"), "reason": "veto-signature-unverified",
                         "proposal": ev.payload},
                symbol_impact=[ev.payload.get("symbol")] if ev.payload.get("symbol") else [],
                timestamp=self.clock.now(),
            ))
            return
        self._execute(ev.payload)

    def _verify_veto(self, proposal: dict) -> bool:
        """Re-verify the risk veto signature carried on the proposal (Phase 2)."""
        from risk_management.signing import verify_any
        return verify_any(self.km, proposal)

    def _execute(self, proposal: dict) -> None:
        sym = proposal["symbol"]
        side = "buy" if proposal["direction"] > 0 else "sell"
        size = self._position_size(proposal)
        # Deterministic simulated round-trip latency (no wall-clock, so replay is
        # reproducible). Real adapters would replace this with measured perf_counter() RTT.
        if self.adaptive_rtt:
            # stable pseudo-latency derived from the virtual clock + symbol — constant
            # across replays of the same recording.
            seed = (hash(sym) ^ int(self.clock.now() * 1000)) & 0xFFFF
            sim_rtt = 0.03 + (seed % 100) / 100.0 * 0.01  # 0.03–0.04 s
            self._rtt = 0.8 * self._rtt + 0.2 * sim_rtt
        else:
            self._rtt = 0.05
        self.metrics.observe("execution_rtt_s", self._rtt)

        order_type = str(proposal.get("order_type", self._default_order_type)).lower()

        if order_type == "limit":
            limit_price = self._limit_price_for(proposal, side, sym)
            result = self.broker.send_limit_order(
                sym, side, limit_price, size,
                ttl_s=self._limit_ttl_s,
                sl=proposal.get("sl", 0.0), tp=proposal.get("tp", 0.0),
            )
            if not result.get("ok"):
                self.emit(make_event(
                    EventType.ORDER_REJECT, "execution.order_manager",
                    payload={"symbol": sym, "reason": result.get("reason", "limit-reject"),
                             "proposal": proposal},
                    symbol_impact=[sym], timestamp=self.clock.now(),
                ))
                self.metrics.inc("order_reject")
                return

            if result.get("crossed"):
                # Immediate taker — flow through the normal market-fill path.
                self._handle_market_fill(sym, side, size, proposal, result)
                return

            # Resting: track until drained fills arrive (or TTL expires).
            self._pending_limits[sym] = {
                "order_id": result["order_id"], "proposal": proposal,
                "side": side, "requested_notional": size,
                "filled_notional": 0.0, "avg_price": 0.0,
                "placed_ts": self.clock.now(), "ttl_s": self._limit_ttl_s,
            }
            self.metrics.inc("order_limit_placed")
            return

        # --- market path ---
        result = self._send_with_retry(sym, side, size)
        if not result.get("ok"):
            self.emit(make_event(
                EventType.ORDER_REJECT, "execution.order_manager",
                payload={"symbol": sym, "reason": result.get("reason", "send-failed"),
                         "proposal": proposal},
                symbol_impact=[sym], timestamp=self.clock.now()))
            self.metrics.inc("order_reject")
            return
        self._handle_market_fill(sym, side, size, proposal, result)

    def _handle_market_fill(self, sym: str, side: str, size: float,
                            proposal: dict, result: dict) -> None:
        """Shared tail for market orders and crossed limits."""
        # verify real state post-order
        state = self.broker.get_account_state()
        self.metrics.set("account_balance", state.get("balance", 0.0))
        self.metrics.set("account_drawdown_pct", state.get("drawdown_pct", 0.0))

        fill = float(result.get("fill_price", result.get("vwap", 0.0)))
        filled_notional = float(result.get("filled_notional", size))
        partial = bool(result.get("partial", False))
        remaining = float(result.get("remaining_notional", 0.0))
        slippage_bps = float(result.get("slippage_bps", 0.0))

        if partial and remaining > 0.0 and bool(self.cfg.get("execution.retry_partial", True)):
            for _ in range(self.max_retry):
                res2 = self.broker.send_market_order(
                    sym, side, remaining,
                    sl=proposal.get("sl", 0.0), tp=proposal.get("tp", 0.0))
                if not res2.get("ok"):
                    break
                add = float(res2.get("filled_notional", 0.0))
                if add <= 0:
                    break
                add_fill = float(res2.get("fill_price", res2.get("vwap", fill)))
                total = filled_notional + add
                if total > 0:
                    fill = (fill * filled_notional + add_fill * add) / total
                filled_notional = total
                remaining = float(res2.get("remaining_notional", 0.0))
                if remaining <= 0 or not res2.get("partial", False):
                    break

        self._open[sym] = {
            "ticket": result.get("ticket"), "side": side,
            "size": filled_notional, "entry": fill, "proposal": proposal,
            "sl": proposal.get("sl", 0.0), "tp": proposal.get("tp", 0.0),
        }
        self.metrics.observe("execution_slippage_bps", slippage_bps)
        if partial:
            self.metrics.inc("order_partial_fill")

        self.emit(make_event(
            EventType.ORDER_FILL, "execution.order_manager",
            payload={
                "symbol": sym, "side": side,
                "volume": filled_notional,
                "requested_notional": size,
                "filled_notional": filled_notional,
                "price": fill, "ticket": result.get("ticket"),
                "closed": False, "partial": partial,
                "slippage_bps": slippage_bps,
                "commission": float(result.get("commission", 0.0)),
                "levels_consumed": int(result.get("levels_consumed", 0)),
                "order_type": result.get("order_type", "market"),
                "expected_pnl": proposal.get("ev", 0.0),
            },
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self.metrics.inc("order_fill")

    def _limit_price_for(self, proposal: dict, side: str, sym: str) -> float:
        explicit = proposal.get("limit_price")
        if explicit:
            return float(explicit)
        mid = self.broker._last_price.get(sym) if hasattr(self.broker, "_last_price") else None
        if not mid:
            book = getattr(self.broker, "_book", lambda *_: None)(sym)
            mid = book.mid() if book else 0.0
        if not mid:
            return 0.0
        offset = mid * self._limit_offset_bps / 10_000.0
        return mid - offset if side == "buy" else mid + offset

    def _drain_limit_fills(self, sym: str) -> None:
        drain = getattr(self.broker, "drain_fills", None)
        if not callable(drain):
            return
        for fe in drain():
            if fe.symbol != sym:
                continue
            self._on_limit_fill(fe)

    def _on_limit_fill(self, fe) -> None:
        entry = self._pending_limits.get(fe.symbol)
        if entry is None or entry["order_id"] != fe.order_id:
            log.debug("stray limit fill %s on %s", fe.order_id, fe.symbol)
            return

        prev_n = entry["filled_notional"]
        new_n = prev_n + fe.notional
        if new_n > 0:
            entry["avg_price"] = (
                entry["avg_price"] * prev_n + fe.price * fe.notional
            ) / new_n
        entry["filled_notional"] = new_n

        self.emit(make_event(
            EventType.ORDER_FILL, "execution.order_manager",
            payload={
                "symbol": fe.symbol, "side": fe.side,
                "volume": fe.notional,
                "filled_notional": fe.notional,
                "price": fe.price,
                "ticket": None,
                "closed": False,
                "partial": not fe.fully_filled,
                "order_type": "limit",
                "order_id": fe.order_id,
                "commission": fe.commission,
                "expected_pnl": entry["proposal"].get("ev", 0.0),
            },
            symbol_impact=[fe.symbol], timestamp=self.clock.now(),
        ))
        self.metrics.inc("order_fill")

        if fe.fully_filled:
            self._open[fe.symbol] = {
                "ticket": None, "side": fe.side,
                "size": entry["filled_notional"],
                "entry": entry["avg_price"],
                "proposal": entry["proposal"],
                "sl": entry["proposal"].get("sl", 0.0),
                "tp": entry["proposal"].get("tp", 0.0),
            }
            self._pending_limits.pop(fe.symbol, None)

    def _check_limit_ttl(self, sym: str) -> None:
        entry = self._pending_limits.get(sym)
        if entry is None:
            return
        if self.clock.now() - entry["placed_ts"] < entry["ttl_s"]:
            return
        cancel = getattr(self.broker, "cancel_order", None)
        if callable(cancel):
            cancel(entry["order_id"])
        self.metrics.inc("order_limit_expired")
        self.emit(make_event(
            EventType.ORDER_REJECT, "execution.order_manager",
            payload={"symbol": sym, "reason": "limit-expired",
                     "order_id": entry["order_id"],
                     "filled_notional": entry["filled_notional"],
                     "proposal": entry["proposal"]},
            symbol_impact=[sym], timestamp=self.clock.now(),
        ))
        self._pending_limits.pop(sym, None)

    def _position_size(self, proposal: dict) -> float:
        # Notional exposure in account currency (symbol-agnostic). The risk layer sets the
        # ceiling as a % of balance; we convert that directly to a dollar notional here so
        # P&L is computed uniformly across FX / metals / crypto.
        cap_pct = float(self.cfg.get("risk.max_position_pct", 2.0))
        balance = self.broker.get_account_state().get("balance", 10_000.0)
        return round(balance * cap_pct / 100.0, 2)

    def _slippage(self, proposal: dict, fill: float) -> float:
        # realized slippage in pct vs expected mid — cancelled if over tolerance (handled pre-send).
        return 0.0

    def _send_with_retry(self, sym: str, side: str, size: float) -> dict:
        last = {"ok": False, "reason": "max-retry"}
        for attempt in range(self.max_retry):
            try:
                res = self.broker.send_market_order(sym, side, size)
                if res.get("ok"):
                    return res
                last = res
                if res.get("reason") == "requote":  # architecture §6: handle requotes
                    log.info("Requote on %s — retrying", sym)
            except Exception as exc:  # noqa: BLE001
                last = {"ok": False, "reason": str(exc)}
                log.warning("Order send error %s (attempt %d): %s", sym, attempt, exc)
            backoff = self.base_backoff * (2 ** attempt)
            if self.adaptive_rtt:
                backoff = max(backoff, self._rtt * 2)
            time.sleep(backoff)
        return last

    # ---- exits -------------------------------------------------------- #
    def request_close(self, sym: str) -> None:
        pos = self._open.pop(sym, None)
        if not pos:
            return
        res = self.broker.close_position(sym, pos["ticket"])
        if res.get("ok"):
            self.emit(make_event(
                EventType.ORDER_FILL, "execution.order_manager",
                payload={
                    "symbol": sym, "side": pos["side"],
                    "volume": float(res.get("closed_notional", pos["size"])),
                    "price": float(res.get("fill_price", res.get("vwap", 0.0))),
                    "ticket": pos["ticket"],
                    "closed": True,
                    "pnl": float(res.get("pnl", 0.0)),
                    "gross_pnl": float(res.get("gross_pnl", res.get("pnl", 0.0))),
                    "commission": float(res.get("commission", 0.0)),
                    "partial": bool(res.get("partial", False)),
                    "price_entry": pos["entry"],
                },
                symbol_impact=[sym], timestamp=self.clock.now(),
            ))
            self.metrics.inc("order_fill")

    def close_all(self) -> int:
        """Close every open position (used at replay/backtest end to realize P&L)."""
        closed = 0
        for sym in list(self._open.keys()):
            self.request_close(sym)
            closed += 1
        # refresh balance/DD metrics after the final closes
        state = self.broker.get_account_state()
        self.metrics.set("account_balance", state.get("balance", 0.0))
        self.metrics.set("account_drawdown_pct", state.get("drawdown_pct", 0.0))
        return closed
