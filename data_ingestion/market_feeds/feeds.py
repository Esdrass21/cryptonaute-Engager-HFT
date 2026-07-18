"""Market data feeds (Layer 1).

Backends selected by ``cfg.data.market.source``: ``sim`` (deterministic GBM, default),
``mt5`` (live MetaTrader5), ``crypto`` (ccxt WebSocket). Each emits TICK / CANDLE / ORDER_BOOK.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from typing import Dict, List, Optional

import numpy as np

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent
from core.metrics import Metrics

from ..normalizer import normalize_candle, normalize_order_book, normalize_tick

log = logging.getLogger("data.market")


class MarketFeed(Component):
    def __init__(self, bus, cfg, clock, metrics, name="market"):
        super().__init__(bus, cfg, clock, metrics, name)
        self.symbols: List[str] = cfg.get("symbols", ["EURUSD"])
        self.timeframe: str = cfg.get("timeframe", "M5")

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.CONTROL, self._on_control)
        log.info("%s market feed online for %s", self.name, self.symbols)

    def _on_control(self, ev):
        if ev.payload.get("action") == "stop":
            self.stop()


class SimMarketFeed(MarketFeed):
    """Deterministic GBM price generator. Seeded for reproducible replays/tests."""

    def __init__(self, bus, cfg, clock, metrics, seed: int = 42):
        super().__init__(bus, cfg, clock, metrics, name="sim.market")
        self.seed = seed
        self.interval = float(cfg.get("data.market.sim.interval_s", 1.0))
        self.candles_per_tick = int(cfg.get("data.market.sim.candles_per_tick", 5))
        self.trend_strength = float(cfg.get("data.market.sim.trend_strength", 0.0))
        self.mean_reversion = float(cfg.get("data.market.sim.mean_reversion", 0.01))
        self._rng = random.Random(seed)
        self._prices: Dict[str, float] = {}
        self._base_vol: Dict[str, float] = {}
        self._candle_state: Dict[str, Dict[str, float]] = {}
        # persistent trend regime per symbol (fair microstructure: trends exist & persist)
        self._trend: Dict[str, float] = {}
        self._trend_life: Dict[str, int] = {}
        self._anchor: Dict[str, float] = {}  # OU anchor (slowly drifting) per symbol
        self._tick_count = 0
        self._seq = 0
        self._init_prices()

    def _init_prices(self) -> None:
        for sym in self.symbols:
            if sym.startswith("XAU"):
                p, v = 2300.0, 0.0008
            elif sym.startswith("BTC"):
                p, v = 60000.0, 0.002
            else:
                p, v = 1.05, 0.0004
            self._prices[sym] = p
            self._base_vol[sym] = v
            self._candle_state[sym] = {"open": p, "high": p, "low": p, "close": p, "volume": 0.0}
            # OU anchor: price reverts to a slowly-moving anchor, so trends persist locally
            # (capturable by MA-cross / z-score) but prices stay stationary (no explosion).
            self._anchor[sym] = p
            self._trend[sym] = 0.0
            self._trend_life[sym] = 0

    def _refresh_trend(self, sym: str) -> None:
        """Occasionally pick a new anchor drift that persists for many steps.

        Pure GBM (trend_strength == 0) has no edge; a nonzero strength injects realistic,
        persistent trends the momentum / z-score indicators can actually detect. Deterministic
        via the seeded RNG, so replays stay reproducible.
        """
        if self._trend_life[sym] > 0:
            self._trend_life[sym] -= 1
            return
        # new regime: drift in [-strength, +strength], persists 40-150 steps (frequent
        # enough reversals that a trend-follower gets many round-trip opportunities).
        self._trend[sym] = self._rng.uniform(-self.trend_strength, self.trend_strength)
        self._trend_life[sym] = self._rng.randint(40, 150)

    def _sim_tick(self, sym: str) -> float:
        v = self._base_vol[sym]
        self._refresh_trend(sym)
        drift = self._trend[sym]
        # anchor drifts by the persistent regime; price reverts to anchor (OU dynamics)
        kappa = self.mean_reversion  # mean-reversion speed — keeps trends local & prices bounded
        log_p = math.log(self._prices[sym])
        log_a = math.log(self._anchor[sym])
        shock = self._rng.gauss(0.0, v)
        new_log = log_p + kappa * (log_a - log_p) + drift + shock
        self._anchor[sym] = math.exp(log_a + drift)  # anchor follows the regime drift
        p = math.exp(new_log)
        self._prices[sym] = max(p, 1e-6)
        return self._prices[sym]

    def _publish_tick(self, sym: str, price: float) -> None:
        spread = price * self._base_vol[sym] * 0.5
        bid, ask = price - spread, price + spread
        ev = normalize_tick("sim.market", sym, bid=bid, ask=ask, last=price,
                            volume=self._rng.uniform(0.1, 2.0), time=self.clock.now(), sequence=self._seq)
        self.emit(ev)
        self._update_candle(sym, price, ev.payload["volume"])

    def _update_candle(self, sym: str, price: float, vol: float) -> None:
        st = self._candle_state[sym]
        st["high"] = max(st["high"], price)
        st["low"] = min(st["low"], price)
        st["close"] = price
        st["volume"] += vol
        self._tick_count += 1
        if self._tick_count % self.candles_per_tick == 0:
            ev = normalize_candle("sim.market", sym, self.timeframe, open_=st["open"], high=st["high"],
                                  low=st["low"], close=st["close"], volume=st["volume"],
                                  time=self.clock.now(), sequence=self._seq)
            self.emit(ev)
            st.update(open=price, high=price, low=price, close=price, volume=0.0)

    def _publish_book(self, sym: str, mid: float) -> None:
        spread = mid * self._base_vol[sym] * 0.5
        depth = 5
        bids = [[mid - spread * (i + 1), self._rng.uniform(1, 5)] for i in range(depth)]
        asks = [[mid + spread * (i + 1), self._rng.uniform(1, 5)] for i in range(depth)]
        ev = normalize_order_book("sim.market", sym, bids=bids, asks=asks, time=self.clock.now(), sequence=self._seq)
        self.emit(ev)

    def run_once(self) -> None:
        for sym in self.symbols:
            price = self._sim_tick(sym)
            self._publish_tick(sym, price)
            if self.cfg.get("analytics.order_flow.enabled", False):
                self._publish_book(sym, price)
            self._seq += 1

    def run_realtime(self, steps: Optional[int] = None) -> None:
        i = 0
        while self._running and (steps is None or i < steps):
            self.run_once()
            i += 1
            if self.interval > 0 and self._running:
                time.sleep(self.interval)


class MT5MarketFeed(MarketFeed):
    def start(self) -> None:
        super().start()
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                log.warning("MT5 init failed (%s) — feed disabled", mt5.last_error())
                return
            self._mt5 = mt5
            for sym in self.symbols:
                self._mt5.symbol_select(sym, True)
            log.info("MT5 feed connected")
            asyncio.ensure_future(self._pump())
        except Exception as exc:
            log.warning("MT5 feed unavailable: %s", exc)

    async def _pump(self) -> None:
        while self._running:
            for sym in self.symbols:
                t = self._mt5.symbol_info_tick(sym)
                if t:
                    self.emit(normalize_tick("mt5", sym, bid=t.bid, ask=t.ask, last=t.last,
                                             volume=t.volume, time=float(t.time), sequence=self._seq))
                    self._seq += 1
            await asyncio.sleep(self.interval)


class CryptoMarketFeed(MarketFeed):
    """Live crypto market data via ccxt REST polling.

    Exchange is configurable via ``execution.exchange`` (default binance), same registry as
    the broker, so feed + execution stay on the same venue. Public market data (ticker,
    OHLCV, order book) needs NO API key; keys are only required to trade. Polls each symbol
    every ``interval`` seconds and emits TICK / CANDLE / ORDER_BOOK. Network errors degrade
    gracefully so a hiccup never crashes the loop.

    Only symbols in ccxt unified form (e.g. ``BTC/USDT``) are polled; MT5-style symbols
    (``EURUSD``, ``XAUUSD``) are skipped — those belong to the MT5 feed.
    """

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="crypto.market")
        self.interval = float(cfg.get("data.market.crypto.interval_s", 2.0))
        self.exchange_id = str(cfg.get("execution.exchange", "binance")).lower()
        self._seq = 0
        self._ex = None
        # Normalize symbols to ccxt unified form: BTCUSDT -> BTC/USDT (best-effort).
        self._pairs = {s: self._to_ccxt_symbol(s) for s in self.symbols}

    @staticmethod
    def _to_ccxt_symbol(sym: str) -> Optional[str]:
        if "/" in sym:
            return sym
        # Only auto-split crypto-native quote currencies. USD/EUR are ambiguous with
        # forex/metals (EURUSD, XAUUSD), so those must be given in explicit BASE/QUOTE form.
        for quote in ("USDT", "USDC", "BTC"):
            if sym.endswith(quote) and len(sym) > len(quote):
                return f"{sym[:-len(quote)]}/{quote}"
        return None  # not a recognizable crypto pair -> skip

    def start(self) -> None:
        super().start()
        try:
            import ccxt

            klass = getattr(ccxt, self.exchange_id, ccxt.binance)
            self._ex = klass({"enableRateLimit": True})
            active = [s for s, p in self._pairs.items() if p]
            log.info("Crypto feed (%s) live for %s", self.exchange_id, active)
        except Exception as exc:  # noqa: BLE001
            log.warning("Crypto feed unavailable: %s", exc)
            self._ex = None

    def _poll_symbol(self, sym: str, pair: str) -> None:
        try:
            t = self._ex.fetch_ticker(pair)
            bid = float(t.get("bid") or t.get("last") or 0.0)
            ask = float(t.get("ask") or t.get("last") or 0.0)
            last = float(t.get("last") or (bid + ask) / 2)
            vol = float(t.get("baseVolume") or 0.0)
            self.emit(normalize_tick(self.exchange_id, sym, bid=bid, ask=ask, last=last,
                                     volume=vol, time=self.clock.now(), sequence=self._seq))
            ohlcv = self._ex.fetch_ohlcv(pair, timeframe="1m", limit=1)
            if ohlcv:
                o, h, l, c, v = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4], ohlcv[-1][5]
                self.emit(normalize_candle(self.exchange_id, sym, self.timeframe, open_=float(o),
                                           high=float(h), low=float(l), close=float(c),
                                           volume=float(v), time=self.clock.now(), sequence=self._seq))
            if self.cfg.get("analytics.order_flow.enabled", False):
                book = self._ex.fetch_order_book(pair, limit=5)
                self.emit(normalize_order_book(self.exchange_id, sym, bids=book["bids"][:5],
                                               asks=book["asks"][:5], time=self.clock.now(),
                                               sequence=self._seq))
        except Exception as exc:  # noqa: BLE001
            log.warning("Crypto poll failed for %s: %s", pair, exc)

    def run_once(self) -> None:
        if self._ex is None:
            return
        for sym, pair in self._pairs.items():
            if pair:
                self._poll_symbol(sym, pair)
                self._seq += 1

    def run_realtime(self, steps: Optional[int] = None) -> None:
        i = 0
        while self._running and (steps is None or i < steps):
            self.run_once()
            i += 1
            if self.interval > 0 and self._running:
                time.sleep(self.interval)


def build_market_feed(bus: MessageBus, cfg: Config, clock: Clock, metrics: Metrics) -> MarketFeed:
    source = cfg.get("data.market.source", "sim")
    if source == "mt5":
        return MT5MarketFeed(bus, cfg, clock, metrics)
    if source == "crypto":
        return CryptoMarketFeed(bus, cfg, clock, metrics)
    return SimMarketFeed(bus, cfg, clock, metrics)
