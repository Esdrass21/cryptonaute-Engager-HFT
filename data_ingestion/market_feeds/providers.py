"""External market-data providers (Layer 1).

Supported providers (selected by ``data.market.provider``):
  * ``alpaca``        — Alpaca data API (crypto + US stocks + FX), keys ALPACA_API_KEY/SECRET
  * ``alphavantage``  — Alpha Vantage (FX + digital currency + stocks), key ALPHAVANTAGE_API_KEY
  * ``twelvedata``    — Twelve Data time_series (FX, metals, crypto, stocks), key TWELVEDATA_API_KEY
  * ``finazon``       — Finazon time_series (custom publisher/dataset), key FINAZON_API_KEY

Two modes:
  * Live polling  — :class:`ProviderMarketFeed` polls the latest bars and emits TICK/CANDLE.
  * History dump  — ``fetch_bars()`` returns closed candles for ``tools/fetch_history.py``,
    which writes a JSONL replay file usable by ``main.py --tune`` / ``--validate``.

Symbol routing is per-provider via ``data.market.providers.<name>.symbol_map``; unmapped
symbols are skipped (never guessed). All HTTP calls are defensive: a provider hiccup logs
a warning and skips the poll instead of crashing the loop.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

from core.config import Clock, Config
from core.events import NormalizedEvent
from core.metrics import Metrics

from .feeds import MarketFeed
from ..normalizer import normalize_candle, normalize_tick

log = logging.getLogger("data.providers")

_TIMEFRAMES = {"M1": 60.0, "M5": 300.0, "M15": 900.0, "H1": 3600.0}


def _parse_ts(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class ProviderFeed(MarketFeed):
    """Base for REST market-data providers."""

    provider_name = "base"
    DEFAULT_SYMBOL_MAP: Dict[str, Optional[str]] = {}

    def __init__(self, bus, cfg: Config, clock: Clock, metrics: Metrics):
        super().__init__(bus, cfg, clock, metrics, name=f"{self.provider_name}.market")
        self.interval = float(cfg.get("data.market.providers.interval_s", 30.0))
        self._session = requests.Session()
        self._session.headers.update(self._headers())
        self.symbol_map: Dict[str, Optional[str]] = dict(self.DEFAULT_SYMBOL_MAP)
        self.symbol_map.update(
            cfg.get(f"data.market.providers.{self.provider_name}.symbol_map", {}) or {}
        )
        self.extra_params: Dict[str, str] = (
            cfg.get(f"data.market.providers.{self.provider_name}.params", {}) or {}
        )
        self._last_bar_ts: Dict[str, float] = {}

    # ---- provider hooks ------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {}

    def _base_url(self) -> str:
        raise NotImplementedError

    def _timeframe_str(self) -> str:
        raise NotImplementedError

    def _request(self, symbol: str, start: Optional[datetime],
                 end: Optional[datetime], limit: int) -> dict:
        raise NotImplementedError

    def _extract_bars(self, data: dict) -> List[dict]:
        """Return list of {ts, open, high, low, close, volume} dicts (ascending)."""
        raise NotImplementedError

    # ---- shared plumbing ------------------------------------------------
    def map_symbol(self, sym: str) -> Optional[str]:
        mapped = self.symbol_map.get(sym)
        if mapped is None or (isinstance(mapped, str) and mapped.lower() in ("", "none")):
            log.debug("%s: symbol %s not mapped — skipped", self.provider_name, sym)
            return None
        return mapped

    def fetch_bars(self, symbol: str, timeframe: str,
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None,
                   limit: int = 500) -> List[NormalizedEvent]:
        """Fetch closed candles as normalized CANDLE events (ascending by time)."""
        mapped = self.map_symbol(symbol)
        if mapped is None:
            return []
        data = self._request(mapped, start, end, limit)
        bars = self._extract_bars(data)
        events: List[NormalizedEvent] = []
        for bar in bars:
            ev = normalize_candle(self.provider_name, symbol, timeframe,
                                  open_=bar["open"], high=bar["high"], low=bar["low"],
                                  close=bar["close"], volume=bar.get("volume", 0.0),
                                  time=bar["ts"], closed=True, sequence=self._seq)
            events.append(ev)
            self._seq += 1
        return events

    def run_once(self) -> None:
        for sym in self.symbols:
            mapped = self.map_symbol(sym)
            if mapped is None:
                continue
            try:
                events = self.fetch_bars(sym, self.timeframe, limit=3)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s poll failed for %s: %s", self.provider_name, sym, exc)
                self.metrics.inc("provider_poll_failed", provider=self.provider_name)
                continue
            for ev in events:
                ts = ev.timestamp
                if ts <= self._last_bar_ts.get(sym, 0.0):
                    continue
                self._last_bar_ts[sym] = ts
                self.emit(ev)
                close = float(ev.payload["close"])
                self.emit(normalize_tick(self.provider_name, sym, bid=close, ask=close,
                                         last=close, volume=ev.payload.get("volume", 0.0),
                                         time=ts, sequence=self._seq))
                self._seq += 1


# --------------------------------------------------------------------------- #
# Alpaca                                                                       #
# --------------------------------------------------------------------------- #
class AlpacaFeed(ProviderFeed):
    provider_name = "alpaca"
    DEFAULT_SYMBOL_MAP = {"EURUSD": "EUR/USD", "XAUUSD": None, "BTCUSDT": "BTC/USD"}
    _TF = {"M1": "1Min", "M5": "5Min", "M15": "15Min", "H1": "1Hour", "D1": "1Day"}

    def _headers(self) -> Dict[str, str]:
        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_API_SECRET", "")
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _base_url(self) -> str:
        return "https://data.alpaca.markets"

    def _timeframe_str(self) -> str:
        return self._TF.get(self.timeframe, "5Min")

    def _route(self, mapped: str) -> str:
        if "/" in mapped:
            base = mapped.split("/", 1)[0].upper()
            if base in ("BTC", "ETH", "SOL", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT"):
                return "crypto"
        return "fx"

    def _request(self, symbol: str, start: Optional[datetime],
                 end: Optional[datetime], limit: int) -> dict:
        route = self._route(symbol)
        params: Dict[str, str] = {"symbols": symbol, "timeframe": self._timeframe_str(),
                                  "limit": str(min(limit, 10000))}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
        if route == "crypto":
            path = "/v1beta3/crypto/us/bars"
        else:
            path = "/v1beta1/forex/us/bars"
        resp = self._session.get(self._base_url() + path, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _extract_bars(self, data: dict) -> List[dict]:
        bars = data.get("bars", [])
        if isinstance(bars, dict):  # keyed by symbol
            bars = next(iter(bars.values()), [])
        out = []
        for bar in bars:
            ts = _parse_ts(bar.get("t"))
            if ts is None:
                continue
            out.append({"ts": ts, "open": float(bar["o"]), "high": float(bar["h"]),
                        "low": float(bar["l"]), "close": float(bar["c"]),
                        "volume": float(bar.get("v", 0.0) or 0.0)})
        out.sort(key=lambda b: b["ts"])
        return out


# --------------------------------------------------------------------------- #
# Twelve Data                                                                  #
# --------------------------------------------------------------------------- #
class TwelveDataFeed(ProviderFeed):
    provider_name = "twelvedata"
    DEFAULT_SYMBOL_MAP = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD", "BTCUSDT": "BTC/USD"}
    _TF = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "D1": "1day"}

    def _headers(self) -> Dict[str, str]:
        return {}

    def _base_url(self) -> str:
        return "https://api.twelvedata.com/time_series"

    def _timeframe_str(self) -> str:
        return self._TF.get(self.timeframe, "5min")

    def _request(self, symbol: str, start: Optional[datetime],
                 end: Optional[datetime], limit: int) -> dict:
        params = {"symbol": symbol, "interval": self._timeframe_str(),
                  "outputsize": str(min(limit, 5000)),
                  "apikey": os.environ.get("TWELVEDATA_API_KEY", "")}
        if start:
            params["start_date"] = start.strftime("%Y-%m-%d %H:%M:%S")
        if end:
            params["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        params.update(self.extra_params)
        resp = self._session.get(self._base_url(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _extract_bars(self, data: dict) -> List[dict]:
        if str(data.get("status", "ok")).lower() == "error":
            raise RuntimeError(str(data.get("message", data)))
        values = data.get("values", []) or []
        out = []
        for row in values:
            ts = _parse_ts(row.get("datetime"))
            if ts is None:
                continue
            out.append({"ts": ts, "open": float(row["open"]), "high": float(row["high"]),
                        "low": float(row["low"]), "close": float(row["close"]),
                        "volume": float(row.get("volume", 0.0) or 0.0)})
        out.sort(key=lambda b: b["ts"])
        return out


# --------------------------------------------------------------------------- #
# Alpha Vantage                                                                #
# --------------------------------------------------------------------------- #
class AlphaVantageFeed(ProviderFeed):
    provider_name = "alphavantage"
    DEFAULT_SYMBOL_MAP = {"EURUSD": "EUR/USD", "XAUUSD": None, "BTCUSDT": "BTC/USD"}
    _TF = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "60min", "D1": "daily"}

    def _headers(self) -> Dict[str, str]:
        return {}

    def _base_url(self) -> str:
        return "https://www.alphavantage.co/query"

    def _timeframe_str(self) -> str:
        return self._TF.get(self.timeframe, "5min")

    def _route(self, mapped: str) -> str:
        base = mapped.split("/", 1)[0].upper()
        if base in ("BTC", "ETH", "SOL", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT"):
            return "crypto"
        return "fx"

    def _request(self, symbol: str, start: Optional[datetime],
                 end: Optional[datetime], limit: int) -> dict:
        base, quote = symbol.split("/", 1)
        params: Dict[str, str] = {"apikey": os.environ.get("ALPHAVANTAGE_API_KEY", ""),
                                  "outputsize": "full"}
        if self._route(symbol) == "crypto":
            params.update({"function": "DIGITAL_CURRENCY_DAILY",
                           "symbol": base, "market": quote})
        else:
            params.update({"function": "FX_INTRADAY",
                           "from_symbol": base, "to_symbol": quote,
                           "interval": self._timeframe_str()})
        params.update(self.extra_params)
        resp = self._session.get(self._base_url(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _extract_bars(self, data: dict) -> List[dict]:
        note = data.get("Note") or data.get("Error Message") or data.get("Information")
        if note:
            raise RuntimeError(str(note)[:200])
        series = None
        for key, val in data.items():
            if isinstance(val, dict) and "Time Series" in key:
                series = val
                break
        if series is None:
            return []
        out = []
        for ts_str, row in series.items():
            ts = _parse_ts(ts_str)
            if ts is None:
                continue
            def _pick(*names: str) -> Optional[str]:
                for n in names:
                    if n in row:
                        return row[n]
                return None
            o = _pick("1. open", "1a. open (USD)", "1b. open (USD)")
            h = _pick("2. high", "2a. high (USD)", "2b. high (USD)")
            lo = _pick("3. low", "3a. low (USD)", "3b. low (USD)")
            c = _pick("4. close", "4a. close (USD)", "4b. close (USD)")
            v = _pick("5. volume", "5. volume (USD)", "6. market cap (USD)")
            if c is None:
                continue
            out.append({"ts": ts, "open": float(o or c), "high": float(h or c),
                        "low": float(lo or c), "close": float(c),
                        "volume": float(v or 0.0)})
        out.sort(key=lambda b: b["ts"])
        return out


# --------------------------------------------------------------------------- #
# Finazon                                                                      #
# --------------------------------------------------------------------------- #
class FinazonFeed(ProviderFeed):
    provider_name = "finazon"
    DEFAULT_SYMBOL_MAP = {"EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "BTCUSDT": "BTCUSDT"}
    _TF = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h", "D1": "1d"}

    def _headers(self) -> Dict[str, str]:
        return {}

    def _base_url(self) -> str:
        return "https://api.finazon.io/1.0/time_series"

    def _timeframe_str(self) -> str:
        return self._TF.get(self.timeframe, "5m")

    def _request(self, symbol: str, start: Optional[datetime],
                 end: Optional[datetime], limit: int) -> dict:
        params: Dict[str, str] = {
            "symbol": symbol,
            "interval": self._timeframe_str(),
            "outputsize": str(min(limit, 10000)),
            "apikey": os.environ.get("FINAZON_API_KEY", ""),
        }
        if start:
            params["started_at"] = str(int(start.timestamp()))
        if end:
            params["ended_at"] = str(int(end.timestamp()))
        params.update(self.extra_params)
        resp = self._session.get(self._base_url(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _extract_bars(self, data: dict) -> List[dict]:
        rows = data.get("data", []) or []
        out = []
        for row in rows:
            ts = _parse_ts(row.get("t") or row.get("time") or row.get("timestamp"))
            if ts is None:
                continue
            out.append({"ts": ts, "open": float(row["o"]), "high": float(row["h"]),
                        "low": float(row["l"]), "close": float(row["c"]),
                        "volume": float(row.get("v", 0.0) or 0.0)})
        out.sort(key=lambda b: b["ts"])
        return out


_PROVIDERS = {
    "alpaca": AlpacaFeed,
    "alphavantage": AlphaVantageFeed,
    "twelvedata": TwelveDataFeed,
    "finazon": FinazonFeed,
}


def build_provider_feed(name: str, bus, cfg: Config, clock: Clock, metrics: Metrics) -> ProviderFeed:
    klass = _PROVIDERS.get(name.lower())
    if klass is None:
        raise ValueError(f"unknown provider '{name}' (choices: {sorted(_PROVIDERS)})")
    feed = klass(bus, cfg, clock, metrics)
    mapped = [s for s in feed.symbols if feed.map_symbol(s)]
    log.info("%s provider feed ready for %s", feed.name, mapped)
    return feed


def fetch_history(name: str, cfg: Config, symbols: List[str], timeframe: str,
                  start: str, end: Optional[str]) -> List[NormalizedEvent]:
    """Pull full candle history from a provider (for tools/fetch_history.py)."""
    from core.metrics import Metrics

    klass = _PROVIDERS.get(name.lower())
    if klass is None:
        raise ValueError(f"unknown provider '{name}' (choices: {sorted(_PROVIDERS)})")
    clock = Clock()
    feed = klass.__new__(klass)  # bypass bus wiring; history-only use
    MarketFeed.__init__(feed, None, cfg, clock, Metrics(), name=f"{name}.history")  # type: ignore[arg-type]
    feed.interval = 0.0
    feed.symbol_map = dict(klass.DEFAULT_SYMBOL_MAP)
    feed.symbol_map.update(cfg.get(f"data.market.providers.{name}.symbol_map", {}) or {})
    feed.extra_params = cfg.get(f"data.market.providers.{name}.params", {}) or {}
    feed._session = requests.Session()
    feed._session.headers.update(feed._headers())
    feed._last_bar_ts = {}

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
    events: List[NormalizedEvent] = []
    for sym in symbols:
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=30), end_dt)
            batch = feed.fetch_bars(sym, timeframe, start=cursor, end=chunk_end, limit=500)
            for ev in batch:
                events.append(ev)
                # Companion tick: SL/TP and mark-to-market are tick-driven.
                close = float(ev.payload["close"])
                events.append(normalize_tick(name, sym, bid=close, ask=close, last=close,
                                             volume=ev.payload.get("volume", 0.0),
                                             time=ev.timestamp + 0.5))
            if batch:
                last_ts = batch[-1].timestamp
                cursor = datetime.fromtimestamp(last_ts, tz=timezone.utc) + timedelta(seconds=1)
            else:
                cursor = chunk_end
            if len(batch) < 2:
                break
            time.sleep(0.3)  # respect provider rate limits
    return events
