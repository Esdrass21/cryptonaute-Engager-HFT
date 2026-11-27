# tools/fetch_history.py
"""Dump real candle history from a market-data provider into a replay JSONL.

The output is directly consumable by the validation stack:
    python -m tools.fetch_history --provider alpaca --symbols BTCUSDT \
        --timeframe M5 --start 2025-01-01 --out data/history/btc_m5.jsonl
    python main.py --tune data/history/btc_m5.jsonl --trials 40

Providers:
  * ``ccxt``          — public crypto exchange data (no key needed, default binance)
  * ``mt5``           — MetaTrader 5 terminal history, all broker symbols incl. the
                        full forex/commodities universe (no key; terminal must run)
  * ``alpaca``        — Alpaca data API (crypto + FX + stocks; ALPACA_API_KEY/SECRET)
  * ``alphavantage``  — Alpha Vantage REST (ALPHAVANTAGE_API_KEY)
  * ``alphavantage_lib`` — RomelTorres/alpha_vantage wrapper (same key)
  * ``twelvedata``    — Twelve Data REST (TWELVEDATA_API_KEY)
  * ``finazon``       — Finazon REST (FINAZON_API_KEY)
  * ``finazon_grpc``  — official finazon-grpc-python client (FINAZON_API_KEY; datasets
                        crypto|forex|us_stocks_essential, paging until --end)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC
from pathlib import Path

from core.config import Config
from data_ingestion.market_feeds.providers import fetch_history

log = logging.getLogger("tools.fetch_history")


def _fetch_ccxt(cfg: Config, symbols: list[str], timeframe: str,
                start_ms: int, end_ms: int) -> list:
    """Public crypto OHLCV via ccxt (no API key)."""
    import ccxt

    exchange_id = str(cfg.get("execution.exchange", "binance")).lower()
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h", "D1": "1d"}.get(timeframe, "5m")

    def to_pair(sym: str) -> str:
        if "/" in sym:
            return sym
        for quote in ("USDT", "USDC"):
            if sym.endswith(quote):
                return f"{sym[:-len(quote)]}/{quote}"
        return sym

    events = []
    from data_ingestion.normalizer import normalize_candle, normalize_tick
    for sym in symbols:
        pair = to_pair(sym)
        cursor = start_ms
        while cursor < end_ms:
            rows = ex.fetch_ohlcv(pair, timeframe=tf, since=cursor, limit=1000)
            if not rows:
                break
            for r in rows:
                ts, o, h, lo, c, v = r
                events.append(normalize_candle(exchange_id, sym, timeframe,
                                               open_=o, high=h, low=lo, close=c,
                                               volume=v, time=ts / 1000.0))
                # One tick per candle: SL/TP checks and broker mark-to-market are
                # tick-driven, so a candles-only recording wedges the FSM.
                events.append(normalize_tick(exchange_id, sym, bid=c, ask=c, last=c,
                                             volume=v, time=ts / 1000.0 + 0.5))
            last = rows[-1][0]
            if last <= cursor:
                break
            cursor = last + 1
    return events


def _fetch_mt5(cfg: Config, symbols: list[str], timeframe: str,
               start_ms: int, end_ms: int) -> list:
    """History from a running MetaTrader 5 terminal (forex/commodities universe)."""
    import MetaTrader5 as mt5

    from data_ingestion.normalizer import normalize_candle, normalize_tick

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1,
              "D1": mt5.TIMEFRAME_D1}
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
    smap = cfg.get("data.market.mt5.symbol_map", {}) or {}
    start_dt = pd_datetime(start_ms)
    end_dt = pd_datetime(end_ms)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    events = []
    try:
        for sym in symbols:
            broker_sym = smap.get(sym, sym)
            rates = mt5.copy_rates_range(broker_sym, tf, start_dt, end_dt)
            if rates is None or len(rates) == 0:
                log.warning("mt5: no rates for %s (%s)", sym, broker_sym)
                continue
            sinfo = mt5.symbol_info(broker_sym)
            point = float(sinfo.point) if (sinfo and sinfo.point) else 0.0
            has_spread = "spread" in rates.dtype.names
            for r in rates:
                ts = float(r["time"])
                o, h = float(r["open"]), float(r["high"])
                lo, c = float(r["low"]), float(r["close"])
                v = float(r["tick_volume"])
                events.append(normalize_candle("mt5", sym, timeframe, open_=o,
                                               high=h, low=lo, close=c, volume=v, time=ts))
                spread_pts = float(r["spread"]) if has_spread else 0.0
                spread = spread_pts * point
                events.append(normalize_tick("mt5", sym, bid=c - spread / 2.0,
                                             ask=c + spread / 2.0, last=c, volume=v,
                                             time=ts + 0.5))
    finally:
        mt5.shutdown()
    events.sort(key=lambda e: e.timestamp)
    return events


def pd_datetime(ms: int):
    from datetime import datetime
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def _fetch_finazon_grpc(cfg: Config, symbols: list[str], timeframe: str,
                        start_ms: int, end_ms: int) -> list:
    """History via the official finazon-grpc-python client (paged)."""
    from finazon_grpc_python.common.errors import FinazonGrpcRequestError
    from finazon_grpc_python.time_series_service import GetTimeSeriesRequest, TimeSeriesService

    from data_ingestion.normalizer import normalize_candle, normalize_tick

    key = os.environ.get("FINAZON_API_KEY", "")
    if not key:
        raise RuntimeError("FINAZON_API_KEY not set — add it to .env")
    dataset = str(cfg.get("data.market.providers.finazon_grpc.dataset", "crypto"))
    interval = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h", "D1": "1d"}.get(timeframe, "5m")
    service = TimeSeriesService(key)
    events = []
    for sym in symbols:
        cursor = start_ms // 1000
        while cursor < end_ms // 1000:
            try:
                resp = service.get_time_series(GetTimeSeriesRequest(
                    ticker=sym, dataset=dataset, interval=interval,
                    page_size=1000, started_at=cursor,
                ))
            except FinazonGrpcRequestError as exc:
                log.warning("finazon_grpc %s: %s", sym, exc.message)
                break
            rows = list(resp.result)
            if not rows:
                break
            for item in rows:
                ts = int(item.timestamp)
                events.append(normalize_candle("finazon_grpc", sym, timeframe,
                                               open_=float(item.open), high=float(item.high),
                                               low=float(item.low), close=float(item.close),
                                               volume=float(item.volume), time=float(ts)))
                events.append(normalize_tick("finazon_grpc", sym, bid=float(item.close),
                                             ask=float(item.close), last=float(item.close),
                                             volume=float(item.volume), time=float(ts) + 0.5))
            last = int(rows[-1].timestamp)
            if last <= cursor:
                break
            cursor = last + 1
    return events


def _fetch_alphavantage_lib(cfg: Config, symbols: list[str], timeframe: str,
                            start_ms: int, end_ms: int) -> list:
    """History via the RomelTorres/alpha_vantage wrapper (FX/daily granularity)."""
    from alpha_vantage.cryptocurrencies import CryptoCurrencies
    from alpha_vantage.foreignexchange import ForeignExchange
    from alpha_vantage.timeseries import TimeSeries

    from data_ingestion.normalizer import normalize_candle

    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set — add it to .env")
    interval = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "60min", "D1": "daily"}.get(timeframe, "5min")
    events = []
    for sym in symbols:
        if "/" not in sym:
            sym = f"{sym[:-3]}/{sym[-3:]}" if sym.endswith(("USD", "EUR")) else sym
        base, quote = sym.split("/", 1)
        try:
            if base in ("BTC", "ETH", "SOL", "DOGE", "LTC", "BCH"):
                data, _ = CryptoCurrencies(key=key, output_format="pandas") \
                    .get_digital_currency_daily(symbol=base, market=quote)
            elif len(base) == 3 and len(quote) == 3 and timeframe == "D1":
                data, _ = ForeignExchange(key=key, output_format="pandas") \
                    .get_currency_exchange_rate_daily(from_currency=base, to_currency=quote)
            else:
                data, _ = TimeSeries(key=key, output_format="pandas") \
                    .get_intraday(symbol=sym, interval=interval, outputsize="full")
        except Exception as exc:  # noqa: BLE001
            log.warning("alphavantage_lib %s: %s", sym, exc)
            continue
        for idx, row in data.iterrows():
            ts = idx.timestamp() if hasattr(idx, "timestamp") else float(idx)
            o = row.iloc[0]
            h = row.iloc[1] if "high" in str(row.index[1]).lower() else row.iloc[0]
            lo = row.iloc[2] if len(row.index) > 2 else o
            c = row.iloc[3] if len(row.index) > 3 else o
            events.append(normalize_candle("alphavantage_lib", sym, timeframe,
                                           open_=float(o), high=float(h), low=float(lo),
                                           close=float(c), volume=0.0, time=float(ts)))
    return events


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ccxt",
                    choices=["ccxt", "mt5", "alpaca", "alphavantage", "alphavantage_lib",
                             "twelvedata", "finazon", "finazon_grpc"])
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15", "H1", "D1"])
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: now)")
    ap.add_argument("--out", default="data/history/history.jsonl")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    default_cfg = Path("config/default.yaml")
    cfg_path = args.config or (str(default_cfg) if default_cfg.exists() else None)
    cfg = Config.load(cfg_path) if cfg_path else Config.load()
    if args.symbols.strip().upper() == "ALL":
        symbols = [s.strip().upper() for s in cfg.get("symbols", []) if s.strip()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    from datetime import datetime
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int((datetime.fromisoformat(args.end).replace(tzinfo=UTC)
                  if args.end else datetime.now(UTC)).timestamp() * 1000)

    if args.provider == "ccxt":
        events = _fetch_ccxt(cfg, symbols, args.timeframe, start_ms, end_ms)
    elif args.provider == "mt5":
        events = _fetch_mt5(cfg, symbols, args.timeframe, start_ms, end_ms)
    elif args.provider == "finazon_grpc":
        events = _fetch_finazon_grpc(cfg, symbols, args.timeframe, start_ms, end_ms)
    elif args.provider == "alphavantage_lib":
        events = _fetch_alphavantage_lib(cfg, symbols, args.timeframe, start_ms, end_ms)
    else:
        events = fetch_history(args.provider, cfg, symbols, args.timeframe,
                               args.start, args.end)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_dict()) + "\n")
    log.info("Wrote %d candles (%s %s) -> %s", len(events), ",".join(symbols), args.timeframe, out)
    return 0 if events else 1


if __name__ == "__main__":
    sys.exit(main())
