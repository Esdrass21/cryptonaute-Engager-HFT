"""One-shot validation of the Binance SIGNED order path (no fill risk).

Places a LIMIT order FAR below market (won't fill) then CANCELS it immediately. This proves
the signed POST /api/v3/order + DELETE /api/v3/order endpoints work with your key — without
ever risking a fill. No position is opened. Safe by construction.

Usage:
    python scripts/validate_binance_order.py            # BTCUSDT, default exchange from config
    python scripts/validate_binance_order.py --symbol ETHUSDT
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# repo-root bootstrap so `import core` works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()  # pull BINANCE_API_KEY / BINANCE_API_SECRET from .env

import ccxt
from core.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETH/BTC",
                    help="trading pair to validate (use a base asset you HOLD, e.g. ETH/BTC)")
    ap.add_argument("--exchange", default=None,
                    help="override execution.exchange (default from config)")
    args = ap.parse_args()

    cfg = Config.load()
    exchange_id = (args.exchange or cfg.get("execution.exchange", "binance")).lower()
    klass = getattr(ccxt, exchange_id, ccxt.binance)
    ex = klass({
        "apiKey": cfg.secret("BINANCE_API_KEY", ""),
        "secret": cfg.secret("BINANCE_API_SECRET", ""),
        "enableRateLimit": True,
    })

    # Resolve the ccxt market key: if it already has a slash use it; otherwise let
    # load_markets() + the symbols list map it (e.g. BTCUSDT -> BTC/USDT).
    raw = args.symbol.upper()
    ex.load_markets()
    if raw in ex.markets:
        sym = raw
    else:
        # naive USDT/USDC suffix split; otherwise fall back to raw and let ccxt error clearly
        if raw.endswith("USDT") or raw.endswith("USDC"):
            cand = f"{raw[:-4]}/{raw[-4:]}"
            sym = cand if cand in ex.markets else raw
        else:
            sym = raw
    mkt = ex.markets.get(sym)
    if not mkt:
        print(f"[ERR] {sym} not tradable on {exchange_id}")
        return 1

    ticker = ex.fetch_ticker(sym)
    last = float(ticker["last"])
    # Limit price 50% below market -> will NOT fill.
    # Some pairs report price precision as 0 (no tick limit); clamp to a safe default.
    price_prec = int(mkt["precision"]["price"])
    if price_prec <= 0:
        price_prec = 8
    amt_prec = int(round(-math.log10(float(mkt["precision"]["amount"]))))  # 1e-05 -> 5
    step = float(mkt["precision"]["amount"])
    min_amt = float(mkt["limits"]["amount"]["min"] or 0.0)
    min_cost = float(mkt["limits"]["cost"]["min"] or 10.0)
    limit_price = round(last * 0.5, price_prec)
    # Target a notional safely above the exchange minimum so the order is ACCEPTED
    # (this validates the signed path) — still far below market, so it will NOT fill.
    target_notional = max(min_cost * 2.0, 12.0)
    qty = target_notional / limit_price
    qty = max(qty, min_amt)
    qty = round(qty / step) * step
    qty = float(f"{qty:.{amt_prec}f}")

    print(f"[2] market last={last:.2f}  ->  limit BUY {qty} {sym} @ {limit_price:.2f} "
          f"(far below market, no fill expected)")
    order = ex.create_order(sym, "limit", "buy", qty, limit_price)
    oid = order["id"]
    print(f"[3] ORDER ACCEPTED  id={oid}  status={order.get('status')}")
    print(f"[4] cancelling immediately (zero fill risk)...")
    cancel = ex.cancel_order(oid, sym)
    print(f"[5] CANCELLED  status={cancel.get('status', 'canceled')}")
    print("\nOK — signed order path validated (place + cancel) with NO position opened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
