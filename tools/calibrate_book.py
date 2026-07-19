# tools/calibrate_book.py
"""
Fit execution.book.* parameters from a recorded L2 feed.

Input format: JSONL, one record per line:
    {"ts": <float>, "symbol": "<str>", "bids": [[px, size], ...],
     "asks": [[px, size], ...]}

Fits:
  tick_size_bps       — median (best_ask - best_bid) / mid * 10_000 / 2
  top_size_notional   — median of best-level notional
  depth_growth        — median of (level_k+1_notional / level_k_notional)
  size_jitter         — median of |ln(actual / expected_smooth)|

Output:
  config/book_overrides.yaml with the fitted values, keyed under
  execution.book.

Usage:
  python -m tools.calibrate_book --input data/l2/eurusd.jsonl --symbol EURUSD \
      --out config/book_overrides.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("tools.calibrate_book")


def _iter_records(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_levels(raw) -> List[Tuple[float, float]]:
    out = []
    for lvl in raw or []:
        if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            try:
                out.append((float(lvl[0]), float(lvl[1])))
            except (TypeError, ValueError):
                continue
    return out


def fit_params(
    records: Iterable[dict],
    symbol: Optional[str] = None,
) -> Dict[str, float]:
    """
    Return {tick_size_bps, top_size_notional, depth_growth, size_jitter, n}.
    Every stat is a median across records, so outliers can't skew the fit.
    """
    tick_bps: List[float] = []
    top_notional: List[float] = []
    growth: List[float] = []
    jitter: List[float] = []

    for rec in records:
        if symbol and rec.get("symbol") != symbol:
            continue
        bids = _to_levels(rec.get("bids"))
        asks = _to_levels(rec.get("asks"))
        if not bids or not asks:
            continue
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        bb, ba = bids[0][0], asks[0][0]
        mid = (bb + ba) / 2.0
        if mid <= 0:
            continue
        spread_bps = (ba - bb) / mid * 10_000.0
        tick_bps.append(spread_bps / 2.0)

        top_notional.append(bb * bids[0][1])
        top_notional.append(ba * asks[0][1])

        for levels in (bids, asks):
            notions = [p * s for p, s in levels[:6]]
            if len(notions) < 3:
                continue
            for k in range(len(notions) - 1):
                if notions[k] > 0:
                    growth.append(notions[k + 1] / notions[k])
            try:
                g = float(np.median([
                    notions[k + 1] / notions[k]
                    for k in range(len(notions) - 1)
                    if notions[k] > 0
                ]))
                if g <= 0:
                    continue
                for k, n in enumerate(notions):
                    expected = notions[0] * (g ** k)
                    if expected > 0 and n > 0:
                        jitter.append(abs(math.log(n / expected)))
            except Exception:  # noqa: BLE001
                pass

    def _med(xs: List[float], default: float) -> float:
        return float(np.median(xs)) if xs else default

    return {
        "tick_size_bps":       round(_med(tick_bps, 1.0), 4),
        "top_size_notional":   round(_med(top_notional, 50_000.0), 2),
        "depth_growth":        round(_med(growth, 1.6), 4),
        "size_jitter":         round(_med(jitter, 0.25), 4),
        "n":                   len(tick_bps),
    }


def write_overrides(
    params: Dict[str, float],
    out: Path,
    symbol: Optional[str] = None,
) -> None:
    import yaml as _yaml
    doc = {"execution": {"book": {
        "tick_size_bps":     params["tick_size_bps"],
        "top_size_notional": params["top_size_notional"],
        "depth_growth":      params["depth_growth"],
        "size_jitter":       params["size_jitter"],
    }}}
    if symbol:
        doc["execution"]["book"]["_calibrated_for"] = symbol
    doc["execution"]["book"]["_n_records"] = params["n"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_yaml.safe_dump(doc, default_flow_style=False, sort_keys=False),
                   encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL of L2 snapshots")
    ap.add_argument("--symbol", default=None,
                    help="If set, only records for this symbol are fitted")
    ap.add_argument("--out", default="config/book_overrides.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    records = list(_iter_records(Path(args.input)))
    if not records:
        raise SystemExit(f"no records in {args.input}")
    log.info("loaded %d records", len(records))

    params = fit_params(records, symbol=args.symbol)
    if params["n"] == 0:
        raise SystemExit("no valid L2 snapshots found")

    write_overrides(params, Path(args.out), symbol=args.symbol)
    print(f"\nFitted on {params['n']} snapshots"
          + (f" for {args.symbol}" if args.symbol else ""))
    print(f"  tick_size_bps      = {params['tick_size_bps']}")
    print(f"  top_size_notional  = {params['top_size_notional']:.0f}")
    print(f"  depth_growth       = {params['depth_growth']}")
    print(f"  size_jitter        = {params['size_jitter']}")
    print(f"\nwrote -> {args.out}")
    print("merge into config/default.yaml under execution.book")


if __name__ == "__main__":
    main()
