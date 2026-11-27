# validation/drift_analysis.py
"""
Measure post-release drift on a recorded history.

Input:
  * A JSONL file of events (same format the replay harness consumes).
    Each line is a dict with keys `type`, `ts`, `payload` (or a
    NormalizedEvent with `.to_dict()`).

For every RELEASE_ACTUAL with a `surprise` field, and for every symbol
present in the candle/tick stream, measure the realized return over a
configurable set of horizons, signed by the surprise direction:

    signed_return(h) = sign(surprise) * (price(ts+h) - price(ts)) / price(ts)

For each (release_id, symbol, horizon):
  * n            — number of observations
  * mean         — mean signed return
  * stddev       — stddev of signed returns
  * sharpe       — mean / stddev * sqrt(observations per year)
  * hit_rate     — fraction with signed_return > 0
  * dd_p05       — 5th percentile of the *cumulative* signed return
                   (proxy for a reasonable stop)

Output:
  * A JSON report next to the input file.
  * A YAML snippet with suggested DriftRules (paste into config).

This closes the loop: the drift strategy stops being a prior and becomes
an empirical fit to your own venue and data.

Usage:
    python -m validation.drift_analysis --history data/events.jsonl \
        --symbols EURUSD,BTCUSDT --out config/drift_rules.yaml
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("validation.drift")

DEFAULT_HORIZONS_S = [300.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0]


# --------------------------------------------------------------------------- #
# Event loading                                                               #
# --------------------------------------------------------------------------- #

def _load_events(path: str | Path) -> list[dict]:
    """Load a JSONL of events. Accepts dicts or objects with .to_dict()."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if hasattr(obj, "to_dict"):
                obj = obj.to_dict()
            out.append(obj)
    return out


def _normalise(ev: dict) -> dict:
    """Return {type, ts, payload} regardless of source format."""
    t = ev.get("type")
    ts = ev.get("ts", ev.get("timestamp"))
    payload = ev.get("payload")
    if payload is None:
        # Flattened format — everything except type/ts is payload
        payload = {k: v for k, v in ev.items() if k not in ("type", "ts", "timestamp")}
    return {"type": t, "ts": float(ts) if ts is not None else 0.0, "payload": payload or {}}


# --------------------------------------------------------------------------- #
# Price series                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class PriceSeries:
    ts: list[float]
    px: list[float]

    def at_or_after(self, t: float) -> float | None:
        i = bisect.bisect_left(self.ts, t)
        if i >= len(self.ts):
            return None
        return self.px[i]

    def at_or_before(self, t: float) -> float | None:
        i = bisect.bisect_right(self.ts, t) - 1
        if i < 0:
            return None
        return self.px[i]


def build_price_series(events: Iterable[dict], symbols: Sequence[str]) -> dict[str, PriceSeries]:
    by_sym: dict[str, list[tuple[float, float]]] = {s: [] for s in symbols}
    for ev in events:
        e = _normalise(ev)
        if e["type"] not in ("candle", "tick"):
            continue
        p = e["payload"]
        sym = p.get("symbol")
        if sym not in by_sym:
            continue
        px = None
        if e["type"] == "candle":
            px = p.get("close")
        else:
            px = p.get("mid") or ((p.get("bid", 0) + p.get("ask", 0)) / 2 or None)
        if px is None:
            continue
        by_sym[sym].append((e["ts"], float(px)))
    out: dict[str, PriceSeries] = {}
    for s, pts in by_sym.items():
        pts.sort(key=lambda x: x[0])
        out[s] = PriceSeries(ts=[p[0] for p in pts], px=[p[1] for p in pts])
    return out


# --------------------------------------------------------------------------- #
# Per-release drift aggregation                                               #
# --------------------------------------------------------------------------- #

@dataclass
class DriftBucket:
    release_id: str
    symbol: str
    horizon_s: float
    n: int
    mean: float
    stddev: float
    sharpe_annualised: float
    hit_rate: float
    dd_p05: float


def _annualisation_factor(horizon_s: float) -> float:
    # Treat one release per month as the base cadence; overridable.
    return math.sqrt(12.0)


def analyse_release(
    release_id: str,
    releases: list[dict],
    series: PriceSeries,
    symbol: str,
    horizons: Sequence[float],
) -> list[DriftBucket]:
    buckets: list[DriftBucket] = []
    for h in horizons:
        signed_rets: list[float] = []
        for r in releases:
            ts = r["ts"]
            surprise = float(r["payload"].get("surprise", 0.0))
            if surprise == 0.0:
                continue
            p0 = series.at_or_after(ts)
            p1 = series.at_or_after(ts + h)
            if p0 is None or p1 is None or p0 <= 0:
                continue
            signed_rets.append(math.copysign(1.0, surprise) * (p1 - p0) / p0)
        if not signed_rets:
            continue
        arr = np.array(signed_rets, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = (mean / sd) * _annualisation_factor(h) if sd > 0 else 0.0
        hit = float((arr > 0).mean())
        dd_p05 = float(np.percentile(arr, 5))
        buckets.append(DriftBucket(
            release_id=release_id, symbol=symbol, horizon_s=h,
            n=len(arr), mean=mean, stddev=sd,
            sharpe_annualised=sharpe, hit_rate=hit, dd_p05=dd_p05,
        ))
    return buckets


def analyse(
    events: list[dict],
    symbols: Sequence[str],
    horizons: Sequence[float] = tuple(DEFAULT_HORIZONS_S),
    min_obs: int = 8,
) -> list[DriftBucket]:
    # Collect release_actual events keyed by release_id.
    by_release: dict[str, list[dict]] = {}
    for ev in events:
        e = _normalise(ev)
        if e["type"] != "release_actual":
            continue
        rid = e["payload"].get("release_id")
        if not rid:
            continue
        by_release.setdefault(rid, []).append(e)

    series = build_price_series(events, symbols)

    buckets: list[DriftBucket] = []
    for rid, rels in by_release.items():
        if len(rels) < min_obs:
            continue
        for sym in symbols:
            s = series.get(sym)
            if s is None or len(s.ts) < 10:
                continue
            buckets.extend(analyse_release(rid, rels, s, sym, horizons))
    return buckets


# --------------------------------------------------------------------------- #
# Suggesting DriftRules                                                       #
# --------------------------------------------------------------------------- #

def suggest_rules(
    buckets: list[DriftBucket],
    min_sharpe: float = 0.5,
    min_obs: int = 8,
) -> dict[str, dict]:
    """
    Return {release_id: {symbol, polarity, holding_s, sl_pct, tp_pct, position_pct}}.
    Only keeps the best horizon per release_id (highest annualised Sharpe).
    """
    best: dict[str, DriftBucket] = {}
    for b in buckets:
        if b.n < min_obs:
            continue
        if b.sharpe_annualised < min_sharpe:
            continue
        cur = best.get(b.release_id)
        if cur is None or b.sharpe_annualised > cur.sharpe_annualised:
            best[b.release_id] = b

    out: dict[str, dict] = {}
    for rid, b in best.items():
        polarity = 1 if b.mean >= 0 else -1
        # Suggest sl from the 5th percentile, tp from a 2x-of-mean heuristic.
        sl_pct = max(0.05, abs(b.dd_p05) * 100.0) * 1.2
        tp_pct = max(sl_pct * 1.5, abs(b.mean) * 100.0 * 3.0)
        out[rid] = {
            "symbol": b.symbol,
            "polarity": polarity,
            "holding_s": float(b.horizon_s),
            "sl_pct": round(min(5.0, sl_pct), 2),
            "tp_pct": round(min(10.0, tp_pct), 2),
            "position_pct": 0.5,
            "_stats": {
                "n": b.n,
                "mean": round(b.mean, 6),
                "sharpe_annualised": round(b.sharpe_annualised, 3),
                "hit_rate": round(b.hit_rate, 3),
                "dd_p05": round(b.dd_p05, 6),
            },
        }
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True,
                    help="JSONL file of events (from the event log or a recording)")
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated symbols to analyse, e.g. EURUSD,BTCUSDT")
    ap.add_argument("--horizons", default=",".join(str(int(h)) for h in DEFAULT_HORIZONS_S),
                    help="Comma-separated horizons in seconds")
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--min-sharpe", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="Where to write the suggested drift YAML (optional)")
    ap.add_argument("--report", default=None,
                    help="Where to write the full JSON report (optional)")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [float(h) for h in args.horizons.split(",") if h.strip()]

    events = _load_events(args.history)
    log.info("loaded %d events", len(events))
    buckets = analyse(events, symbols, horizons, min_obs=args.min_obs)
    if not buckets:
        print("no drift detected — insufficient releases or price history")
        return

    # Full report
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps([asdict(b) for b in buckets], indent=2))
        print(f"wrote full report -> {rp}")

    # Suggested rules
    suggestions = suggest_rules(buckets, min_sharpe=args.min_sharpe, min_obs=args.min_obs)

    print(f"\n{'release_id':22s} {'symbol':10s} {'horizon':>8s} {'n':>4s} "
          f"{'mean%':>8s} {'sharpe':>8s} {'hit':>6s}")
    for b in sorted(buckets, key=lambda x: -x.sharpe_annualised)[:15]:
        print(f"{b.release_id:22s} {b.symbol:10s} {b.horizon_s:>8.0f} {b.n:>4d} "
              f"{b.mean*100:>+8.3f} {b.sharpe_annualised:>8.2f} {b.hit_rate:>6.3f}")

    if not suggestions:
        print("\nno rule suggestions cleared the minimum Sharpe threshold")
        return

    print("\nSuggested drift rules:")
    for rid, rule in suggestions.items():
        s = rule.pop("_stats", {})
        print(f"  {rid:20s} -> {rule['symbol']} polarity={rule['polarity']:+d} "
              f"hold={rule['holding_s']:.0f}s sl={rule['sl_pct']}% "
              f"tp={rule['tp_pct']}%  (n={s.get('n')}, sharpe={s.get('sharpe_annualised')})")

    if args.out:
        import yaml as _yaml
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = {"strategy": {"release_drift": {"rules": {
            rid: dict(rule.items()) for rid, rule in suggestions.items()
        }}}}
        out.write_text(_yaml.safe_dump(doc, default_flow_style=False, sort_keys=False))
        print(f"\nwrote suggested rules -> {out}")
        print("paste under strategy.release_drift.rules in config/default.yaml")


if __name__ == "__main__":
    main()
