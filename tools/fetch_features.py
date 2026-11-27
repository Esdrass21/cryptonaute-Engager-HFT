# tools/fetch_features.py
"""
Assemble the training CSVs for the NFP and CPI nowcasts from public sources.

Inputs (all free-tier):
  * FRED    — ICSA, CCSA, ADPMNUSNERSA, JTSHIR, JTSJOL, MANEMP, RSAFS, PCEPI,
              PCEPILFE, DFF, DGS2, DGS10, DCOILWTICO, GASREGW, PAYEMS,
              CPIAUCSL, CUSR0000SETA02, IR
  * Zillow  — ZORI US row (CSV download)
  * BLS     — NFP + unemployment (via FRED PAYEMS / UNRATE)

Outputs:
  * data/features/nfp_features.csv
  * data/features/cpi_features.csv
  * data/features/fetch_report.json  (what succeeded, what was skipped)

Usage:
  python -m tools.fetch_features --out data/features
  python -m tools.fetch_features --out data/features --start 2015-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger("tools.fetch_features")


# --------------------------------------------------------------------------- #
# Feature → FRED series map                                                    #
# --------------------------------------------------------------------------- #

NFP_FRED = {
    "jobless_claims_4w_avg": ("ICSA",     "4w_avg"),
    "adp_private":           ("ADPMNUSNERSA", "level"),
    "jolts_hires":           ("JTSHIR",   "level"),
    "ism_manufacturing_emp": ("MANEMP",   "level"),
    "challenger_cuts":       ("JTSLDR",   "level"),
    "prior_nfp":             ("PAYEMS",   "mom_change"),
    "temp_help_change":      ("PAYEMS",   "mom_change"),
    "target_nfp":            ("PAYEMS",   "mom_change"),
    "target_unrate":         ("UNRATE",   "level"),
}

CPI_FRED = {
    "gas_price_wow":       ("GASREGW",       "wow_pct"),
    "oil_wow":             ("DCOILWTICO",    "wow_pct"),
    "import_prices_mom":   ("IR",            "mom_pct"),
    "used_cars_mom":       ("CUSR0000SETA02","mom_pct"),
    "prior_cpi_mom":       ("CPIAUCSL",      "mom_pct"),
    "target_cpi_mom":      ("CPIAUCSL",      "mom_pct"),
    "prior_core_cpi_mom":  ("CPILFESL",      "mom_pct"),
    "target_core_cpi_mom": ("CPILFESL",      "mom_pct"),
}

ZORI_METRO_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)


# --------------------------------------------------------------------------- #
# FRED fetcher                                                                 #
# --------------------------------------------------------------------------- #

def _fetch_fred_series(client, series_id: str, start: str) -> pd.DataFrame:
    obs = client.observations(series_id, start=start, limit=10_000, sort="asc")
    rows = [(o.date, o.value) for o in obs if o.value is not None]
    if not rows:
        return pd.DataFrame(columns=["date", series_id])
    df = pd.DataFrame(rows, columns=["date", series_id])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _apply_transform(df: pd.DataFrame, col: str, transform: str) -> pd.DataFrame:
    s = df[col].astype(float)
    if transform == "level":
        pass
    elif transform == "mom_change":
        s = s.diff()
    elif transform == "mom_pct":
        s = s.pct_change() * 100.0
    elif transform == "wow_pct":
        s = s.pct_change() * 100.0
    elif transform == "4w_avg":
        s = s.rolling(4, min_periods=1).mean()
    else:
        raise ValueError(f"unknown transform {transform!r}")
    out = df.copy()
    out[col] = s
    return out


# --------------------------------------------------------------------------- #
# ZORI fetcher (rent)                                                          #
# --------------------------------------------------------------------------- #

def _fetch_zori(region: str = "United States") -> pd.DataFrame | None:
    from data_ingestion.news_feeds.retry import HttpClient
    from data_ingestion.news_feeds.zillow import parse_zori_csv

    http = HttpClient(rate_per_sec=0.5, burst=1.0)
    try:
        resp = http.get(ZORI_METRO_URL, timeout=20.0)
    except Exception as exc:
        log.warning("ZORI fetch failed: %s", exc)
        return None
    finally:
        http.close()

    if resp.status_code != 200:
        return None
    points = parse_zori_csv(resp.text, region=region, max_points=200)
    if not points:
        return None

    by_date = {p.date: p.value for p in points}
    rows = []
    for p in points:
        ly = f"{int(p.date[:4]) - 1:04d}-{p.date[5:]}"
        if ly in by_date and by_date[ly] > 0:
            yoy = (p.value - by_date[ly]) / by_date[ly] * 100.0
            rows.append((p.date, yoy))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "rent_yoy"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# --------------------------------------------------------------------------- #
# Assemble NFP                                                                 #
# --------------------------------------------------------------------------- #

def build_nfp(client, start: str, zori: pd.DataFrame | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature, (sid, transform) in NFP_FRED.items():
        try:
            raw = _fetch_fred_series(client, sid, start)
        except Exception as exc:
            log.warning("FRED fetch failed for %s (%s): %s", feature, sid, exc)
            continue
        if raw.empty:
            log.warning("FRED returned empty for %s (%s)", feature, sid)
            continue
        t = _apply_transform(raw, sid, transform)
        t = t.rename(columns={sid: feature})
        frames.append(t)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df = df.resample("ME").last()
    return df


# --------------------------------------------------------------------------- #
# Assemble CPI                                                                 #
# --------------------------------------------------------------------------- #

def build_cpi(client, start: str, zori: pd.DataFrame | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature, (sid, transform) in CPI_FRED.items():
        try:
            raw = _fetch_fred_series(client, sid, start)
        except Exception as exc:
            log.warning("FRED fetch failed for %s (%s): %s", feature, sid, exc)
            continue
        if raw.empty:
            continue
        t = _apply_transform(raw, sid, transform)
        t = t.rename(columns={sid: feature})
        frames.append(t)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df = df.resample("ME").last()

    if zori is not None and not zori.empty:
        z = zori.resample("ME").last()
        df = df.join(z, how="left")
        df["rent_yoy"] = df["rent_yoy"].ffill()

    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--fred-key", default=None,
                    help="override; default reads FRED_API_KEY from env")
    ap.add_argument("--skip-zori", action="store_true")
    ap.add_argument("--skip-nfp", action="store_true")
    ap.add_argument("--skip-cpi", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    key = args.fred_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY not set (register free at https://fredaccount.stlouisfed.org/apikeys)")

    from data_ingestion.news_feeds.fred import FredClient
    client = FredClient(key)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "start": args.start,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    zori = None
    if not args.skip_zori:
        zori = _fetch_zori()
        if zori is not None:
            zori.to_csv(out / "zori_yoy.csv")
            report["zori"] = {"rows": len(zori)}
            log.info("ZORI: %d monthly rows", len(zori))
        else:
            report["zori"] = {"error": "fetch or parse failed"}

    if not args.skip_nfp:
        nfp = build_nfp(client, args.start, zori)
        if nfp.empty:
            report["nfp"] = {"error": "no data"}
        else:
            nfp.to_csv(out / "nfp_features.csv")
            report["nfp"] = {"rows": len(nfp), "cols": list(nfp.columns)}
            log.info("NFP: %d rows, %d features", len(nfp), len(nfp.columns))

    if not args.skip_cpi:
        cpi = build_cpi(client, args.start, zori)
        if cpi.empty:
            report["cpi"] = {"error": "no data"}
        else:
            cpi.to_csv(out / "cpi_features.csv")
            report["cpi"] = {"rows": len(cpi), "cols": list(cpi.columns)}
            log.info("CPI: %d rows, %d features", len(cpi), len(cpi.columns))

    (out / "fetch_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote -> {out}")
    print(f"report -> {out / 'fetch_report.json'}")

    client.close()


if __name__ == "__main__":
    main()
