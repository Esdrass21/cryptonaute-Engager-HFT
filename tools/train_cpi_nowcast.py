# tools/train_cpi_nowcast.py
"""
Fit the CPI nowcast on historical public data.

Inputs: a CSV with columns
    date, cpi_mom, gas_price_wow, oil_wow, rent_yoy, used_cars_mom,
    import_prices_mom, prior_cpi_mom, prior_core_cpi_mom

Sources (all public):
  * CPI MoM:             BLS series CUUR0000SA0 (via FRED CPIAUCSL pct_change)
  * Gas prices:          FRED GASREGW (weekly)
  * WTI oil:             FRED DCOILWTICO
  * Rent YoY:            Zillow ZORI CSV (US row)
  * Used cars:           Manheim UVVI press release, or FRED CUSR0000SETA02
  * Import prices MoM:   FRED IR (or IRDDS)
  * Prior CPI MoM:       lag of cpi_mom
  * Prior core CPI MoM:  lag of core_cpi_mom

We do NOT ship the data. You fetch it, join on date, and pass the CSV here.

Usage:
    python tools/train_cpi_nowcast.py --csv data/cpi_features.csv \
        --out config/nowcast/cpi_coefficients.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "gas_price_wow",
    "oil_wow",
    "rent_yoy",
    "used_cars_mom",
    "import_prices_mom",
    "prior_cpi_mom",
    "prior_core_cpi_mom",
]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    n, k = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])
    A = Xa.T @ Xa + lam * np.eye(k + 1)
    A[0, 0] -= lam * 0.9       # don't regularize the intercept
    return np.linalg.solve(A, Xa.T @ y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="config/nowcast/cpi_coefficients.json")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--regime-scale", type=float, default=1.2,
                    help="multiplier applied to in-sample RMSE for the emitted stddev")
    ap.add_argument("--target", default="cpi_mom",
                    help="column to predict (default: cpi_mom)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    df = df.dropna(subset=[args.target]).copy()
    X = df[FEATURES].fillna(0.0).values.astype(float)
    y = df[args.target].values.astype(float)

    # chronological train / holdout split
    split = int(len(df) * 0.8)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    beta = ridge_fit(Xtr, ytr, lam=args.lam)
    yhat_te = np.hstack([np.ones((len(Xte), 1)), Xte]) @ beta
    rmse_te = float(np.sqrt(np.mean((yhat_te - yte) ** 2)))

    # Also report against consensus: what is the ratio of "wins" (|error| < |prior|)?
    if "prior_cpi_mom" in df.columns:
        prior_te = df[args.target].shift(1).dropna().values[:len(yte)]
        if len(prior_te) == len(yte):
            errs_model = np.abs(yhat_te - yte)
            errs_prior = np.abs(prior_te - yte)
            win_rate = float((errs_model < errs_prior).mean())
        else:
            win_rate = float("nan")
    else:
        win_rate = float("nan")

    coef = {"intercept": float(beta[0])}
    for i, f in enumerate(FEATURES, 1):
        coef[f] = float(beta[i])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "coefficients": coef,
        "rmse": rmse_te,
        "regime_scale": args.regime_scale,
        "target": args.target,
        "n_train": int(len(Xtr)),
        "n_holdout": int(len(Xte)),
        "win_rate_vs_prior": None if win_rate != win_rate else win_rate,
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
        "features": FEATURES,
    }, indent=2), encoding="utf-8")

    print(f"fitted on {len(Xtr)} obs; holdout RMSE = {rmse_te:.4f} pp CPI")
    if win_rate == win_rate:
        print(f"model beats 'prior month' on {win_rate*100:.1f}% of holdout obs")
    print(f"wrote {out}")
    print("\nCoefficients (sorted by |value|):")
    for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {k:28s} {v:+.4f}")


if __name__ == "__main__":
    main()
