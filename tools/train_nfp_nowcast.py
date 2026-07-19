# tools/train_nfp_nowcast.py
"""
Fit the NFP nowcast on historical public data.

Inputs: a CSV with columns
    date, nfp, jobless_claims_4w_avg, jobless_claims_change, adp_private,
    jolts_hires, ism_manufacturing_emp, ism_services_emp, challenger_cuts,
    prior_nfp, temp_help_change

Sources for the CSV (all public):
  * NFP / unemployment: BLS API v2 (api.bls.gov) — series CES0000000001
  * Jobless claims: DOL weekly releases, or FRED series ICSA
  * ADP: ADP employment report, or FRED series ADPMNUSNERSA
  * JOLTS: BLS series JTSHIR
  * ISM: ISM press releases, or FRED series MANEMP for proxy
  * Challenger: challengergray.com public releases
  * Temp help: BLS series CES6056132001

We do NOT ship the data. You fetch it via the BLS API (registration is free)
and FRED (registration is free). This script does the fitting.

Usage:
    python tools/train_nfp_nowcast.py --csv data/nfp_features.csv \
        --out config/nowcast/nfp_coefficients.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "jobless_claims_4w_avg",
    "jobless_claims_change",
    "adp_private",
    "jolts_hires",
    "ism_manufacturing_emp",
    "ism_services_emp",
    "challenger_cuts",
    "prior_nfp",
    "temp_help_change",
]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge. Adds an intercept column internally."""
    n, k = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])
    A = Xa.T @ Xa + lam * np.eye(k + 1)
    # don't regularize the intercept much
    A[0, 0] -= lam * 0.9
    return np.linalg.solve(A, Xa.T @ y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="config/nowcast/nfp_coefficients.json")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--regime-scale", type=float, default=1.3,
                    help="multiplier applied to in-sample RMSE for the emitted stddev")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["nfp"] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    df = df.dropna(subset=["nfp"]).copy()
    X = df[FEATURES].fillna(0.0).values.astype(float)
    y = df["nfp"].values.astype(float)

    # train / holdout split (last 20% as holdout, chronological)
    split = int(len(df) * 0.8)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    beta = ridge_fit(Xtr, ytr, lam=args.lam)
    yhat_te = np.hstack([np.ones((len(Xte), 1)), Xte]) @ beta
    rmse_te = float(np.sqrt(np.mean((yhat_te - yte) ** 2)))

    coef = {"intercept": float(beta[0])}
    for i, f in enumerate(FEATURES, 1):
        coef[f] = float(beta[i])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "coefficients": coef,
        "rmse": rmse_te,
        "regime_scale": args.regime_scale,
        "n_train": int(len(Xtr)),
        "n_holdout": int(len(Xte)),
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
        "features": FEATURES,
    }, indent=2), encoding="utf-8")

    print(f"fitted on {len(Xtr)} obs; holdout RMSE = {rmse_te:.1f}k NFP")
    print(f"wrote {out}")
    print("\nTop coefficients (sorted by |value|):")
    for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {k:28s} {v:+.3f}")


if __name__ == "__main__":
    main()
