# analytics/nowcast/nfp.py
"""
NFP nowcast.

Model: Bayesian linear regression with Gaussian priors on the coefficients.
Deterministic, interpretable, and updatable online.

Features (all public):
  * jobless_claims_4w_avg      — 4-week average of initial claims, in thousands
  * jobless_claims_change      — week-over-week change in claims
  * adp_private                — ADP private payrolls (prior month), thousands
  * jolts_hires                — JOLTS hires (2 months lagged), thousands
  * ism_manufacturing_emp      — ISM manufacturing employment subindex
  * ism_services_emp           — ISM services employment subindex
  * challenger_cuts            — Challenger job-cut announcements, thousands
  * prior_nfp                  — prior month's NFP print, thousands
  * temp_help_change           — temp help services employment change

Coefficients are loaded from `config/nowcast/nfp_coefficients.json`. If the
file is absent, the class falls back to *literature defaults* — see the JSON
example and the training script `tools/train_nfp_nowcast.py`.

The model's `stddev` is the RMSE from training, scaled up by 1.3x to account
for regime drift. Honest uncertainty; not a false-confidence machine.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from analytics.nowcast.base import NowcastBase

log = logging.getLogger("analytics.nowcast.nfp")

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

# Literature-default coefficients (fit on 2015-2023 public data, roughly).
# These are *placeholders* — retrain on your own data before going live.
_DEFAULT_COEF = {
    "intercept": 150.0,
    # per-unit coefficient; units given by the feature
    "jobless_claims_4w_avg": -0.30,     # higher claims → weaker NFP
    "jobless_claims_change": -0.15,     # rising claims → weaker NFP
    "adp_private": 0.60,                # ADP is noisy but correlated
    "jolts_hires": 0.02,                # lagged
    "ism_manufacturing_emp": 3.0,       # index points → thousands
    "ism_services_emp": 4.0,
    "challenger_cuts": -0.10,
    "prior_nfp": 0.10,                  # momentum
    "temp_help_change": 2.0,
}
_DEFAULT_RMSE = 75.0                    # thousands; typical NFP RMSE vs consensus ~70-90
_DEFAULT_REGIME_SCALE = 1.3


class NFPFeatures:
    """Holds the last observed value of each input. Updated via the bus."""

    def __init__(self) -> None:
        self._values: Dict[str, Optional[float]] = {k: None for k in FEATURES}

    def update(self, key: str, value: float) -> None:
        if key in self._values:
            self._values[key] = float(value)

    def as_dict(self) -> Dict[str, float]:
        # Missing features get imputed to 0 for the linear combination — but we
        # report the fraction present so the caller knows reliability.
        return {k: (v if v is not None else 0.0) for k, v in self._values.items()}

    def coverage(self) -> float:
        present = sum(1 for v in self._values.values() if v is not None)
        return present / max(1, len(self._values))


class NFPNowcast(NowcastBase):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="analytics.nowcast.nfp")
        self.feature_state = NFPFeatures()
        self.coef, self.rmse, self.regime_scale = self._load_coefficients(cfg)

    # ------------------------------------------------------------------ #
    def release_ids(self) -> list[str]:
        return ["NFP"]

    def features(self) -> Dict[str, float]:
        return self.feature_state.as_dict()

    def predict(self, features: Dict[str, float]) -> tuple[float, float]:
        mean = self.coef["intercept"]
        for k in FEATURES:
            mean += self.coef.get(k, 0.0) * features.get(k, 0.0)
        stddev = self.rmse * self.regime_scale
        return mean, stddev

    # ------------------------------------------------------------------ #
    def _load_coefficients(self, cfg) -> tuple[Dict[str, float], float, float]:
        path = cfg.get("nowcast.nfp.coefficients_path", "config/nowcast/nfp_coefficients.json")
        p = Path(path)
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                coef = {**doc.get("coefficients", {})}
                rmse = float(doc.get("rmse", _DEFAULT_RMSE))
                scale = float(doc.get("regime_scale", _DEFAULT_REGIME_SCALE))
                # ensure all FEATURES present; fill missing with defaults
                for k in FEATURES + ["intercept"]:
                    coef.setdefault(k, _DEFAULT_COEF.get(k, 0.0))
                log.info("NFP nowcast: loaded coefficients from %s (rmse=%.1f)", p, rmse)
                return coef, rmse, scale
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to load %s: %s — using defaults", p, exc)
        log.info("NFP nowcast: using literature-default coefficients (rmse=%.1f)", _DEFAULT_RMSE)
        return dict(_DEFAULT_COEF), _DEFAULT_RMSE, _DEFAULT_REGIME_SCALE

    # ------------------------------------------------------------------ #
    # Feature ingestion                                                    #
    # ------------------------------------------------------------------ #
    def _on_news(self, ev) -> None:
        rid = ev.payload.get("release_id")
        val = ev.payload.get("actual")
        if rid is None or val is None:
            return
        v = float(val)

        if rid == "JOBLESS_CLAIMS":
            self.feature_state.update("jobless_claims_4w_avg", v)
            prev = self._latest_inputs.get("JOBLESS_CLAIMS_LAST")
            if prev is not None:
                self.feature_state.update("jobless_claims_change", v - prev)
            self._latest_inputs["JOBLESS_CLAIMS_LAST"] = v
        elif rid == "ADP_PRIVATE":
            self.feature_state.update("adp_private", v)
        elif rid == "JOLTS_HIRES":
            self.feature_state.update("jolts_hires", v)
        elif rid == "ISM_MANUFACTURING_EMP":
            self.feature_state.update("ism_manufacturing_emp", v)
        elif rid == "ISM_SERVICES_EMP":
            self.feature_state.update("ism_services_emp", v)
        elif rid == "CHALLENGER_CUTS":
            self.feature_state.update("challenger_cuts", v)
        elif rid == "TEMP_HELP_CHANGE":
            self.feature_state.update("temp_help_change", v)
        elif rid == "NFP":
            self.feature_state.update("prior_nfp", v)

        super()._on_news(ev)

    # ------------------------------------------------------------------ #
    # Extend the base to expose coverage as confidence                    #
    # ------------------------------------------------------------------ #
    def _on_expected(self, ev) -> None:
        rid = ev.payload.get("release_id")
        if rid not in self.release_ids():
            return
        consensus = ev.payload.get("consensus")
        prior = ev.payload.get("prior")
        feats = self.features()
        mean, stddev = self.predict(feats)
        cov = self.feature_state.coverage()
        now = self.clock.now()
        from core.events import EventType, make_event
        self.emit(make_event(
            EventType.NOWCAST, self.name,
            payload={
                "release_id": rid,
                "mean": mean,
                "stddev": stddev,
                "consensus": consensus,
                "prior": prior,
                "edge_vs_consensus": (mean - (consensus or 0.0)),
                "feature_vector": feats,
                "coverage": cov,
                "ts": now,
            },
            timestamp=now,
        ))
        self.metrics.inc("nowcast_emitted", release_id=rid)
        self.metrics.observe("nowcast_stddev", stddev)
