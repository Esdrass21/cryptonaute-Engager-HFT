# analytics/nowcast/cpi.py
"""
CPI nowcast.

Model: ridge regression over public price inputs, same shape as the NFP nowcast.
Coefficients loaded from `config/nowcast/cpi_coefficients.json`; falls back to
literature defaults if absent.

Features (all public, all free-tier):
  * gas_price_wow           — weekly regular gas price, WoW % change
  * oil_wow                 — WTI weekly % change (leads gas by ~1 week)
  * rent_yoy                — Zillow ZORI YoY (shelter ≈ 35% of CPI)
  * used_cars_mom           — Manheim used vehicle value index, MoM %
  * import_prices_mom       — FRED IR MoM % (leads core goods)
  * prior_cpi_mom           — prior month's CPI MoM (persistence)
  * prior_core_cpi_mom      — prior core CPI MoM

Output: NOWCAST {release_id: "CPI_MOM" | "CPI_YOY" | "CORE_CPI_YOY",
                 mean, stddev, edge_vs_consensus, feature_vector, coverage}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from analytics.nowcast.base import NowcastBase

log = logging.getLogger("analytics.nowcast.cpi")

FEATURES = [
    "gas_price_wow",
    "oil_wow",
    "rent_yoy",
    "used_cars_mom",
    "import_prices_mom",
    "prior_cpi_mom",
    "prior_core_cpi_mom",
]

_DEFAULT_COEF = {
    "intercept": 0.24,
    "gas_price_wow":  0.018,
    "oil_wow":        0.004,
    "rent_yoy":       0.012,
    "used_cars_mom":  0.008,
    "import_prices_mom": 0.030,
    "prior_cpi_mom":  0.28,
    "prior_core_cpi_mom": 0.18,
}
_DEFAULT_RMSE = 0.09
_DEFAULT_REGIME_SCALE = 1.2


class _CPIState:
    def __init__(self) -> None:
        self.v: Dict[str, float | None] = {k: None for k in FEATURES}
        self._prev_gas: float | None = None
        self._prev_oil: float | None = None
        self._prev_used: float | None = None
        self._prev_import: float | None = None

    def update_gas(self, v: float) -> None:
        if self._prev_gas is not None and self._prev_gas > 0:
            self.v["gas_price_wow"] = (v - self._prev_gas) / self._prev_gas * 100.0
        self._prev_gas = v

    def update_oil(self, v: float) -> None:
        if self._prev_oil is not None and self._prev_oil > 0:
            self.v["oil_wow"] = (v - self._prev_oil) / self._prev_oil * 100.0
        self._prev_oil = v

    def update_used(self, v: float) -> None:
        if self._prev_used is not None and self._prev_used > 0:
            self.v["used_cars_mom"] = (v - self._prev_used) / self._prev_used * 100.0
        self._prev_used = v

    def update_import(self, v: float) -> None:
        if self._prev_import is not None and self._prev_import > 0:
            self.v["import_prices_mom"] = (v - self._prev_import) / self._prev_import * 100.0
        self._prev_import = v

    def set(self, key: str, value: float) -> None:
        if key in self.v:
            self.v[key] = float(value)

    def as_dict(self) -> Dict[str, float]:
        return {k: (v if v is not None else 0.0) for k, v in self.v.items()}

    def coverage(self) -> float:
        n = sum(1 for v in self.v.values() if v is not None)
        return n / max(1, len(self.v))


class CPINowcast(NowcastBase):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="analytics.nowcast.cpi")
        self.state = _CPIState()
        self.coef, self.rmse, self.scale = self._load(cfg)

    def release_ids(self) -> list[str]:
        return ["CPI_MOM", "CPI_YOY", "CORE_CPI_YOY"]

    def features(self) -> Dict[str, float]:
        return self.state.as_dict()

    def predict(self, features: Dict[str, float]) -> tuple[float, float]:
        mean = self.coef["intercept"]
        for k in FEATURES:
            mean += self.coef.get(k, 0.0) * features.get(k, 0.0)
        return mean, self.rmse * self.scale

    def _load(self, cfg) -> tuple[Dict[str, float], float, float]:
        path = cfg.get("nowcast.cpi.coefficients_path",
                       "config/nowcast/cpi_coefficients.json")
        p = Path(path)
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                coef = dict(doc.get("coefficients", {}))
                for k in FEATURES + ["intercept"]:
                    coef.setdefault(k, _DEFAULT_COEF.get(k, 0.0))
                return coef, float(doc.get("rmse", _DEFAULT_RMSE)), \
                       float(doc.get("regime_scale", _DEFAULT_REGIME_SCALE))
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to load %s: %s", p, exc)
        return dict(_DEFAULT_COEF), _DEFAULT_RMSE, _DEFAULT_REGIME_SCALE

    def _on_news(self, ev) -> None:
        rid = ev.payload.get("release_id")
        val = ev.payload.get("actual")
        if rid is None or val is None:
            return
        v = float(val)

        if rid == "GAS_REGULAR_WEEKLY":
            self.state.update_gas(v)
        elif rid == "WTI_OIL":
            self.state.update_oil(v)
        elif rid == "ZORI_YOY":
            self.state.set("rent_yoy", v)
        elif rid == "MANHEIM_UVVI":
            self.state.update_used(v)
        elif rid == "IMPORT_PRICES_MOM":
            self.state.update_import(v)
        elif rid == "CPI_MOM":
            self.state.set("prior_cpi_mom", v)
        elif rid == "CORE_CPI_MOM":
            self.state.set("prior_core_cpi_mom", v)

        super()._on_news(ev)

    def _on_expected(self, ev) -> None:
        rid = ev.payload.get("release_id")
        if rid not in self.release_ids():
            return
        consensus = ev.payload.get("consensus")
        prior = ev.payload.get("prior")
        feats = self.features()
        mean, stddev = self.predict(feats)
        cov = self.state.coverage()
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
