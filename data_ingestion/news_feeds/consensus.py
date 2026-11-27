# data_ingestion/news_feeds/consensus.py
"""
Consensus feed.

Consensus is the single most important input to the nowcast — without it,
`edge_vs_consensus` is meaningless. This module resolves consensus from the
best available source, in priority order:

  1. Explicit config overrides (`data.consensus.overrides`)
  2. FXMacroData API (`prediction_type=market_consensus`) — free USD tier
  3. Cached last-known value from a prior poll (graceful degradation)

The HTTP call is made **once per release**, at the point `RELEASE_EXPECTED`
fires (T-120s). If the feed is unavailable, we emit consensus=None and the
strategy treats the nowcast as a raw forecast with no edge calculation.

FXMacroData endpoints (from their docs):
  GET https://api.fxmacrodata.com/v1/predictions/{currency}/{indicator}
      ?prediction_type=market_consensus
      &pre_release_only=true
      &api_key=...
  GET https://api.fxmacrodata.com/v1/announcements/{currency}/{indicator}

Indicator slugs used: inflation, unemployment_rate, policy_rate, gdp, retail_sales.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.component import Component
from core.events import EventType, make_event

from .retry import HttpClient

log = logging.getLogger("data.consensus")

FXM_BASE = "https://api.fxmacrodata.com/v1"

# Map our internal release_id -> (currency, fxmacrodata_indicator_slug)
_RELEASE_TO_FXM: dict[str, tuple[str, str]] = {
    "CPI_YOY":            ("usd", "inflation"),
    "CORE_CPI_YOY":       ("usd", "core_inflation"),
    "PCE_YOY":            ("usd", "pce"),
    "NFP":                ("usd", "non_farm_payrolls"),
    "UNEMPLOYMENT_RATE":  ("usd", "unemployment_rate"),
    "FOMC_RATE":          ("usd", "policy_rate_midpoint"),
    "GDP_QOQ":            ("usd", "gdp"),
    "RETAIL_SALES_MOM":   ("usd", "retail_sales"),
    "JOBLESS_CLAIMS":     ("usd", "initial_jobless_claims"),
}


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Consensus:
    release_id: str
    value: float | None
    prior: float | None
    source: str                 # "config" | "fxmacrodata" | "cache" | "none"
    announcement_id: str | None = None
    raw: dict[str, Any] | None = None


class ConsensusFeed(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.consensus")
        self.api_key = cfg.secret("FXMACRODATA_API_KEY")
        self._overrides: dict[str, dict[str, Any]] = {
            o["release_id"]: o
            for o in (cfg.get("data.consensus.overrides", []) or [])
        }
        self._cache: dict[str, Consensus] = {}
        self._http: HttpClient | None = None
        if self.api_key:
            self._http = HttpClient(rate_per_sec=1.0, burst=3.0,
                                    user_agent="Cryptonaute/3.0 (research)")
        else:
            log.info("consensus: no FXMACRODATA_API_KEY — using config overrides only")

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.RELEASE_EXPECTED, self.on_expected)
        log.info("consensus feed online (overrides=%d, fxm=%s)",
                 len(self._overrides), bool(self.api_key))

    # ------------------------------------------------------------------ #
    def on_expected(self, ev) -> None:
        rid = ev.payload.get("release_id")
        if not rid:
            return

        c = self.resolve(rid)
        self.emit(make_event(
            EventType.NEWS, "data.consensus",
            payload={
                "id": f"consensus:{rid}:{int(ev.payload.get('scheduled_ts', 0))}",
                "ts": self.clock.now(),
                "headline": f"consensus {rid}={c.value} (source={c.source})",
                "source": "consensus",
                "release_id": rid,
                "consensus": c.value,
                "prior": c.prior,
                "consensus_source": c.source,
            },
            timestamp=self.clock.now(),
        ))
        self.metrics.inc("consensus_emitted", release_id=rid, source=c.source)

    # ------------------------------------------------------------------ #
    def resolve(self, release_id: str) -> Consensus:
        # 1) config override (highest priority — you always win)
        ov = self._overrides.get(release_id)
        if ov is not None:
            c = Consensus(
                release_id=release_id,
                value=_f(ov.get("consensus")),
                prior=_f(ov.get("prior")),
                source="config",
            )
            self._cache[release_id] = c
            return c

        # 2) FXMacroData
        if self._http is not None:
            c = self._fetch_fxm(release_id)
            if c is not None:
                self._cache[release_id] = c
                return c

        # 3) last-known cache
        if release_id in self._cache:
            return self._cache[release_id]

        # 4) nothing
        return Consensus(release_id=release_id, value=None, prior=None, source="none")

    # ------------------------------------------------------------------ #
    def _fetch_fxm(self, release_id: str) -> Consensus | None:
        pair = _RELEASE_TO_FXM.get(release_id)
        if pair is None:
            return None
        currency, indicator = pair
        url = f"{FXM_BASE}/predictions/{currency}/{indicator}"
        try:
            resp = self._http.get(url, params={
                "prediction_type": "market_consensus",
                "pre_release_only": "true",
                "api_key": self.api_key,
            }, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            log.debug("FXMacroData fetch failed for %s: %s", release_id, exc)
            return None

        if resp.status_code != 200:
            log.debug("FXMacroData %s -> %s", release_id, resp.status_code)
            return None

        try:
            doc = resp.json()
        except Exception:  # noqa: BLE001
            return None

        rows = doc.get("data") or []
        if not rows:
            return None
        row = rows[0]
        # Prefer market_consensus if the API returns it unfiltered.
        for r in rows:
            if r.get("prediction_type") == "market_consensus":
                row = r
                break

        return Consensus(
            release_id=release_id,
            value=_f(row.get("predicted_value")),
            prior=None,
            source="fxmacrodata",
            announcement_id=row.get("announcement_id"),
            raw=row,
        )

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._http is not None:
            self._http.close()


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
