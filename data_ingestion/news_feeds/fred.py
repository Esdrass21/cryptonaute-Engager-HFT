# data_ingestion/news_feeds/fred.py
"""
FRED / ALFRED client.

Gives you:
  * `series(id)`               -> metadata (units, frequency, seasonal adj.)
  * `observations(id, ...)`    -> current observations
  * `vintages(id, realtime_start, realtime_end)` -> ALFRED revision history
  * `release_dates(release_id)` -> past + future scheduled release dates
  * `series_vintage_dates(id)` -> every date this series was revised

ALFRED is the reason this matters: to backtest a nowcast without leaking the
future, you must use the value **as it was known at the time**. FRED alone
returns the latest revision. ALFRED via `realtime_start=realtime_end=D` returns
the value known on date D.

Auth:
  * API key from env `FRED_API_KEY` (32-char lowercase alphanumeric).
  * Registration: https://fredaccount.stlouisfed.org/apikeys

Replay mode:
  * `enabled=False` on the HttpClient. In replay, the bus re-delivers recorded
    NEWS events; no network call is ever made.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.component import Component
from core.events import EventType, make_event

from .retry import HttpClient

log = logging.getLogger("data.fred")

FRED_BASE = "https://api.stlouisfed.org/fred"

# --------------------------------------------------------------------------- #
# Series map: FRED series ID -> (release_id, human label, units)              #
# --------------------------------------------------------------------------- #
SERIES_MAP: dict[str, tuple[str, str, str]] = {
    "ICSA":        ("JOBLESS_CLAIMS",       "Initial jobless claims",     "thousands"),
    "CCSA":        ("CONTINUED_CLAIMS",     "Continued claims",            "thousands"),
    "ADPMNUSNERSA":("ADP_PRIVATE",          "ADP private payrolls",        "thousands"),
    "JTSHIR":      ("JOLTS_HIRES",          "JOLTS hires",                 "thousands"),
    "JTSJOL":      ("JOLTS_OPENINGS",       "JOLTS job openings",          "thousands"),
    "JTSLDR":      ("JOLTS_LAYOFFS",        "JOLTS layoffs & discharges",  "thousands"),
    "MANEMP":      ("MANUFACTURING_EMP",    "Manufacturing employment",    "thousands"),
    "RSAFS":       ("RETAIL_SALES_MOM",     "Retail sales advance",        "millions"),
    "PCEPI":       ("PCE_YOY",              "PCE price index",             "index"),
    "PCEPILFE":    ("CORE_PCE_YOY",         "Core PCE price index",        "index"),
    "DFF":         ("FED_FUNDS_EFFECTIVE",  "Fed funds effective rate",    "percent"),
    "DFEDTARU":    ("FED_FUNDS_TARGET_UPPER","Fed funds target upper",     "percent"),
    "DGS2":        ("UST_2Y",               "2-year Treasury yield",       "percent"),
    "DGS10":       ("UST_10Y",              "10-year Treasury yield",      "percent"),
    "DCOILWTICO":  ("WTI_OIL",              "WTI crude oil",               "USD/bbl"),
    "GASREGW":     ("GAS_REGULAR_WEEKLY",   "Weekly US regular gas price", "USD/gal"),
    "UNRATE":      ("UNEMPLOYMENT_RATE",    "Unemployment rate",           "percent"),
    "PAYEMS":      ("NFP",                  "Total nonfarm payrolls",      "thousands"),
    "CPIAUCSL":    ("CPI_YOY",              "CPI-U all items",             "index"),
}


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Observation:
    date: str            # YYYY-MM-DD
    value: float | None
    realtime_start: str | None = None
    realtime_end: str | None = None

    def to_dict(self) -> dict:
        return {
            "date": self.date, "value": self.value,
            "realtime_start": self.realtime_start, "realtime_end": self.realtime_end,
        }


# --------------------------------------------------------------------------- #
# Client                                                                       #
# --------------------------------------------------------------------------- #

class FredClient:
    def __init__(self, api_key: str, *, http: HttpClient | None = None) -> None:
        if not api_key or len(api_key) < 20:
            raise ValueError("FRED API key looks invalid (need 32-char key)")
        self.api_key = api_key
        self.http = http or HttpClient(rate_per_sec=2.0, burst=5.0)

    # ------------------------------------------------------------------ #
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        p = dict(params)
        p["api_key"] = self.api_key
        p.setdefault("file_type", "json")
        url = f"{FRED_BASE}/{endpoint}"
        resp = self.http.get(url, params=p, timeout=8.0)
        if resp.status_code == 400:
            raise ValueError(f"FRED 400: {resp.text[:200]}")
        if resp.status_code == 429:
            log.warning("FRED rate-limited after retries")
            raise RuntimeError("FRED rate limit exhausted")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    def series(self, series_id: str) -> dict[str, Any]:
        doc = self._get("series", {"series_id": series_id})
        return doc.get("seriess", [{}])[0]

    def observations(
        self,
        series_id: str,
        *,
        start: str | None = None,       # YYYY-MM-DD
        end: str | None = None,
        limit: int = 10_000,
        sort: str = "asc",
    ) -> list[Observation]:
        params: dict[str, Any] = {"series_id": series_id, "limit": limit, "sort_order": sort}
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        doc = self._get("series/observations", params)
        return [
            Observation(
                date=o["date"],
                value=_to_float(o.get("value")),
                realtime_start=o.get("realtime_start"),
                realtime_end=o.get("realtime_end"),
            )
            for o in doc.get("observations", [])
        ]

    # ------------------------------------------------------------------ #
    def vintages(
        self,
        series_id: str,
        *,
        realtime_start: str,
        realtime_end: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
        vintage_dates: list[str] | None = None,
    ) -> list[Observation]:
        """
        ALFRED: return the value of `series_id` as it was known during the
        real-time period [realtime_start, realtime_end].
        """
        if vintage_dates and (realtime_start or realtime_end):
            raise ValueError("vintage_dates is mutually exclusive with realtime bounds")
        params: dict[str, Any] = {"series_id": series_id, "limit": 10_000, "sort_order": "asc"}
        if vintage_dates:
            params["vintage_dates"] = ",".join(vintage_dates)
        else:
            params["realtime_start"] = realtime_start
            if realtime_end:
                params["realtime_end"] = realtime_end
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        doc = self._get("series/observations", params)
        return [
            Observation(
                date=o["date"],
                value=_to_float(o.get("value")),
                realtime_start=o.get("realtime_start"),
                realtime_end=o.get("realtime_end"),
            )
            for o in doc.get("observations", [])
        ]

    def vintage_dates(
        self,
        series_id: str,
        *,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> list[str]:
        """Every date on which this series was revised."""
        params: dict[str, Any] = {"series_id": series_id}
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end
        doc = self._get("series/vintagedates", params)
        return list(doc.get("vintage_dates", []))

    # ------------------------------------------------------------------ #
    def release_dates(self, release_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        """
        Past + future scheduled dates for a release (CPI, NFP, etc.).

        FRED release IDs:
          * 10  = Consumer Price Index
          * 50  = Employment Situation (NFP)
          * 180 = Producer Price Index
        """
        doc = self._get("release/dates", {"release_id": release_id, "limit": limit,
                                          "include_release_dates_with_no_data": "true"})
        return doc.get("release_dates", [])

    def releases(self) -> list[dict[str, Any]]:
        doc = self._get("releases", {"limit": 1000})
        return doc.get("releases", [])

    def close(self) -> None:
        self.http.close()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_float(v: Any) -> float | None:
    if v is None or v == "." or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Bus component                                                               #
# --------------------------------------------------------------------------- #

class FredFeed(Component):
    """
    On each `on_tick` advance, checks whether any tracked FRED series is due
    for release; if so, emits a NEWS event with the newest observation.

    In replay mode, this component is inert — the recorded NEWS events are
    re-delivered by the bus.
    """

    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.fred")
        self.api_key = cfg.secret("FRED_API_KEY")
        self.series = list(cfg.get("data.fred.series", list(SERIES_MAP.keys())))
        self.poll_interval_s = float(cfg.get("data.fred.poll_interval_s", 3600.0))
        self._last_poll = 0.0
        self._last_value: dict[str, float | None] = {}
        self._client: FredClient | None = None
        if self.api_key:
            try:
                self._client = FredClient(self.api_key)
            except Exception as exc:  # noqa: BLE001
                log.warning("FRED client init failed: %s", exc)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._client is None:
            log.warning("FRED feed disabled (no FRED_API_KEY)")
            return
        log.info("FRED feed online (series=%d, poll=%ss)",
                 len(self.series), self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._client is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            return  # replay: no network
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        self._poll()

    def _poll(self) -> None:
        for sid in self.series:
            try:
                obs = self._client.observations(sid, limit=2, sort="desc")
            except Exception as exc:  # noqa: BLE001
                log.debug("FRED poll failed for %s: %s", sid, exc)
                self.metrics.inc("fred_poll_failed", series=sid)
                continue
            if not obs:
                continue
            latest = obs[0]
            if latest.value is None:
                continue
            prev = self._last_value.get(sid)
            if prev is not None and latest.value == prev:
                continue
            self._last_value[sid] = latest.value
            meta = SERIES_MAP.get(sid, (sid, sid, ""))
            release_id, label, units = meta
            self.emit(make_event(
                EventType.NEWS, "data.fred",
                payload={
                    "id": f"fred:{sid}:{latest.date}",
                    "ts": self.clock.now(),
                    "headline": f"{label}: {latest.value} {units}",
                    "source": "FRED",
                    "series_id": sid,
                    "release_id": release_id,
                    "actual": latest.value,
                    "date": latest.date,
                    "units": units,
                },
                timestamp=self.clock.now(),
            ))
            self.metrics.inc("fred_observation", series=sid)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
