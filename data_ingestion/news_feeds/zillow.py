# data_ingestion/news_feeds/zillow.py
"""
Public rent + used-car feeds for the CPI nowcast.

Two sources:

1. Zillow ZORI (Observed Rent Index) — free CSV, monthly update.
   https://www.zillow.com/research/data/
   We use the metro-level file, filter to the "United States" row,
   compute YoY from the last two same-month observations, emit NEWS.

2. Manheim Used Vehicle Value Index — proprietary. Two paths:
   a) Cox Automotive's public research page — scrape the headline number
      (fragile; URL structure changes; tolerant parser with fallback).
   b) FRED proxy `CUSR0000SETA02` (CPI Used Cars) — one-month lagging,
      but reliable and free. Used as fallback when scraping fails.

Both feeds are replay-safe: in `clock.mode == "virtual"`, no network call.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass

from core.component import Component
from core.events import EventType, make_event

from .retry import HttpClient

log = logging.getLogger("data.zillow")

# --------------------------------------------------------------------------- #
# Defaults — update the URLs if Zillow or Cox change theirs.                  #
# --------------------------------------------------------------------------- #
ZORI_METRO_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)
ZORI_US_REGION = "United States"

MANHEIM_PAGE_URL = (
    "https://www.coxautoinc.com/insights/manheim-used-vehicle-value-index/"
)

# --------------------------------------------------------------------------- #
# ZORI parsing                                                                #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ZoriPoint:
    region: str
    date: str            # YYYY-MM-01
    value: float         # USD/month


def parse_zori_csv(text: str, region: str = ZORI_US_REGION,
                   max_points: int = 36) -> list[ZoriPoint]:
    """
    Parse a Zillow ZORI CSV. Keeps the last `max_points` monthly observations
    for the requested region. Returns points ascending by date.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []
    # Identify date columns — everything matching YYYY-MM-DD.
    date_cols = [c for c in reader.fieldnames if re.match(r"\d{4}-\d{2}-\d{2}", c)]
    if not date_cols:
        return []
    date_cols.sort()

    region_key = None
    for candidate in ("RegionName", "Region", "region_name"):
        if candidate in reader.fieldnames:
            region_key = candidate
            break
    if region_key is None:
        return []

    for row in reader:
        if (row.get(region_key) or "").strip() != region:
            continue
        points: list[ZoriPoint] = []
        for c in date_cols[-max_points:]:
            raw = (row.get(c) or "").strip()
            if not raw:
                continue
            try:
                points.append(ZoriPoint(region=region, date=c, value=float(raw)))
            except ValueError:
                continue
        return points
    return []


def zori_yoy(points: list[ZoriPoint]) -> float | None:
    """YoY % change using the latest point and the same-month-last-year point."""
    if len(points) < 13:
        return None
    latest = points[-1]
    # Find the point 12 months earlier.
    ly = _month_minus_12(latest.date)
    for p in points:
        if p.date == ly:
            if p.value == 0:
                return None
            return (latest.value - p.value) / p.value * 100.0
    return None


def _month_minus_12(date_str: str) -> str:
    y, m, d = date_str.split("-")
    return f"{int(y)-1:04d}-{m}-{d}"


# --------------------------------------------------------------------------- #
# Manheim parsing (best-effort; falls back to FRED proxy)                     #
# --------------------------------------------------------------------------- #

_MANHEIM_RE = re.compile(
    r"(?:Manheim Used Vehicle Value Index|UVVI)[^.]{0,80}?(\d{2,3}(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_manheim_html(html: str) -> float | None:
    m = _MANHEIM_RE.search(html)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Feed component                                                               #
# --------------------------------------------------------------------------- #

class ZillowFeed(Component):
    """
    Downloads ZORI CSV + Manheim index on a schedule. Emits NEWS events with
    `release_id` matching what the CPI nowcast consumes.

    Config:
        data.news.zillow.enabled: true
        data.news.zillow.poll_interval_s: 86400
        data.news.zillow.metro_url: <override>
        data.news.zillow.manheim_url: <override>
        data.news.zillow.us_region: "United States"
    """

    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.zillow")
        self.poll_interval_s = float(cfg.get("data.news.zillow.poll_interval_s", 86400.0))
        self.metro_url = str(cfg.get("data.news.zillow.metro_url", ZORI_METRO_URL))
        self.manheim_url = str(cfg.get("data.news.zillow.manheim_url", MANHEIM_PAGE_URL))
        self.us_region = str(cfg.get("data.news.zillow.us_region", ZORI_US_REGION))
        self._last_poll = 0.0
        self._last_values: dict[str, float] = {}
        self._http: HttpClient | None = None
        if bool(cfg.get("data.news.zillow.enabled", True)):
            self._http = HttpClient(rate_per_sec=0.5, burst=1.0)

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self.on_tick)
        if self._http is None:
            log.warning("Zillow feed disabled by config")
            return
        log.info("Zillow feed online (poll=%ss)", self.poll_interval_s)

    def on_tick(self, ev) -> None:
        if not self._running or self._http is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            return
        now = self.clock.now()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now
        self._poll()

    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        self._poll_zori()
        self._poll_manheim()

    def _poll_zori(self) -> None:
        try:
            resp = self._http.get(self.metro_url, timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            log.debug("ZORI fetch failed: %s", exc)
            self.metrics.inc("zori_fetch_failed")
            return
        if resp.status_code != 200:
            self.metrics.inc("zori_fetch_failed")
            return
        points = parse_zori_csv(resp.text, region=self.us_region)
        if not points:
            log.debug("ZORI: no rows for region %s", self.us_region)
            return
        yoy = zori_yoy(points)
        if yoy is None:
            return
        prev = self._last_values.get("ZORI_YOY")
        if prev is not None and abs(prev - yoy) < 1e-6:
            return
        self._last_values["ZORI_YOY"] = yoy
        now = self.clock.now()
        self.emit(make_event(
            EventType.NEWS, "data.zillow",
            payload={
                "id": f"zori:{points[-1].date}",
                "ts": now,
                "headline": f"Zillow ZORI YoY: {yoy:.2f}% ({points[-1].date})",
                "source": "ZILLOW",
                "release_id": "ZORI_YOY",
                "actual": yoy,
                "date": points[-1].date,
                "units": "percent",
                "raw_value": points[-1].value,
            },
            timestamp=now,
        ))
        self.metrics.inc("zori_observation")

    def _poll_manheim(self) -> None:
        # Primary: scrape Cox's public research page.
        value: float | None = None
        try:
            resp = self._http.get(self.manheim_url, timeout=10.0)
            if resp.status_code == 200:
                value = parse_manheim_html(resp.text)
        except Exception as exc:  # noqa: BLE001
            log.debug("Manheim scrape failed: %s", exc)

        source = "manheim"
        if value is None:
            # Fallback: FRED's used cars CPI as a correlated proxy.
            value = self._fetch_fred_used_cars_proxy()
            source = "fred_proxy"

        if value is None:
            self.metrics.inc("manheim_fetch_failed")
            return
        prev = self._last_values.get("MANHEIM_UVVI")
        if prev is not None and abs(prev - value) < 1e-6:
            return
        self._last_values["MANHEIM_UVVI"] = value
        now = self.clock.now()
        self.emit(make_event(
            EventType.NEWS, "data.zillow",
            payload={
                "id": f"manheim:{int(now)}",
                "ts": now,
                "headline": f"Manheim UVVI ({source}): {value}",
                "source": "MANHEIM" if source == "manheim" else "FRED_PROXY",
                "release_id": "MANHEIM_UVVI",
                "actual": value,
                "units": "index",
                "proxy": source == "fred_proxy",
            },
            timestamp=now,
        ))
        self.metrics.inc("manheim_observation", source=source)

    def _fetch_fred_used_cars_proxy(self) -> float | None:
        """
        Pull CUSR0000SETA02 (CPI used cars and trucks) from the public FRED
        CSV endpoint — no key required for the downloader.
        """
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               "?id=CUSR0000SETA02")
        try:
            resp = self._http.get(url, timeout=10.0)
            if resp.status_code != 200:
                return None
        except Exception:  # noqa: BLE001
            return None
        lines = [line for line in resp.text.splitlines() if line and not line.startswith("DATE")]
        if len(lines) < 2:
            return None
        last, prev = lines[-1], lines[-2]
        try:
            v_last = float(last.split(",")[1])
            v_prev = float(prev.split(",")[1])
        except (IndexError, ValueError):
            return None
        if v_prev == 0:
            return None
        # Return MoM % change as the proxy signal.
        return (v_last - v_prev) / v_prev * 100.0

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
