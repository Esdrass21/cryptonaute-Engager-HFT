# data_ingestion/news_feeds/calendar.py
"""
Release calendar for US macro data.

Sources (in priority order):
1. Explicit config override — `data.calendar.overrides`
2. Cached scrape of public schedule pages (BLS, Fed, DOL, Treasury, Census)
3. Deterministic rules (e.g. "NFP = first Friday of the month at 08:30 ET")

Timezone: all US macro releases are anchored to America/New_York, and the
calendar exposes UTC timestamps for the rest of the platform.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("data.calendar")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScheduledRelease:
    release_id: str                 # e.g. "CPI_YOY", "NFP", "FOMC_RATE"
    source: str                     # "BLS" | "FED" | "DOL" | "CENSUS" | "BEA"
    scheduled_ts: float             # UTC epoch seconds
    consensus: Optional[float] = None
    prior: Optional[float] = None
    units: str = ""
    url: str = ""                   # direct release URL (if known)

    def to_payload(self) -> dict:
        return {
            "release_id": self.release_id,
            "source": self.source,
            "scheduled_ts": self.scheduled_ts,
            "consensus": self.consensus,
            "prior": self.prior,
            "units": self.units,
            "url": self.url,
        }


# --------------------------------------------------------------------------- #
# Rules                                                                        #
# --------------------------------------------------------------------------- #

def _et(y: int, m: int, d: int, hh: int, mm: int) -> float:
    """Return UTC epoch seconds for an America/New_York wall-clock time."""
    dt = datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(UTC)
    return dt.timestamp()


def _first_weekday(y: int, m: int, weekday: int) -> int:
    """Day-of-month of the first `weekday` (Mon=0) in the given month."""
    d = datetime(y, m, 1)
    offset = (weekday - d.weekday()) % 7
    return 1 + offset


def _third_wednesday(y: int, m: int) -> int:
    first = _first_weekday(y, m, 2)   # Wednesday
    return first + 14


def _nth_weekday(y: int, m: int, nth: int, weekday: int) -> int:
    """Day-of-month of the nth weekday (Mon=0) in a month. nth is 1-indexed."""
    first = _first_weekday(y, m, weekday)
    return first + (nth - 1) * 7


def _months_between(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield (y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


def _fomc_dates_2025_2026() -> List[tuple[int, int, int]]:
    """
    FOMC meeting decision dates (second day), hard-coded from the Fed's
    published calendar. Update annually from
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    """
    return [
        # 2025
        (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
        (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
        # 2026
        (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
        (2026, 7, 29), (2026, 9, 16), (2026, 11, 4), (2026, 12, 16),
    ]


# --------------------------------------------------------------------------- #
# Calendar                                                                     #
# --------------------------------------------------------------------------- #

class ReleaseCalendar:
    """
    Enumerates scheduled US macro releases over a time window.

    Deterministic in (window_start, window_end). No network required for the
    rule-based path; the scraper is optional and only runs when enabled.
    """

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._overrides: Dict[str, ScheduledRelease] = {}
        if cfg is not None:
            for o in cfg.get("data.calendar.overrides", []) or []:
                try:
                    self._overrides[o["release_id"]] = ScheduledRelease(
                        release_id=o["release_id"],
                        source=o.get("source", "override"),
                        scheduled_ts=float(o["scheduled_ts"]),
                        consensus=o.get("consensus"),
                        prior=o.get("prior"),
                        units=o.get("units", ""),
                        url=o.get("url", ""),
                    )
                except KeyError as exc:
                    log.warning("calendar override malformed: %s", exc)

    # ------------------------------------------------------------------ #
    def releases_between(self, start_ts: float, end_ts: float) -> List[ScheduledRelease]:
        """Return all scheduled releases in [start_ts, end_ts], sorted by time."""
        out: List[ScheduledRelease] = list(self._overrides.values())

        start = datetime.fromtimestamp(start_ts, tz=UTC).astimezone(ET)
        end = datetime.fromtimestamp(end_ts, tz=UTC).astimezone(ET)

        # iterate months in the window
        for (y, m) in _months_between(start, end):
            out.extend(self._rules_for_month(y, m))

        # dedupe by release_id + scheduled_ts, filter to window, sort
        seen = set()
        filtered: List[ScheduledRelease] = []
        for r in out:
            key = (r.release_id, r.scheduled_ts)
            if key in seen:
                continue
            if start_ts <= r.scheduled_ts <= end_ts:
                seen.add(key)
                filtered.append(r)
        filtered.sort(key=lambda r: r.scheduled_ts)
        return filtered

    # ------------------------------------------------------------------ #
    def _rules_for_month(self, y: int, m: int) -> List[ScheduledRelease]:
        rules: List[ScheduledRelease] = []

        # NFP — first Friday of the month, 08:30 ET
        nfp_day = _first_weekday(y, m, 4)  # Friday
        nfp_ts = _et(y, m, nfp_day, 8, 30)
        rules.append(ScheduledRelease(
            release_id="NFP", source="BLS", scheduled_ts=nfp_ts,
            units="thousands", url="https://www.bls.gov/news.release/empsit.nr0.htm",
        ))

        # Unemployment rate — same release
        rules.append(ScheduledRelease(
            release_id="UNEMPLOYMENT_RATE", source="BLS", scheduled_ts=nfp_ts,
            units="percent",
        ))

        # CPI — usually around the 10th–14th. Rule: second Tuesday unless it falls
        # before the 10th, in which case the following Tuesday. Approximation;
        # real dates get replaced by the scraper when available.
        cpi_day = _nth_weekday(y, m, 2, 2)  # second Tuesday
        if cpi_day < 10:
            cpi_day = _nth_weekday(y, m, 3, 2)  # third Tuesday
        cpi_ts = _et(y, m, cpi_day, 8, 30)
        rules.append(ScheduledRelease(
            release_id="CPI_YOY", source="BLS", scheduled_ts=cpi_ts,
            units="percent", url="https://www.bls.gov/news.release/cpi.nr0.htm",
        ))
        rules.append(ScheduledRelease(
            release_id="CPI_MOM", source="BLS", scheduled_ts=cpi_ts, units="percent",
        ))
        rules.append(ScheduledRelease(
            release_id="CORE_CPI_YOY", source="BLS", scheduled_ts=cpi_ts, units="percent",
        ))

        # PPI — day after CPI at 08:30 ET
        ppi_ts = cpi_ts + 24 * 3600
        rules.append(ScheduledRelease(
            release_id="PPI_YOY", source="BLS", scheduled_ts=ppi_ts,
            units="percent", url="https://www.bls.gov/news.release/ppi.nr0.htm",
        ))

        # Jobless claims — every Thursday at 08:30 ET
        for day in range(1, 32):
            try:
                d = datetime(y, m, day)
            except ValueError:
                break
            if d.weekday() == 3:  # Thursday
                rules.append(ScheduledRelease(
                    release_id="JOBLESS_CLAIMS", source="DOL",
                    scheduled_ts=_et(y, m, day, 8, 30),
                    units="thousands",
                    url="https://www.dol.gov/ui/data.pdf",
                ))

        # Retail sales — mid-month, ~15th-17th, 08:30 ET
        rs_day = _nth_weekday(y, m, 3, 2)  # third Tuesday approximation
        rules.append(ScheduledRelease(
            release_id="RETAIL_SALES_MOM", source="CENSUS",
            scheduled_ts=_et(y, m, rs_day, 8, 30),
            units="percent",
        ))

        # PCE — end of month, 08:30 ET
        pce_day = 28
        rules.append(ScheduledRelease(
            release_id="PCE_YOY", source="BEA",
            scheduled_ts=_et(y, m, pce_day, 8, 30),
            units="percent",
        ))

        # FOMC — hard-coded dates
        for (fy, fm, fd) in _fomc_dates_2025_2026():
            if fy == y and fm == m:
                # decision at 14:00 ET, press conf at 14:30
                rules.append(ScheduledRelease(
                    release_id="FOMC_RATE", source="FED",
                    scheduled_ts=_et(y, m, fd, 14, 0),
                    units="percent",
                    url="https://www.federalreserve.gov/newsevents/pressreleases/monetary{}.htm".format(
                        datetime(y, m, fd).strftime("%Y%m%d")),
                ))
                rules.append(ScheduledRelease(
                    release_id="FOMC_STATEMENT", source="FED",
                    scheduled_ts=_et(y, m, fd, 14, 0),
                ))

        return rules
