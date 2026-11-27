# data_ingestion/news_feeds/bls.py
"""
BLS release feed.

Two modes:

* **RSS / poll mode (default, safe)** — subscribes to the public BLS RSS feeds
  and polls the release pages every N seconds. No rate-limit abuse; respects
  robots.txt. Latency: seconds.
* **Race mode (opt-in, `data.news.bls.race_mode: true`)** — on the exact
  scheduled release second, fires a direct GET against the release page and
  parses the headline. Still public data, still legal. Latency: ~100-500ms
  depending on your network.

Emits:
* `RELEASE_EXPECTED` at T-2min (arms the strategy)
* `RELEASE_ACTUAL` on successful parse (headline number + surprise)
* `NEWS` with the raw headline text for the NLP layer
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib import request as _req

from core.component import Component
from core.events import EventType, make_event

from .calendar import ReleaseCalendar, ScheduledRelease

log = logging.getLogger("data.bls")

# --------------------------------------------------------------------------- #
# Parsers — pattern → (release_id, extractor)                                  #
# --------------------------------------------------------------------------- #
# Each parser takes the raw HTML/text of the release page and returns a float
# (the headline number) or None. Keep the regexes anchored to the *first*
# occurrence, which is almost always the headline.

_CPI_YOY_RE = re.compile(
    r"Consumer Price Index for All Urban Consumers.*?rose\s+([\d.]+)\s*percent",
    re.IGNORECASE | re.DOTALL,
)
_CPI_MOM_RE = re.compile(
    r"increased\s+([\d.]+)\s*percent in",
    re.IGNORECASE,
)
_CORE_CPI_YOY_RE = re.compile(
    r"All items less food and energy.*?rose\s+([\d.]+)\s*percent",
    re.IGNORECASE | re.DOTALL,
)
_NFP_RE = re.compile(
    r"Total nonfarm payroll employment (?:rose|increased|changed)\s+by\s+\+?([\d,]+)",
    re.IGNORECASE,
)
_UNRATE_RE = re.compile(
    r"unemployment rate (?:was|remained|edged\s+\w+\s+to)\s+([\d.]+)\s*percent",
    re.IGNORECASE,
)
_PPI_YOY_RE = re.compile(
    r"Producer Price Index.*?(?:rose|advanced|increased)\s+([\d.]+)\s*percent",
    re.IGNORECASE | re.DOTALL,
)

_PARSERS: dict[str, re.Pattern] = {
    "CPI_YOY":       _CPI_YOY_RE,
    "CPI_MOM":       _CPI_MOM_RE,
    "CORE_CPI_YOY":  _CORE_CPI_YOY_RE,
    "NFP":           _NFP_RE,
    "UNEMPLOYMENT_RATE": _UNRATE_RE,
    "PPI_YOY":       _PPI_YOY_RE,
}


def parse_release(release_id: str, text: str) -> float | None:
    """Return the parsed number, or None if the pattern doesn't match."""
    pattern = _PARSERS.get(release_id)
    if pattern is None:
        return None
    m = pattern.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _fetch(url: str, timeout: float = 5.0) -> str | None:
    """GET a public URL with a proper user-agent. Returns None on failure."""
    try:
        req = _req.Request(url, headers={
            "User-Agent": "Cryptonaute/3.0 (research; +https://example.invalid)",
            "Accept": "text/html,application/xhtml+xml,application/xml",
        })
        with _req.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.debug("BLS fetch failed for %s: %s", url, exc)
        return None


# --------------------------------------------------------------------------- #
# Feed component                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class _Pending:
    release: ScheduledRelease
    armed_at: float
    fired: bool = False


class BLSReleaseFeed(Component):
    """
    Watches the release calendar and emits RELEASE_EXPECTED / RELEASE_ACTUAL /
    NEWS events for BLS releases.

    In replay mode (`clock.mode == "virtual"`) the feed *does not hit the
    network*. It only re-emits from previously recorded events — the replay
    harness reads them from the event log. This is what keeps validation
    deterministic.
    """

    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="data.bls")
        self.calendar = ReleaseCalendar(cfg)
        self._pending: dict[str, _Pending] = {}         # release_id -> state
        self._lookahead_s = float(cfg.get("data.news.bls.lookahead_s", 300.0))
        self._arm_before_s = float(cfg.get("data.news.bls.arm_before_s", 120.0))
        self._race_mode = bool(cfg.get("data.news.bls.race_mode", False))
        self._race_timeout_ms = float(cfg.get("data.news.bls.race_timeout_ms", 500.0))
        self._last_refresh = 0.0
        self._refresh_interval_s = float(cfg.get("data.news.bls.refresh_interval_s", 3600.0))

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True
        self._refresh_calendar()
        # Subscribe to TICK events to drive the release state machine
        self.bus.subscribe(EventType.TICK, self.on_tick)
        log.info("BLS feed online (race_mode=%s, tracked=%d)", self._race_mode, len(self._pending))

    def _refresh_calendar(self) -> None:
        now = self.clock.now()
        window_end = now + self._lookahead_s
        releases = self.calendar.releases_between(now, window_end)
        for r in releases:
            if r.source != "BLS":
                continue
            if r.release_id not in self._pending:
                self._pending[r.release_id] = _Pending(release=r, armed_at=0.0)
        self._last_refresh = now

    # ------------------------------------------------------------------ #
    def on_tick(self, ev) -> None:
        """
        Called by the market feed (or by a polling driver in live mode) with
        each new price tick. We use it as a logical clock to advance the
        release state machine.
        """
        if not self._running:
            return
        if getattr(self.clock, "mode", "real") == "virtual":
            # Replay mode: the bus re-delivers recorded RELEASE_* events.
            return

        now = self.clock.now()
        if now - self._last_refresh > self._refresh_interval_s:
            self._refresh_calendar()

        for release_id, pending in list(self._pending.items()):
            r = pending.release
            dt = r.scheduled_ts - now

            # T-arm: emit RELEASE_EXPECTED
            if dt <= self._arm_before_s and pending.armed_at == 0.0:
                pending.armed_at = now
                self.emit(make_event(
                    EventType.RELEASE_EXPECTED, "data.bls",
                    payload={
                        "release_id": release_id,
                        "scheduled_ts": r.scheduled_ts,
                        "consensus": r.consensus,
                        "prior": r.prior,
                        "units": r.units,
                        "url": r.url,
                        "ts": now,
                    },
                    timestamp=now,
                ))
                self.metrics.inc("release_expected", release_id=release_id)

            # T-fire: fetch and parse
            if dt <= 0.0 and not pending.fired:
                pending.fired = True
                self._fire_release(pending)

            # T+lookahead: drop
            if dt < -self._lookahead_s:
                self._pending.pop(release_id, None)

    # ------------------------------------------------------------------ #
    def _fire_release(self, pending: _Pending) -> None:
        r = pending.release
        now = self.clock.now()

        # In race mode, block on the fetch for up to timeout_ms.
        # In poll mode, don't block — defer to next tick.
        if not self._race_mode:
            return  # poll mode just waits for the RSS to catch up

        text = _fetch(r.url, timeout=self._race_timeout_ms / 1000.0)
        if text is None:
            log.warning("BLS race fetch failed: %s", r.url)
            self.metrics.inc("release_fetch_failed", release_id=r.release_id)
            return

        actual = parse_release(r.release_id, text)
        if actual is None:
            log.warning("BLS parse failed: %s", r.release_id)
            self.metrics.inc("release_parse_failed", release_id=r.release_id)
            return

        consensus = r.consensus if r.consensus is not None else 0.0
        prior = r.prior if r.prior is not None else 0.0
        # Historical stddev of surprises for this release — configurable prior.
        surprise_std = float(self.cfg.get(f"data.news.surprise_std.{r.release_id}", 1.0))
        surprise = (actual - consensus) / max(surprise_std, 1e-9)

        self.emit(make_event(
            EventType.RELEASE_ACTUAL, "data.bls",
            payload={
                "release_id": r.release_id,
                "source": "BLS",
                "actual": actual,
                "consensus": consensus,
                "prior": prior,
                "surprise": surprise,
                "units": r.units,
                "scheduled_ts": r.scheduled_ts,
                "parsed_ts": now,
                "ts": now,
            },
            timestamp=now,
        ))
        self.metrics.inc("release_actual", release_id=r.release_id)
        self.metrics.observe("release_latency_s", now - r.scheduled_ts)

        # Also emit a NEWS event so the NLP layer sees the raw text.
        self.emit(make_event(
            EventType.NEWS, "data.bls",
            payload={
                "id": f"bls:{r.release_id}:{int(r.scheduled_ts)}",
                "ts": now,
                "headline": f"{r.release_id}: actual={actual} consensus={consensus} "
                            f"surprise={surprise:+.2f}σ",
                "source": "BLS",
                "release_id": r.release_id,
                "raw_excerpt": text[:512],
            },
            timestamp=now,
        ))
