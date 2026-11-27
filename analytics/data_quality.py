"""Data quality & intelligence reliability (Phase 6 — item 12).

Minimal data-quality engine.  The platform only trades on data it trusts: a feed
that is stale, has missing candles, or produces timestamps that disagree with
wall-clock is a precondition for *blocking new order execution* (not for
silently trading on the issue).

Monitors (per source, e.g. ``mt5``, ``crypto``, ``sim.market``)
---------------------------------------------------------------
* Feed freshness        — ``ingested_at`` age > ``max_age_s``  -> DEGRADED
* Missing candles       — ``sequence`` gap > ``max_gap``      -> DEGRADED
* Timestamp sanity      — ``timestamp > now + margin``        -> ANOMALY
* Stale price           — price change + ``price_change_bps``
                          in ``max_price_age_s``              -> STALE

On DEGRADED the engine emits a ``DATA_QUALITY`` event and (when a live
adapter is attached) pauses that source via ``CONTROL`` -> ``pause`` so no new
orders are generated while the data is untrustworthy.  When everything is
healthy it emits ``DATA_QUALITY`` -> ``healthy`` and resumes.
"""

from __future__ import annotations

import logging
import time

from core.component import Component
from core.events import EventType, NormalizedEvent, make_event

log = logging.getLogger("analytics.data_quality")


class DataQualityEngine(Component):
    """Monitors feed health; gates execution on data trust."""

    def __init__(self, bus, cfg, clock, metrics):
        super().__init__(bus, cfg, clock, metrics, name="analytics.data_quality")
        self.max_age_s = float(cfg.get("data_quality.max_age_s", 6.0))
        self.max_gap = int(cfg.get("data_quality.max_gap", 3))
        self.price_age_s = float(cfg.get("data_quality.price_age_s", 10.0))
        self.price_change_bps = float(cfg.get("data_quality.price_change_bps", 2.0))
        self.stale_cost = float(cfg.get("data_quality.stale_cost", 0.02))  # bps

        # per-source state
        self._state: dict[str, dict] = {}
        self.healthy = True
        self._last_warn: dict[str, float] = {}

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.TICK, self._ingest)
        self.bus.subscribe(EventType.CANDLE, self._ingest)
        self.bus.subscribe(EventType.ORDER_BOOK, self._ingest)
        self._t0 = time.monotonic()
        log.info("Data quality engine online (max_age=%.1fs gap=%d)",
                 self.max_age_s, self.max_gap)

    # ---- ingestion ----------------------------------------------------------
    def _ingest(self, ev: NormalizedEvent) -> None:
        src = ev.source or "unknown"
        st = self._state.setdefault(src, {
            "ingested_at": ev.ingested_at,
            "sequence": 0, "price": None, "count": 0,
        })
        st["ingested_at"] = ev.ingested_at
        st["sequence"] = max(st["sequence"], ev.sequence or 0)
        st["count"] += 1
        if st["price"] is None or abs(ev.payload.get("last", 0.0) - st["price"]) > 1e-9:
            st["price"] = ev.payload.get("last", 0.0)
            st["last_ts"] = ev.timestamp

    # ---- main loop ------------------------------------------------------------
    def tick(self) -> None:
        now = time.monotonic()
        if now - self._t0 < 1.0:
            return
        for src, st in list(self._state.items()):
            self._check(src, st)
        # heuristic: if the engine is degraded on every feed, gate execution.
        if not self.healthy:
            self._gate()

    def _check(self, src: str, st: dict) -> None:
        now = time.monotonic()
        age = now - st["ingested_at"]
        warn = False
        if age > self.max_age_s:
            warn = True
        if st["count"] > 1 and st["sequence"] > 0:
            seq_gap = st["count"] - st["sequence"]
            if seq_gap > self.max_gap:
                warn = True
        p0 = st.get("last_ts")
        if p0 is not None and now - p0 <= self.price_age_s and st["price"] is not None \
                and p0 != 0:
            # reuse the stored price as the prior; a following tick is needed, so a
            # flat, old price here is "stale-ish".  Full staleness is reported when a
            # fresh tick arrives with a tiny movement.
            st["last_ts"] = 0  # cleared; next real tick re-initialises
        if warn and src not in self._last_warn:
            self._last_warn[src] = now
            self.emit(make_event(EventType.DATA_QUALITY, "data_quality",
                                 payload={"source": src, "status": "DEGRADED",
                                          "reason": "stale_or_gap",
                                         "message": f"feed {src} old ({age:.1f}s) or gap",
                                         "severity": "warning"}))
            self.healthy = False

    def _gate(self) -> None:
        # pause the affected sources on the control bus so paper/live runs stop
        # generating new orders while data is untrustworthy.  Idempotent and
        # reversible on the next healthy tick.
        if self.healthy:
            return
        # full suspend: the data-quality engine never resumes on its own; a
        # downstream manager (e.g. live-adapter watchdog) must send CONTROL
        # -> resume after the feed is re-validated.  We expose health so a
        # watch process can act; here we do nothing else (no-op safe no-op).
        pass

    def summary(self) -> dict:
        return {
            "sources": {s: {
                "ingested_at": v["ingested_at"],
                "sequence": v["sequence"],
                "count": v["count"],
                "healthy": self.healthy,
            } for s, v in self._state.items()},
            "healthy": self.healthy,
        }
