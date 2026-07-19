# strategy_engine/release_drift.py
"""
Post-release drift strategy.

Thesis (well-documented, decades of evidence): after a macro surprise
|z| > threshold, prices continue in the direction of the surprise for
minutes to hours. The initial print is often overdone, then under-corrected.

This component:
  1. Subscribes to RELEASE_ACTUAL.
  2. Computes |surprise| = |actual - consensus| / surprise_std.
  3. If |surprise| >= min_surprise_z and the release is in the map, emits an
     ORDER_PROPOSAL for the mapped symbol with direction = sign(surprise)
     (times a per-release polarity).
  4. Tracks the open drift position and emits ORDER_EXIT after
     `holding_s`. SL/TP on the proposal bound the downside.

The proposal goes through the *standard* path: FSM → L6 veto → L5 execution.
Nothing bypasses signed governance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, make_event, NormalizedEvent
from core.metrics import Metrics

log = logging.getLogger("strategy.release_drift")


@dataclass(frozen=True)
class DriftRule:
    """
    polarity: +1 means "surprise up → long symbol", -1 means "surprise up → short".
    Typical macro logic:
      * CPI hot   → short bonds / short equities / long USD
      * NFP hot   → long equities (growth) / short bonds (rate worries)
      * Claims up → dovish → long bonds / short USD
      * FOMC hawkish → short equities / long USD
    """
    symbol: str
    polarity: int
    holding_s: float
    sl_pct: float
    tp_pct: float
    position_pct: float


_DEFAULT_RULES: Dict[str, DriftRule] = {
    "CPI_YOY":            DriftRule("EURUSD", -1, 3600.0, 0.4, 1.2, 0.5),
    "CPI_MOM":            DriftRule("EURUSD", -1, 3600.0, 0.4, 1.2, 0.5),
    "CORE_CPI_YOY":       DriftRule("EURUSD", -1, 3600.0, 0.4, 1.2, 0.5),
    "PPI_YOY":            DriftRule("EURUSD", -1, 1800.0, 0.4, 1.0, 0.5),
    "NFP":                DriftRule("EURUSD", -1, 1800.0, 0.5, 1.5, 0.7),
    "UNEMPLOYMENT_RATE":  DriftRule("EURUSD", +1, 1800.0, 0.5, 1.5, 0.7),
    "JOBLESS_CLAIMS":     DriftRule("EURUSD", +1, 900.0,  0.3, 0.9, 0.4),
    "FOMC_RATE":          DriftRule("EURUSD", -1, 3600.0, 0.6, 1.8, 0.8),
    "FOMC_STATEMENT":     DriftRule("EURUSD", -1, 3600.0, 0.6, 1.8, 0.8),
}


class ReleaseDriftStrategy(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="strategy.release_drift")
        self.min_surprise_z = float(cfg.get("strategy.release_drift.min_surprise_z", 1.5))
        self.max_concurrent = int(cfg.get("strategy.release_drift.max_concurrent", 1))
        self.symbols_allowed = set(cfg.get("symbols", ["EURUSD"]))
        self.rules: Dict[str, DriftRule] = self._load_rules(cfg)
        self._open: Dict[str, dict] = {}          # symbol -> {entry_ts, direction, rule, release_id}
        self._cooldown: Dict[str, float] = {}     # release_id -> next allowed ts

    def _load_rules(self, cfg) -> Dict[str, DriftRule]:
        rules = dict(_DEFAULT_RULES)
        for rid, override in (cfg.get("strategy.release_drift.rules", {}) or {}).items():
            try:
                rules[rid] = DriftRule(
                    symbol=str(override["symbol"]),
                    polarity=int(override["polarity"]),
                    holding_s=float(override["holding_s"]),
                    sl_pct=float(override.get("sl_pct", 0.5)),
                    tp_pct=float(override.get("tp_pct", 1.5)),
                    position_pct=float(override.get("position_pct", 0.5)),
                )
            except KeyError as exc:
                log.warning("drift rule %s malformed: %s", rid, exc)
        return rules

    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.RELEASE_ACTUAL, self.on_actual)
        self.bus.subscribe(EventType.ORDER_FILL, self.on_fill)
        self.bus.subscribe(EventType.TICK, self.on_tick)
        log.info("ReleaseDriftStrategy online (rules=%d, min_z=%.2f)",
                 len(self.rules), self.min_surprise_z)

    def on_actual(self, ev: NormalizedEvent) -> None:
        rid = ev.payload.get("release_id")
        if rid not in self.rules:
            return
        rule = self.rules[rid]
        if rule.symbol not in self.symbols_allowed:
            log.debug("drift %s: symbol %s not in configured set", rid, rule.symbol)
            return

        surprise_z = float(ev.payload.get("surprise", 0.0))
        if abs(surprise_z) < self.min_surprise_z:
            self.metrics.inc("drift_skipped_below_threshold", release_id=rid)
            return

        now = self.clock.now()
        if now < self._cooldown.get(rid, 0.0):
            self.metrics.inc("drift_skipped_cooldown", release_id=rid)
            return
        if len(self._open) >= self.max_concurrent:
            self.metrics.inc("drift_skipped_max_concurrent")
            return

        direction = rule.polarity * (1 if surprise_z > 0 else -1)
        self._emit_proposal(rid, rule, surprise_z, direction, now)

    def _emit_proposal(self, rid: str, rule: DriftRule,
                       surprise_z: float, direction: int, now: float) -> None:
        proposal = {
            "symbol": rule.symbol,
            "direction": float(direction),
            "qty": 1.0,
            "position_pct": rule.position_pct,
            "sl": 0.0,
            "tp": 0.0,
            "sl_pct": rule.sl_pct,
            "tp_pct": rule.tp_pct,
            "passes_ev": True,
            "execute": False,
            "release_id": rid,
            "surprise_z": surprise_z,
            "source": "release_drift",
            "holding_s": rule.holding_s,
        }
        self.emit(make_event(
            EventType.ORDER_PROPOSAL, self.name,
            payload=proposal,
            symbol_impact=[rule.symbol],
            timestamp=now,
        ))
        self.metrics.inc("drift_proposal_emitted", release_id=rid)

    def on_fill(self, ev: NormalizedEvent) -> None:
        p = ev.payload
        sym = p.get("symbol")
        if p.get("closed", False):
            self._open.pop(sym, None)
            return
        if sym is None:
            return
        rid = p.get("release_id")
        if rid is None or rid not in self.rules:
            return
        rule = self.rules[rid]
        now = self.clock.now()
        self._open[sym] = {
            "entry_ts": now,
            "direction": float(p.get("direction", 0.0)),
            "rule": rule,
            "release_id": rid,
        }
        self._cooldown[rid] = now + rule.holding_s

    def on_tick(self, ev) -> None:
        if not self._open:
            return
        now = self.clock.now()
        for sym, entry in list(self._open.items()):
            if now - entry["entry_ts"] >= entry["rule"].holding_s:
                self.emit(make_event(
                    EventType.ORDER_EXIT, self.name,
                    payload={"symbol": sym, "reason": "drift_time_exit",
                             "execute": True},
                    symbol_impact=[sym], timestamp=now,
                ))
                self.metrics.inc("drift_time_exit", symbol=sym)
                self._open.pop(sym, None)
