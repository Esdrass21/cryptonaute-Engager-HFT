"""Strategy Router + Shadow Mode — Layer 4b (routing) & simulation mode.

The Strategy Router is a higher-level orchestrator that:
1. Subscribes to ORDER_PROPOSAL events from multiple strategies (TSMOM, noise_band, etc.)
2. Combines/routes them into a single unified proposal per symbol
3. Emits a combined ORDER_PROPOSAL for the FSM + execution layer

Shadow Mode is a simulation/paper mode:
- When enabled, all ORDER_PROPOSAL events are logged to a shadow log
  and sent to a "shadow execution" sink — no real broker orders are placed.
- When disabled (default), normal live execution proceeds.

This component is deterministic, fast, and reusable in both live and replay paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.bus import MessageBus
from core.component import Component
from core.config import Clock, Config
from core.events import EventType, NormalizedEvent, make_event
from core.metrics import Metrics

log = logging.getLogger("strategy.router")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_SHADOW_CFG = {
    "enabled": False,           # false = live execution, true = shadow simulation
    "shadow_log": "data/shadow_proposals.jsonl",  # where shadow orders are logged
    "combine_mode": "priority", # priority | weighted | first_wins
    "priority_strategy_order": ["tsmom", "noise_band"],  # order of precedence
    "max_concurrent_proposals": 1,   # cap on simultaneous proposals per symbol
}

# ---------------------------------------------------------------------------
# Strategy Router
# ---------------------------------------------------------------------------
class StrategyRouter(Component):
    """Routes strategy ORDER_PROPOSAL events to the FSM + execution layer.

    The router subscribes to ORDER_PROPOSAL events from all enabled strategies
    (TSMOM, noise_band, ...) and combines them per symbol using one of three
    combine modes:

      * ``priority``      — the highest-priority strategy's proposal wins
      * ``weighted``      — signals are combined by configurable weights
      * ``first_wins``    — the first proposal received (by timestamp) wins

    In shadow mode, proposals are logged to ``shadow_log`` and a dry-run
    ``ORDER_PROPOSAL`` (execute=False) is emitted for the FSM — no live orders.
    """

    def __init__(
        self,
        bus: MessageBus,
        cfg: Config,
        clock: Clock,
        metrics: Metrics,
        shadow_enabled: bool | None = None,
    ) -> None:
        super().__init__(bus, cfg, clock, metrics, name="strategy.router")

        sec = "strategy.router"
        self.combine_mode = cfg.get(f"{sec}.combine_mode", "priority")
        self.priority_order = list(
            cfg.get(f"{sec}.priority_strategy_order", ["tsmom", "noise_band"])
        )
        self.max_concurrent = int(cfg.get(f"{sec}.max_concurrent_proposals", 1))
        self.shadow_enabled = bool(
            shadow_enabled if shadow_enabled is not None else cfg.get("strategy.shadow.enabled", False)
        )
        self.shadow_log_path = cfg.get(f"{sec}.shadow_log", "data/shadow_proposals.jsonl")

        # Per-symbol proposal queue: symbol -> list of (timestamp, payload)
        self._queue: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        self._last_shadow_seq = 0

        # Open shadow log file (append mode)
        self._shadow_fh = Path(self.shadow_log_path)
        self._shadow_fh.parent.mkdir(parents=True, exist_ok=True)
        self._shadow_fh.touch(exist_ok=True)

        log.info(
            "StrategyRouter online (combine=%s, shadow=%s, shadow_log=%s)",
            self.combine_mode, self.shadow_enabled, self.shadow_log_path,
        )

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.ORDER_PROPOSAL, self._ingest_proposal)
        log.info("StrategyRouter subscribed to ORDER_PROPOSAL")

    # -- event handlers ----------------------------------------------------
    def _ingest_proposal(self, ev: NormalizedEvent) -> None:
        sym = ev.payload.get("symbol")
        if sym is None:
            return

        # Skip proposals from this router itself
        if ev.payload.get("source") == self.name:
            return

        # Register source name on the payload for routing clarity
        ev = NormalizedEvent(
            event_id=ev.event_id,
            timestamp=ev.timestamp,
            source=self.name,
            type=ev.type,
            payload=dict(ev.payload) | {"source": self.name},
            symbol_impact=list(ev.symbol_impact),
            sequence=ev.sequence,
        )

        # Enqueue
        now = self.clock.now()
        self._queue.setdefault(sym, []).append((now, ev.payload))

        # Drain the queue per symbol
        self._drain_symbol(sym, now)

    def _drain_symbol(self, sym: str, now: float) -> None:
        queue = self._queue.get(sym, [])
        if not queue:
            return

        # Respect concurrency cap
        if len(queue) > self.max_concurrent:
            # Keep the most recent proposals and drop older ones
            queue.sort(key=lambda t: t[0], reverse=True)
            self._queue[sym] = queue[: self.max_concurrent]
            queue = self._queue[sym]

        # Combine proposals
        proposals = [p for _, p in queue]
        combined = self._combine(proposals, sym)

        # Shadow mode: log + emit non-executing proposal
        if self.shadow_enabled:
            self._write_shadow(combined, sym, now)
            # In shadow mode we still emit to the FSM so it can validate the logic,
            # but the execution layer will not act on it (execute=False).
            self._emit_combined(sym, combined, now, execute=False)
        else:
            # Live mode: emit with execute flag as-is (routed proposal carries it)
            self._emit_combined(sym, combined, now, execute=bool(combined.get("execute", False)))

        # Clear queue for this symbol (approaches should be eager)
        queue.clear()

    # -- combine strategy ---------------------------------------------------
    def _combine(self, proposals: list[dict[str, Any]], sym: str) -> dict[str, Any]:
        if not proposals:
            return {"symbol": sym, "direction": 0.0, "passes_ev": False, "execute": False, "source": "router", "timestamp": self.clock.now()}

        mode = self.combine_mode
        if mode == "priority":
            # Highest priority strategy wins
            for name in self.priority_order:
                for p in proposals:
                    if p.get("source") == name:
                        return self._annotate(p, sym)
            # Fall back to first
            return self._annotate(proposals[0], sym)

        if mode == "first_wins":
            return self._annotate(proposals[0], sym)

        if mode == "weighted":
            return self._weighted_combine(proposals, sym)

        log.warning("Unknown combine_mode=%s, falling back to priority", mode)
        return self._combine(proposals, sym, mode="priority")

    def _weighted_combine(self, proposals: list[dict[str, Any]], sym: str) -> dict[str, Any]:
        """Weighted blend of direction/confidence across sources."""
        weights = {}
        for name in self.priority_order:
            weights[name] = 1.0 / (self.priority_order.index(name) + 1)
        total_w = sum(weights.values()) or 1.0

        direction = 0.0
        confidence = 0.0
        ev_sum = 0.0
        n = 0
        for p in proposals:
            src = p.get("source", "unknown")
            w = weights.get(src, 1.0)
            d = float(p.get("direction", 0.0) or 0.0)
            c = float(p.get("confidence", 0.0) or 0.0)
            e = float(p.get("ev", 0.0) or 0.0)
            direction += w * d
            confidence += w * min(1.0, abs(d) * 0.5 + c * 0.5)
            ev_sum += w * e
            n += 1

        direction = float(max(-1.0, min(1.0, direction / total_w)))
        confidence = float(min(1.0, confidence / max(n, 1)))
        ev_sum = float(ev_sum / total_w)
        passes_ev = ev_sum >= float(self.cfg.get("strategy.ev.min_expected_value", 0.2))

        return {
            "symbol": sym,
            "direction": direction,
            "confidence": confidence,
            "ev": ev_sum,
            "passes_ev": passes_ev,
            "execute": bool(passes_ev and abs(direction) > 0.3),
            "sources": {p.get("source"): p.get("source", "unknown") for p in proposals},
            "source": "router",
            "timestamp": self.clock.now(),
        }

    def _annotate(self, payload: dict[str, Any], sym: str) -> dict[str, Any]:
        return {
            "symbol": sym,
            "direction": float(payload.get("direction", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "ev": float(payload.get("ev", 0.0) or 0.0),
            "passes_ev": bool(payload.get("passes_ev", False)),
            "execute": bool(payload.get("execute", False)),
            "source": "router",
            "timestamp": self.clock.now(),
        }

    # -- shadow mode ---------------------------------------------------------
    def _write_shadow(self, payload: dict[str, Any], sym: str, now: float) -> None:
        self._last_shadow_seq += 1
        record = {
            "seq": self._last_shadow_seq,
            "ts": now,
            "symbol": sym,
            "direction": payload.get("direction", 0.0),
            "ev": payload.get("ev", 0.0),
            "passes_ev": payload.get("passes_ev", False),
            "execute": payload.get("execute", False),
            "source": payload.get("source", "router"),
        }
        line = json.dumps(record) + "\n"
        with self._shadow_fh.open("a", encoding="utf-8") as fh:
            fh.write(line)
        log.info("SHADOW [%d] %s → sign=%+d execute=%s", self._last_shadow_seq, sym, int(payload.get("direction", 0)), payload.get("execute", False))

    # -- emit ----------------------------------------------------------------
    def _emit_combined(self, sym: str, payload: dict[str, Any], now: float, execute: bool) -> None:
        ev = make_event(
            EventType.ORDER_PROPOSAL,
            "strategy.router",
            payload={**payload, "execute": execute},
            symbol_impact=[sym],
            timestamp=now,
        )
        self.bus.publish(ev)
        self.metrics.inc("router_proposal_emitted", symbol=sym)
