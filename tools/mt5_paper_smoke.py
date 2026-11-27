"""Bounded live-MT5 paper-execution smoke test.

Exercises the *real* decision path — MT5 ticks -> analytics -> strategies ->
FSM -> signed risk veto -> paper execution — while the execution broker stays
``paper``, so **no order can reach the broker**. Use it to validate the deploy
loop on live ticks before/after switching strategies.

What it checks
--------------
* MetaTrader5 terminal connectivity and symbol selection (``--preflight``).
* That live ticks actually reach the platform (tick/candle counters).
* That the active strategies emit proposals through the FSM, that the signed
  veto answers, and whether any proposal reaches a paper fill/reject.

Exit codes
----------
0  ticks observed (and, with ``--preflight``, symbols resolved)
1  unexpected error
2  MT5 unavailable / no ticks observed in the window

Usage
-----
    python -m tools.mt5_paper_smoke --preflight
    python -m tools.mt5_paper_smoke --seconds 120
    python -m tools.mt5_paper_smoke --symbols EURUSD,XAUUSD --max-candles 5
    python -m tools.mt5_paper_smoke --json > smoke.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.events import EventType  # noqa: E402


@dataclass
class SessionMonitor:
    """Counts the signal->fill path without touching it (observer only)."""

    bus: Any
    ticks: int = 0
    candles: int = 0
    proposals: int = 0
    veto_allow: int = 0
    veto_deny: int = 0
    fills: int = 0
    exits: int = 0
    rejects: int = 0
    proposal_symbols: set[str] = field(default_factory=set)
    fill_symbols: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.bus.subscribe(None, self.on_event)

    def on_event(self, ev: Any) -> None:
        kind = getattr(ev, "type", "")
        payload = getattr(ev, "payload", {}) or {}
        if kind == EventType.TICK.value:
            self.ticks += 1
        elif kind == EventType.CANDLE.value:
            self.candles += 1
        elif kind == EventType.ORDER_PROPOSAL.value:
            self.proposals += 1
            sym = payload.get("symbol")
            if sym:
                self.proposal_symbols.add(str(sym))
        elif kind == EventType.RISK_VETO.value:
            if str(payload.get("decision", "")).lower() == "allow":
                self.veto_allow += 1
            else:
                self.veto_deny += 1
        elif kind == EventType.ORDER_FILL.value:
            if payload.get("closed", False):
                self.exits += 1
            else:
                self.fills += 1
                sym = payload.get("symbol")
                if sym:
                    self.fill_symbols.add(str(sym))
        elif kind == EventType.ORDER_REJECT.value:
            self.rejects += 1

    def summary(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "candles": self.candles,
            "proposals": self.proposals,
            "veto_allow": self.veto_allow,
            "veto_deny": self.veto_deny,
            "fills": self.fills,
            "exits": self.exits,
            "rejects": self.rejects,
            "proposal_symbols": sorted(self.proposal_symbols),
            "fill_symbols": sorted(self.fill_symbols),
        }


def _ensure_paper_signing_key() -> bool:
    """Paper runs still sign/verify vetoes; generate an ephemeral dev key if absent."""
    if os.environ.get("RISK_SIGNING_KEY"):
        return False
    os.environ["RISK_SIGNING_KEY"] = base64.b64encode(os.urandom(32)).decode("ascii")
    return True


def build_smoke_config(config_path: str | None = None,
                       symbols: list[str] | None = None):
    """Load the deployed config, then force a safe, offline-ish paper MT5 run."""
    from main import _load_cfg

    cfg = _load_cfg(config_path)
    cfg.set("mode", "paper")
    cfg.set("execution.broker", "paper")
    cfg.set("data.market.source", "mt5")

    # Keep the smoke run side-effect free: no durable event log, no Telegram,
    # no external macro/news polling. The market feed stays live.
    cfg.set("event_log.enabled", False)
    cfg.set("notifications.telegram.enabled", False)
    cfg.set("data.news.source", "sim")
    for name in ("fred", "fed", "zillow", "ism", "challenger", "forexfactory", "gdelt"):
        cfg.set(f"data.news.{name}.enabled", False)

    if symbols:
        cfg.set("symbols", list(symbols))
        # Focus the active strategies on the same symbols so the smoke run has
        # a chance to exercise proposals/fills, not just market data.
        for name in ("noise_band", "tsmom"):
            if cfg.get(f"strategy.{name}.symbols", None) is not None:
                cfg.set(f"strategy.{name}.symbols", list(symbols))
    return cfg


def run_preflight(mt5_module: Any, symbols: list[str],
                  symbol_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Initialize the terminal, select symbols and read a tick from each."""
    symbol_map = symbol_map or {}
    if not mt5_module.initialize():
        return {"ok": False, "reason": f"initialize failed: {mt5_module.last_error()}", "symbols": []}

    rows: list[dict[str, Any]] = []
    try:
        for sym in symbols:
            broker_sym = symbol_map.get(sym, sym)
            try:
                selected = bool(mt5_module.symbol_select(broker_sym, True))
                tick = mt5_module.symbol_info_tick(broker_sym)
                rows.append({
                    "symbol": sym,
                    "broker_symbol": broker_sym,
                    "selected": selected,
                    "bid": float(getattr(tick, "bid", 0.0) or 0.0),
                    "ask": float(getattr(tick, "ask", 0.0) or 0.0),
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"symbol": sym, "broker_symbol": broker_sym, "error": str(exc)})
    finally:
        try:
            mt5_module.shutdown()
        except Exception:  # noqa: BLE001
            pass

    ok = any(r.get("bid", 0.0) > 0.0 and r.get("ask", 0.0) > 0.0 for r in rows)
    return {"ok": ok, "symbols": rows}


def _market_connected(orchestrator: Any, monitor: SessionMonitor) -> bool:
    market = getattr(orchestrator, "market", None)
    if market is not None and getattr(market, "_mt5", None) is not None:
        return True
    if getattr(orchestrator, "_connected", False):
        return True
    return monitor.ticks > 0


def run_paper_session(cfg: Any, *, seconds: float = 60.0,
                      max_candles: int | None = None,
                      orchestrator: Any = None,
                      poll_s: float = 0.05) -> dict[str, Any]:
    """Run the platform for a bounded window and return what happened.

    ``orchestrator`` is injectable for tests; when omitted the real
    :class:`main.Orchestrator` is built (MT5 feed + paper broker).
    """
    created_key = False
    if orchestrator is None:
        created_key = _ensure_paper_signing_key()
        from main import Orchestrator

        orchestrator = Orchestrator(cfg)

    monitor = SessionMonitor(orchestrator.bus)
    started = time.time()
    orchestrator.start()
    connected = _market_connected(orchestrator, monitor)
    try:
        while True:
            if max_candles is not None and monitor.candles >= max_candles:
                break
            if seconds is not None and (time.time() - started) >= seconds:
                break
            time.sleep(max(poll_s, 0.0))
    finally:
        try:
            orchestrator.stop()
        except Exception:  # noqa: BLE001
            pass

    report = monitor.summary()
    report.update({
        "duration_s": round(time.time() - started, 2),
        "connected": connected,
        "mode": str(cfg.get("mode", "paper")),
        "broker": str(cfg.get("execution.broker", "paper")),
        "market": str(cfg.get("data.market.source", "mt5")),
        "ephemeral_signing_key": created_key,
    })
    return report


def _print_session_report(report: dict[str, Any]) -> None:
    print("\n=== MT5 paper-execution smoke ===")
    print(f"  mode/broker/feed : {report['mode']} / {report['broker']} / {report['market']}")
    print(f"  connected        : {report['connected']}")
    print(f"  duration         : {report['duration_s']}s")
    print(f"  ticks / candles  : {report['ticks']} / {report['candles']}")
    print(f"  proposals        : {report['proposals']} "
          f"(allow={report['veto_allow']} deny={report['veto_deny']})")
    print(f"  fills / exits    : {report['fills']} / {report['exits']}   rejects: {report['rejects']}")
    if report.get("proposal_symbols"):
        print(f"  proposed symbols : {', '.join(report['proposal_symbols'])}")
    if report.get("fill_symbols"):
        print(f"  filled symbols   : {', '.join(report['fill_symbols'])}")
    if report.get("ephemeral_signing_key"):
        print("  note             : generated an ephemeral RISK_SIGNING_KEY for this paper run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="bounded live-MT5 paper-execution smoke test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-candles", type=int, default=None)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--preflight", action="store_true",
                    help="check terminal/symbol connectivity and exit (no platform run)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    try:
        cfg = build_smoke_config(args.config, symbols or None)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] failed to load config: {exc}")
        return 1

    try:
        import MetaTrader5 as mt5
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] MetaTrader5 python package unavailable: {exc}")
        print("      install it on Windows with the MT5 terminal: pip install MetaTrader5")
        return 2

    if args.preflight:
        report = run_preflight(mt5, list(cfg.get("symbols", [])),
                               cfg.get("data.market.mt5.symbol_map", {}))
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("\n=== MT5 preflight ===")
            print(f"  terminal init: {'OK' if report.get('ok') else 'FAILED'}")
            for row in report.get("symbols", []):
                if "error" in row:
                    print(f"  [ERR] {row['symbol']:<8} -> {row['broker_symbol']}: {row['error']}")
                else:
                    print(f"  [{'OK ' if row['bid'] > 0 else '   '}] {row['symbol']:<8} "
                          f"-> {row['broker_symbol']:<10} bid={row['bid']} ask={row['ask']}")
        return 0 if report.get("ok") else 2

    report = run_paper_session(cfg, seconds=args.seconds, max_candles=args.max_candles)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_session_report(report)

    if report["ticks"] <= 0:
        print("\n[FAIL] no MT5 ticks observed — is the terminal running and logged in?")
        return 2
    print("\n[OK] live ticks reached the platform through the paper execution path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
