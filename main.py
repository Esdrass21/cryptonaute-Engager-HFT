# Cryptonaute HFT — single-file entry point + orchestrator.
#
# Merges the former main.py (CLI) and orchestrator.py (layer wiring) into one module.
# Phase 1: the bus is constructed with an event log so every published event
# is schema-validated and durably appended *before* any handler runs.
#
# Examples
# --------
#     python -m main --paper --steps 500
#     python -m main --generate data/sample.jsonl --gen-steps 3000
#     python -m main --generate data/trend.jsonl --gen-steps 3000 --trendy 0.0015 --reversion 0.004
#     python -m main --tune data/trend.jsonl --trials 24 --min-trades 5 --max-dd 5
#     python -m main --validate data/trend.jsonl
#     python -m main --live --broker crypto --exchange bybit
#     python -m main --check-keys

from __future__ import annotations

import os
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# --- project root import shim ---
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- core ---
# --- layers ---
from analytics.classic.macro import MacroAnalytics
from analytics.macro_surprise import MacroSurpriseEngine
from analytics.nlp.news_scorer import NewsScorer
from analytics.order_flow.orderbook import OrderFlowAnalytics
from analytics.quant.indicators import QuantAnalytics
from analytics.quant.regime import RegimeDetector
from core.audit import AuditLog
from core.bus import MessageBus
from core.checkpoint import Checkpoint, CheckpointState
from core.component import Component
from core.config import Clock, Config, setup_logging
from core.event_log import EventLog, build_event_log
from core.keys import KeyManager, build_key_provider
from core.metrics import Metrics
from core.schema_registry import register_default_schemas

# --- validation harness ---
from data_ingestion.generate_sample import generate, load_jsonl, write_jsonl
from data_ingestion.market_feeds.feeds import build_market_feed
from data_ingestion.news_feeds.feeds import build_news_feed
from execution.order_manager import OrderManager
from execution.trade_journal import TradeJournal
from risk_management.circuit_breaker import CircuitBreaker
from risk_management.portfolio_optimizer import PortfolioOptimizer

# --- risk: two-person rule ---
from risk_management.two_person import DEFAULT_ALLOWED_ACTIONS, TwoPersonRule
from risk_management.veto_service import RiskVetoService
from self_correction.mistral_client import AnomalyDetector, PostTradeReviewer
from strategy_engine.expected_value import ExpectedValueEngine
from strategy_engine.strategy_router import StrategyRouter
from strategy_engine.signal_fusion import SignalFusion
from strategy_engine.state_machine import StrategyOrchestrator
from validation.engine import monte_carlo, vectorized_backtest, walk_forward
from validation.replay_harness import ReplayHarness

log = logging.getLogger("main")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"
TUNED_CONFIG = Path(__file__).resolve().parent / "config" / "tuned.yaml"


# --- Telegram notifications ---
def _send_tg_notification(chat_id: str, text: str, severity: str = "info") -> None:
    """Send a Telegram notification (best-effort, never fails the command)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from notifications.telegram import TelegramNotifier
    from core.bus import MessageBus
    from core.config import Clock
    from core.metrics import Metrics
    import json as _json
    
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    try:
        bus = MessageBus(backend="memory")
        n = TelegramNotifier(bus, _load_cfg(None), Clock(), Metrics())
        n.start()
        if n._sender is not None:
            n._sender.enqueue(text, severity)
        n.stop()
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


# =============================================================================
# Orchestrator
# =============================================================================

class Orchestrator:
    """Wires all six layers onto one MessageBus and runs them."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        setup_logging(cfg.get("observability.log_level", "INFO"))

        # Schemas must be registered before any bus.publish.
        register_default_schemas()

        # Phase 2: key manager + audit log + two-person rule.
        # Accept both the legacy ``core.*`` nesting and the flat keys used by
        # config/default.yaml, so operator settings are never silently ignored.
        audit_cfg = cfg.get_any("core.audit", "audit", default={}) or {}
        self.audit = AuditLog(
            path=audit_cfg.get("path", "data/audit.jsonl"),
            fsync=audit_cfg.get("fsync", True),
        )
        self.key_manager = KeyManager(
            provider=build_key_provider(cfg.get_any("core.keys", "keys", default=None)),
            mode=cfg.get("mode", "paper"),
        )
        two_person_cfg = cfg.get_any("core.two_person", "two_person", default={}) or {}
        self.two_person = TwoPersonRule(
            audit=self.audit,
            allowed_actions=DEFAULT_ALLOWED_ACTIONS,
            min_approvers=int(two_person_cfg.get("min_approvers", 1)),
        )

        self.clock = Clock()
        self.event_log: EventLog | None = build_event_log(cfg.get("event_log"))
        self.bus = MessageBus(
            backend=cfg.get("bus.backend", "memory"),
            redis_url=cfg.get("bus.redis_url"),
            event_log=self.event_log,
        )
        self.metrics = Metrics(
            enable_prometheus=cfg.get("observability.prometheus.enabled", False),
            port=cfg.get("observability.prometheus.port", 9100),
        )
        self.components: list[Component] = []

        # Crash recovery.
        self.checkpoint: Checkpoint | None = None
        self._resume_offset = 0
        if self.event_log is not None:
            ck_path = cfg.get(
                "event_log.checkpoint_path",
                str(getattr(self.event_log, "path", "data/events.jsonl")) + ".checkpoint",
            )
            self.checkpoint = Checkpoint(ck_path)
            prior = self.checkpoint.load()
            if prior is not None:
                self._resume_offset = prior.offset
                log.warning("resuming from checkpoint offset=%d last_ts=%.3f",
                            prior.offset, prior.last_ts)

        self._wire()

    def _wire(self) -> None:
        c, bus, clk, m = self.cfg, self.bus, self.clock, self.metrics
        # Layer 1
        self.market = build_market_feed(bus, c, clk, m)
        self.news = build_news_feed(bus, c, clk, m)
        # --- Phase 6: macro feeds + nowcasts ---------------------------- #
        from analytics.nowcast.cpi import CPINowcast
        from analytics.nowcast.nfp import NFPNowcast
        from data_ingestion.news_feeds.bls import BLSReleaseFeed
        from data_ingestion.news_feeds.challenger import ChallengerFeed
        from data_ingestion.news_feeds.consensus import ConsensusFeed
        from data_ingestion.news_feeds.fed import FedFeed
        from data_ingestion.news_feeds.forexfactory import ForexFactoryFeed
        from data_ingestion.news_feeds.fred import FredFeed
        from data_ingestion.news_feeds.gdelt import GDELTFeed
        from data_ingestion.news_feeds.ism import ISMFeed
        from data_ingestion.news_feeds.zillow import ZillowFeed
        from strategy_engine.release_drift import ReleaseDriftStrategy

        self.bls = BLSReleaseFeed(bus, c, clk, m)
        self.fred = FredFeed(bus, c, clk, m)
        self.consensus = ConsensusFeed(bus, c, clk, m)
        self.nowcast_nfp = NFPNowcast(bus, c, clk, m)
        self.fed = FedFeed(bus, c, clk, m)
        self.zillow = ZillowFeed(bus, c, clk, m)
        self.ism = ISMFeed(bus, c, clk, m)
        self.challenger = ChallengerFeed(bus, c, clk, m)
        self.forexfactory = ForexFactoryFeed(bus, c, clk, m)
        self.gdelt = GDELTFeed(bus, c, clk, m)
        self.nowcast_cpi = CPINowcast(bus, c, clk, m)
        self.release_drift = ReleaseDriftStrategy(bus, c, clk, m)
        from strategy_engine.noise_band import NoiseBandStrategy
        self.noise_band = NoiseBandStrategy(bus, c, clk, m)
        from strategy_engine.tsmom import TSMOMStrategy
        self.tsmom = TSMOMStrategy(bus, c, clk, m)
        # Layer 2
        self.quant = QuantAnalytics(bus, c, clk, m)
        self.regime = RegimeDetector(bus, c, clk, m)
        self.macro = MacroAnalytics(bus, c, clk, m)
        self.macro_surprise = MacroSurpriseEngine(bus, c, clk, m)
        self.order_flow = OrderFlowAnalytics(bus, c, clk, m, enabled=c.get("analytics.order_flow.enabled", False))
        self.nlp = NewsScorer(bus, c, clk, m)
        from analytics.portfolio_var import PortfolioVaR
        self.vaR = PortfolioVaR(bus, c, clk, m)
        # Layer 3
        self.review = PostTradeReviewer(bus, c, clk, m)
        self.anomaly = AnomalyDetector(bus, c, clk, m)
        # L6 first — L4 and L5 depend on the shared veto service / key manager.
        self.veto = RiskVetoService(bus, c, clk, m, key_manager=self.key_manager)
        self.portfolio = PortfolioOptimizer(bus, c, clk, m)
        self.circuit = CircuitBreaker(bus, c, clk, m, two_person=self.two_person)
        # Layer 4
        self.fusion = SignalFusion(bus, c, clk, m)
        self.ev = ExpectedValueEngine(bus, c, clk, m)
        self.fsm = StrategyOrchestrator(bus, c, clk, m, key_manager=self.key_manager)
        # Layer 4b: strategy router combines TSMOM + noise_band + fusion -> FSM
        self.router = StrategyRouter(bus, self.cfg, clk, m, shadow_enabled=self.cfg.get('strategy.shadow.enabled', False))
        # Layer 5
        self.execution = OrderManager(bus, c, clk, m, key_manager=self.key_manager)
        # Audit trail
        self.journal = TradeJournal(bus, c, clk, m)

        # Notification layer — off the critical path, best-effort.
        from notifications.telegram import TelegramNotifier
        self.notifier = TelegramNotifier(bus, c, clk, m)

        self.components = [
            self.market, self.news, self.bls, self.fred, self.consensus, self.nowcast_nfp,
            self.fed, self.zillow, self.ism, self.challenger, self.forexfactory, self.gdelt,
            self.nowcast_cpi, self.release_drift, self.noise_band, self.tsmom,
            self.quant, self.regime, self.macro, self.macro_surprise,
            self.order_flow, self.nlp, self.vaR, self.review, self.anomaly,
            self.fusion, self.ev, self.fsm, self.router, self.execution,
            self.veto, self.portfolio, self.circuit, self.journal, self.notifier,
        ]

    def start(self) -> None:
        for comp in self.components:
            try:
                comp.start()
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to start %s: %s", comp.name, exc)
        log.info("=== Cryptonaute HFT platform ONLINE (mode=%s) ===", self.cfg.get("mode"))
        self._emit_account_metric()

    def _emit_account_metric(self) -> None:
        try:
            state = self.execution.broker.get_account_state()
            self.metrics.set("account_drawdown_pct", state.get("drawdown_pct", 0.0))
            self.metrics.set("account_balance", state.get("balance", 0.0))
        except Exception:  # noqa: BLE001
            pass

    def run(self, steps: int | None = None) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        self.start()
        try:
            if hasattr(self.market, "run_realtime"):
                self.market.run_realtime(steps=steps)
            else:
                while self.market.running:
                    time.sleep(1.0)
        finally:
            self.stop()

    def stop(self) -> None:
        for comp in self.components:
            try:
                comp.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("stop() failed for %s: %s", getattr(comp, "name", comp), exc)
        # Close feeds that own an HTTP session.
        for name in ("fred", "consensus", "fed", "forexfactory", "gdelt", "zillow", "ism", "challenger"):
            obj = getattr(self, name, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("close() failed for %s: %s", name, exc)

        # Persist a recovery checkpoint (best-effort).
        if self.checkpoint is not None and self.event_log is not None:
            try:
                state = {
                    "key_id": self.key_manager.current_key_id()
                    if hasattr(self, "key_manager") else "",
                    "broker_balance": float(
                        self.execution.broker.get_account_state().get("balance", 0.0)
                    ),
                    "open_positions": [
                        {"symbol": p.get("symbol"), "ticket": p.get("ticket"),
                         "side": p.get("side"), "size": p.get("size")}
                        for p in self.execution.broker.get_open_positions()
                    ],
                }
                self.checkpoint.write(CheckpointState(
                    offset=self.event_log.count(),
                    last_ts=self.clock.now(),
                    state=state,
                ))
                log.info("wrote checkpoint offset=%d", self.event_log.count())
            except Exception as exc:  # noqa: BLE001
                log.warning("checkpoint write failed: %s", exc)

        if self.event_log is not None:
            self.event_log.close()
        log.info("Platform stopped. Final metrics: %s", self.metrics.snapshot())

    def snapshot(self) -> dict:
        return self.metrics.snapshot()


# =============================================================================
# Config loading
# =============================================================================

def _load_cfg(path: str | None) -> Config:
    """Load config, auto-merging a gate-passing tuned config when present.

    Precedence: explicit --config > config/tuned.yaml > config/default.yaml.
    """
    if path and Path(path).exists():
        return Config.load(str(DEFAULT_CONFIG), path)
    if TUNED_CONFIG.exists():
        return Config.load(str(DEFAULT_CONFIG), str(TUNED_CONFIG))
    if DEFAULT_CONFIG.exists():
        return Config.load(str(DEFAULT_CONFIG))
    return Config.load()


# =============================================================================
# CLI commands
# =============================================================================

def cmd_run(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🟢 *Cryptonaute — Paper/Live mode activé*
- Mode: `"""+("live" if args.live else "paper")+"""`
- Broker: `"""+str(args.broker)+"""`
- Exchange: `"""+str(args.exchange if args.exchange else "auto")+"""`
- Market: `"""+str(args.market if args.market else "auto")+"""`
        """,
        severity="info",
    )
    cfg.set("mode", "live" if args.live else "paper")
    if args.market:
        cfg.set("data.market.source", args.market)
    if args.news:
        cfg.set("data.news.source", args.news)
    if args.exchange:
        cfg.set("execution.exchange", args.exchange)
    if args.live:
        if args.broker == "crypto":
            cfg.set("data.market.source", args.market or "crypto")
            cfg.set("execution.broker", "crypto")
        else:
            cfg.set("data.market.source", args.market or "mt5")
            cfg.set("execution.broker", "mt5")
    elif args.broker == "crypto":
        cfg.set("data.market.source", args.market or "crypto")
        cfg.set("execution.broker", "crypto")
    orch = Orchestrator(cfg)
    orch.run(steps=args.steps)
    return 0


def cmd_generate(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
📊 *Cryptonaute — Génération d'historique synthétique*
- Fichier: `"""+args.generate+"""`
- Étapes: `"""+str(args.gen_steps)+"""`
- Seed: `"""+str(args.seed)+"""`
- Trendy: `"""+str(args.trendy)+"""`
- Reversion: `"""+str(args.reversion)+"""`
        """,
        severity="info",
    )
    events = generate(cfg, args.gen_steps, seed=args.seed,
                      trend_strength=args.trendy, mean_reversion=args.reversion)
    n = write_jsonl(events, args.generate)
    log.info("Generated %d events -> %s", n, args.generate)
    return 0


def _gate_override(gate: dict, args) -> dict:
    """Read validation.gate with per-mode (paper/live) override selected by
    the --live flag.  Returns min_dsr, max_pbo, max_daily_drawdown_pct."""
    mode = "live" if getattr(args, "live", False) else "paper"
    override = gate.get(mode, {}) if isinstance(gate.get(mode), dict) else {}
    return {
        "min_dsr": float(override.get("min_dsr", gate.get("min_dsr", 0.0))),
        "max_pbo": float(override.get("max_pbo", gate.get("max_pbo", 0.5))),
        "max_daily_drawdown_pct": float(
            override.get("max_daily_drawdown_pct",
                         gate.get("max_daily_drawdown_pct", 2.0))),
    }


def evaluate_gate(base_pass: bool, meta_pass: bool) -> tuple[bool, str]:
    gate_pass = base_pass and meta_pass
    return gate_pass, ("PASS" if gate_pass else "FAIL")


def cmd_validate(args) -> int:
    from pathlib import Path
    import numpy as np

    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🛡️ *Cryptonaute — Validation de stratégie*
- Fichier d'historique: `"""+args.validate+"""`
- Mode: `"""+("live" if args.live else "paper")+"""`
        """,
        severity="info",
    )

    from validation.execution_model import build_execution_model

    cfg = _load_cfg(args.config)
    events = load_jsonl(args.validate)

    # [1/3] Replay harness.
    harness = ReplayHarness(cfg, events)
    harness.run()
    state = harness.account_state()
    net = state["balance"] - cfg.get("validation.backtest.initial_capital", 10000.0)
    dd = state.get("drawdown_pct", 0.0)
    eq_rets = harness.equity_returns()

    from validation.metrics import SharpeStats
    pipe_stats = SharpeStats.from_returns(eq_rets) if len(eq_rets) >= 2 else None

    print("\n[1/3] Replay through real pipeline (paper broker):")
    print(f"      net P&L = {net:+.2f}  drawdown = {dd:.2f}%  "
          f"open positions = {len(harness.open_positions())}")
    if pipe_stats is not None:
        print(f"      pipeline Sharpe (per-event) = {pipe_stats.sharpe:.4f}  "
              f"n = {pipe_stats.n}")

    # [2/3] Vectorized reference backtest.
    sym = cfg.get("symbols", ["EURUSD"])[0]
    closes = [e.payload["close"] for e in events
              if e.type == "candle" and e.payload.get("symbol") == sym]
    gate_pass = False
    if closes:
        arr = np.array(closes)
        signal = np.zeros_like(arr)
        ma_fast = np.concatenate([[arr[0]], (arr[1:] + arr[:-1]) / 2])
        signal[1:] = np.sign(arr[1:] - ma_fast[1:]) * 0.5
        exec_model = build_execution_model(cfg)
        bt = vectorized_backtest(
            arr, signal,
            initial_capital=float(cfg.get("validation.backtest.initial_capital", 10000.0)),
            exec_model=exec_model, symbol=sym,
        )
        wf = walk_forward(arr, lambda tr, te: np.sign(te[1:] - te[:-1]) * 0.5,
                          n_splits=int(cfg.get("validation.walk_forward.n_splits", 5)),
                          embargo_pct=float(cfg.get("validation.walk_forward.embargo_pct", 0.01)),
                          horizon=int(cfg.get("validation.walk_forward.horizon", 1)))
        mc = monte_carlo(bt.trades, n_paths=int(cfg.get("validation.monte_carlo.n_paths", 1000)))
        print("\n[2/3] Vectorized backtest (reference, execution-aware):")
        for k, v in bt.summary().items():
            print(f"      {k:18s}: {v}")
        print("\n[3/3] Monte Carlo (tail risk):")
        for k, v in mc.items():
            print(f"      {k:18s}: {round(v, 2)}")
        avg_wr = sum(r.win_rate for r in wf) / len(wf) if wf else 0.0
        print(f"\n      purged walk-forward folds = {len(wf)}  avg OOS win_rate = {avg_wr:.3f}")

        # --- Gate -------------------------------------------------------- #
        # Single source of truth: validation.gate.  Per-mode override
        # (paper/live) is selected by the --live flag.
        gate = cfg.get("validation.gate", {})
        override = _gate_override(gate, args)
        max_dd = override["max_daily_drawdown_pct"]
        min_dsr = override["min_dsr"]
        max_pbo = override["max_pbo"]

        base_pass = (net > 0.0) and (dd <= max_dd)

        # DSR / PBO from the tuning meta sidecar (if --tune has been run).
        meta = None
        if not getattr(args, "ignore_meta", False):
            meta_path = Path(str(TUNED_CONFIG) + ".meta.json")
            if meta_path.exists():
                try:
                    from validation.tuning import load_meta
                    meta = load_meta(meta_path)
                except Exception as exc:
                    log.warning("failed to load tuning meta: %s", exc)

        meta_pass = True
        if meta is not None:
            dsr_ok = meta.dsr >= min_dsr
            pbo_ok = (meta.pbo != meta.pbo) or (meta.pbo <= max_pbo)  # NaN tolerated
            meta_pass = dsr_ok and pbo_ok

        gate_pass, verdict = evaluate_gate(base_pass, meta_pass)
        print("\nVALIDATION GATE (architecture §7.3):")
        print(f"  net expectancy = {net:+.2f}  (need > 0)")
        print(f"  max drawdown  = {dd:.2f}%  (limit {max_dd:.2f}%)")
        if meta is not None:
            print(f"  tuned DSR     = {meta.dsr:.4f}  (need >= {min_dsr:.4f}, "
                  f"n_trials={meta.n_trials})")
            print(f"  tuned PBO     = {meta.pbo:.4f}  (need <= {max_pbo:.4f})")
            if meta.oos_net is not None:
                print(f"  tuned OOS net = {meta.oos_net:+.2f}  "
                      f"dd={meta.oos_dd:.2f}%  sharpe={meta.oos_sharpe:.4f}")
        else:
            print("  (no tuning meta found — run --tune to enable DSR/PBO gating)")
        print(f"  RESULT: {verdict}")
    else:
        print("No candles found in recorded history for backtest.")
    return 0 if gate_pass else 1


def cmd_tune(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🔧 *Cryptonaute — Tuning de stratégie*
- Historique: `"""+args.tune+"""`
- Trials: `"""+str(args.trials if args.trials else "défaut")+"""`
- Min trades: `"""+str(args.min_trades)+"""`
- Max DD: `"""+str(args.max_dd)+"""`
- IS frac: `"""+str(args.is_frac)+"""`
        """,
        severity="info",
    )
    events = load_jsonl(args.tune)
    from validation.tuning import tune, write_meta, write_tuned_config

    report = tune(
        events, cfg,
        trials=args.trials,
        min_trades=args.min_trades,
        max_dd_pct=args.max_dd,
        is_frac=args.is_frac,
    )

    print(f"\nTuned {report.n_trials} trials against {len(events)} events "
          f"(IS frac={report.is_frac:.2f}, OOS events={report.oos_events}).")
    print(f"{'rank':>4} {'tech':>5} {'news':>5} {'macro':>5} {'reg':>4} "
          f"{'thr':>4} {'evMin':>5} {'net':>10} {'trades':>6} {'dd%':>6} {'PASS':>4}")
    for i, r in enumerate(report.results[:12], 1):
        print(f"{i:>4} {r.params['tech']:>5} {r.params['news']:>5} "
              f"{r.params['macro']:>5} {r.params['regime']:>4} "
              f"{r.params['threshold']:>4} {r.params['ev_min']:>5} "
              f"{r.net_pnl:>10.2f} {r.trades:>6} {r.max_dd_pct:>6.2f} "
              f"{str(r.passed):>4}")

    if report.best is not None:
        out = write_tuned_config(report.best, args.out)
        meta_path = write_meta(report, args.out + ".meta.json")
        print(f"\nBEST (IS): net={report.best.net_pnl:.2f} "
              f"trades={report.best.trades} dd={report.best.max_dd_pct:.2f}%")
        print(f"Wrote tuned config -> {out}")
        print(f"Wrote tuning meta  -> {meta_path}")
    else:
        print("\nNo trial cleared the IS gate (positive expectancy + DD within limit + min trades).")
        print("Consider richer/trending history, enabling order-flow, or widening the grid.")
        return 0

    print(f"\n  IS Sharpe   : {report.is_sharpe:.4f}")
    print(f"  SR0 (null)  : {report.sr0:.4f}  (expected max of {report.n_trials} trials)")
    print(f"  DSR         : {report.dsr:.4f}")
    print(f"  PBO         : {report.pbo:.4f}")
    if report.oos_net is not None:
        print(f"  OOS net     : {report.oos_net:+.2f}  dd={report.oos_dd:.2f}%  "
              f"sharpe={report.oos_sharpe:.4f}  trades={report.oos_trades}")
    return 0


def cmd_check_keys(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🔑 *Cryptonaute — Vérification des credentials*
- Vérification des clés et secrets configurés
        """,
        severity="info",
    )
    def _present(k: str) -> bool:
        v = cfg.secret(k, "")
        return bool(v and v.strip())

    print("\nCredential check (values never shown):\n")
    exchange = str(cfg.get("execution.exchange", "binance")).lower()
    ek = f"{exchange.upper()}_API_KEY"
    es = f"{exchange.upper()}_API_SECRET"
    rows = [
        (f"{exchange} key",    ek,                    _present(ek)),
        (f"{exchange} secret", es,                    _present(es)),
        ("Mistral (L3)",       "MISTRAL_API_KEY",     _present("MISTRAL_API_KEY")),
        ("MT5 login",          "MT5_LOGIN",           _present("MT5_LOGIN")),
        ("MT5 password",       "MT5_PASSWORD",        _present("MT5_PASSWORD")),
        ("MT5 server",         "MT5_SERVER",          _present("MT5_SERVER")),
    ]
    for label, key, ok in rows:
        print(f"  [{'OK ' if ok else '   '}] {label:16s} ({key})")

    if _present(ek) and _present(es):
        print(f"\n  Authenticating with {exchange} (signed balance read, no orders)...")
        try:
            import ccxt
            klass = getattr(ccxt, exchange, ccxt.binance)
            ex = klass({"apiKey": cfg.secret(ek), "secret": cfg.secret(es),
                        "enableRateLimit": True})
            bal = ex.fetch_balance()
            nonzero = {k: v for k, v in bal.get("total", {}).items() if v}
            print(f"  [OK ] {exchange} authenticated — {len(nonzero)} non-zero balances")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR] {exchange} auth failed: {exc}")
            return 1
    else:
        print(f"\n  ({exchange} keys absent — skipping live auth check)")
    return 0


def cmd_rotate_keys(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🔑 *Cryptonaute — Rotation des keys*
        """,
        severity="info",
    )
    from core.audit import AuditLog
    from core.keys import KeyManager, build_key_provider
    from risk_management.two_person import ACTION_KEY_ROTATION, Approval, TwoPersonRule

    audit_cfg = cfg.get_any("core.audit", "audit", default={}) or {}
    two_person_cfg = cfg.get_any("core.two_person", "two_person", default={}) or {}
    audit = AuditLog(audit_cfg.get("path", "data/audit.jsonl"))
    rule = TwoPersonRule(audit, allowed_actions=[ACTION_KEY_ROTATION],
                         min_approvers=two_person_cfg.get("min_approvers", 1))
    approvers = tuple(a for a in (args.approvers or "").split(",") if a)
    rule.check(Approval(actor=args.actor, approvers=approvers,
                        action=ACTION_KEY_ROTATION, payload={"reason": args.reason}))
    km = KeyManager(build_key_provider(cfg.get_any("core.keys", "keys", default=None)),
                    mode=cfg.get("mode", "paper"))
    before = km.current_key_id()
    km.reload()
    after = km.current_key_id()
    print(f"key rotation: {before} -> {after}")
    return 0


def cmd_reset_breaker(args) -> int:
    print("Circuit breaker reset must be performed in-process (see runbook).")
    print("This CLI hook is intentionally a no-op stub so the operation is never shell-reachable.")
    return 2


# =============================================================================
# Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cryptonaute", description="Cryptonaute HFT Quant Platform")
    ap.add_argument("--config", default=None)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--broker", default="mt5", choices=["mt5", "crypto"])
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--market", default=None, choices=["sim", "mt5", "crypto", "hybrid", "providers"])
    ap.add_argument("--news", default=None, choices=["sim", "rss", "factory"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--generate", metavar="OUT", default=None)
    ap.add_argument("--gen-steps", dest="gen_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trendy", dest="trendy", type=float, default=0.0)
    ap.add_argument("--reversion", dest="reversion", type=float, default=0.01)
    ap.add_argument("--validate", metavar="HISTORY", default=None)
    ap.add_argument("--tune", metavar="HISTORY", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--min-trades", dest="min_trades", type=int, default=5)
    ap.add_argument("--max-dd", dest="max_dd", type=float, default=2.0)
    ap.add_argument("--out", default="config/tuned.yaml")
    ap.add_argument("--check-keys", dest="check_keys", action="store_true")
    ap.add_argument("--rotate-keys", dest="rotate_keys", action="store_true")
    ap.add_argument("--actor", default=None)
    ap.add_argument("--approvers", default=None, help="Comma-separated approver IDs")
    ap.add_argument("--reason", default=None)
    ap.add_argument("--is-frac", dest="is_frac", type=float, default=0.8,
                    help="Fraction of history used for in-sample tuning (rest is OOS holdout)")
    ap.add_argument("--ignore-meta", dest="ignore_meta", action="store_true",
                    help="Ignore config/tuned.yaml.meta.json when validating")
    ap.add_argument("--fetch", dest="fetch", action="store_true",
                    help="Download all public data into data/features/")
    ap.add_argument("--fetch-out", dest="fetch_out", default="data/features")
    ap.add_argument("--fetch-start", dest="fetch_start", default="2015-01-01")
    ap.add_argument("--train-nfp", dest="train_nfp", action="store_true",
                    help="Fit the NFP nowcast coefficients")
    ap.add_argument("--train-cpi", dest="train_cpi", action="store_true",
                    help="Fit the CPI nowcast coefficients")
    ap.add_argument("--drift", dest="drift", action="store_true",
                    help="Measure post-release drift and suggest rules")
    ap.add_argument("--drift-history", dest="drift_history", default="data/events.jsonl")
    ap.add_argument("--drift-symbols", dest="drift_symbols", default="EURUSD,BTCUSDT")
    ap.add_argument("--drift-out", dest="drift_out", default="config/drift_rules.yaml")
    ap.add_argument("--test-telegram", dest="test_telegram", action="store_true",
                    help="Send a test message to the configured Telegram channel and exit")
    return ap


def cmd_fetch(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
📥 *Cryptonaute — Téléchargement des données*
- Sortie: `"""+args.fetch_out+"""`
- Début: `"""+args.fetch_start+"""`
- Mode: `"""+cfg.get("mode","paper")+"""`
        """,
        severity="info",
    )
    from tools.fetch_features import main as fetch_main
    argv = ["--out", args.fetch_out, "--start", args.fetch_start]
    sys.argv = ["fetch_features"] + argv
    fetch_main()
    return 0


def cmd_train(args, which: str) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
📈 *Cryptonaute — Ajustage du modèle nowcast*
- Type: `"""+("NFP" if which == "nfp" else "CPI")+"""`
- Fichier de sortie: `"""+("config/nowcast/nfp_coefficients.json" if which == "nfp" else "config/nowcast/cpi_coefficients.json")+"""`
- Mode: `"""+cfg.get("mode","paper")+"""`
        """,
        severity="info",
    )
    import subprocess
    import sys
    tool = "tools/train_nfp_nowcast.py" if which == "nfp" else "tools/train_cpi_nowcast.py"
    csv = "data/features/nfp_features.csv" if which == "nfp" else "data/features/cpi_features.csv"
    out = "config/nowcast/nfp_coefficients.json" if which == "nfp" \
        else "config/nowcast/cpi_coefficients.json"
    return subprocess.call([sys.executable, tool, "--csv", csv, "--out", out])


def cmd_drift(args) -> int:
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
📊 *Cryptonaute — Mesure de drift*
- Historique: `"""+args.drift_history+"""`
- Symbols: `"""+args.drift_symbols+"""`
- Sortie: `"""+args.drift_out+"""`
        """,
        severity="info",
    )
    import subprocess
    import sys
    return subprocess.call([
        sys.executable, "-m", "validation.drift_analysis",
        "--history", args.drift_history,
        "--symbols", args.drift_symbols,
        "--out", args.drift_out,
    ])


def cmd_test_telegram(args) -> int:
    import json as _json
    cfg = _load_cfg(args.config)
    _send_tg_notification(
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("RELAY_CHANNEL"),
        text="""
🧪 *Cryptonaute — Test de connexion Telegram*
        """,
        severity="info",
    )
    from core.bus import MessageBus
    from core.config import Clock
    from core.metrics import Metrics
    from notifications.telegram import TelegramNotifier, _normalize_chat_id, escape_mdv2
    bus = MessageBus(backend="memory")
    n = TelegramNotifier(bus, cfg, Clock(), Metrics())
    n.start()

    if n._sender is None and n._relay_sender is None:
        print("Telegram not configured — set TELEGRAM_BOT_TOKEN and at least one "
              "of TELEGRAM_CHAT_ID / RELAY_CHANNEL in .env")
        return 1

    if n._sender is not None:
        n._sender.enqueue(escape_mdv2("Main channel test — wiring OK."), "info")
        print(f"  main  -> {n.chat_id}")

    if n._relay_sender is not None:
        n._relay_sender.enqueue(
            escape_mdv2("Relay channel test — confirmed signals will appear here."),
            "info",
        )
        print(f"  relay -> {_normalize_chat_id(n.relay_channel)}")

    time.sleep(4.0)
    print("\nstats:", _json.dumps(n.stats(), indent=2))
    n.stop()
    return 0


def main(argv: list | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.check_keys:
        return cmd_check_keys(args)
    if args.fetch:
        return cmd_fetch(args)
    if args.train_nfp:
        return cmd_train(args, "nfp")
    if args.train_cpi:
        return cmd_train(args, "cpi")
    if args.drift:
        return cmd_drift(args)
    if args.test_telegram:
        return cmd_test_telegram(args)
    if args.rotate_keys:
        return cmd_rotate_keys(args)
    if args.generate:
        return cmd_generate(args)
    if args.validate:
        return cmd_validate(args)
    if args.tune:
        return cmd_tune(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
