# Cryptonaute HFT — Quant / News-Driven MFT Execution Platform

News & quant-driven, multi-factor trading platform with a **hard risk-governance layer**
(architecture `D:\Total recal\architecture-hft-quant-platform-v3.md`).

It runs **end-to-end with zero external services** — a deterministic simulation market feed,
a paper broker, and an in-memory event bus — so you can validate strategy logic, replay
recorded history, and backtest before ever touching a live broker. Live adapters (MetaTrader 5,
ccxt crypto, Redis, FinBERT, Mistral) activate only when configured and available.

---

## Quick start

```bash
# 1. Create the environment (Python 3.12)
python -m venv .venv312 && . .venv312/Scripts/activate      # Windows
pip install -e .

# 2. Generate deterministic sample history
cryptonaute --generate data/sample.jsonl --gen-steps 3000

# 3. Validate: replay through the REAL pipeline + vectorized backtest + walk-forward + Monte Carlo
cryptonaute --validate data/sample.jsonl

# 4. Paper-trade simulation (sim market + paper broker)
cryptonaute --paper --steps 500

# 5. Live trading (requires MT5 terminal / exchange keys in config + .env)
cryptonaute --live
```

Run the test suite (deterministic, no network):

```bash
pytest -q
```

---

## Architecture (6 layers)

| Layer | Package | Responsibility |
|------|---------|----------------|
| **L0 Core** | `core/` | Event schema, replayable in-memory bus, clock, config, metrics, component base |
| **L1 Data** | `data_ingestion/` | Market feeds (sim/MT5/crypto), news/RSS/Factory feeds, canonical normalizer |
| **L2 Analytics** | `analytics/` | Quant (MA/RSI/ATR/MACD + trend z-score), regime, macro surprise, order-flow (gated), NLP sentiment |
| **L3 Self-Correction** | `self_correction/` | Mistral post-trade review + anomaly detection — **async, off critical path**, never auto-trades |
| **L4 Strategy** | `strategy_engine/` | Per-symbol FSM, weighted signal fusion, Expected-Value gate (calibrated on real fills) |
| **L5 Execution** | `execution/` | Broker adapter (paper/MT5/crypto), paper broker w/ slippage + SL/TP, order manager w/ retry + adaptive RTT |
| **L6 Risk** | `risk_management/` | **Signed** veto microservice (hard + soft cap, drawdown, correlation), portfolio optimizer, circuit breaker |

Validation (`validation/`): vectorized backtest, walk-forward, Monte-Carlo tail-risk, and a
deterministic replay harness that runs the **real** pipeline components on recorded history.

### Governance model (per architecture §5/§6)

- **L6 Risk is the single, unified veto authority.** L5 Execution **refuses** any order the
  signed veto does not explicitly `allow`.
- **Veto decisions are cryptographically signed** (HMAC-SHA256 over the proposal + decision,
  keyed by `RISK_SIGNING_KEY`). The signature is embedded in every `RISK_VETO` event.
- **Verification is fail-closed and happens twice:**
  1. **L4 FSM** (`StrategyOrchestrator.on_veto`) re-verifies the veto signature before
     forwarding an `allow` to execution. An unverifiable veto is dropped and the position
     reset — counted under `veto_signature_rejected`.
  2. **L5 Execution** (`OrderManager.on_proposal`) is the *ultimate* authority: it
     independently re-verifies the carried veto signature and refuses to place the order
     (emitting `ORDER_REJECT`, reason `veto-signature-unverified`) if it fails. No signature
     → no order, ever.
- **Circuit breaker trips once and requires a human to re-arm** (`reset()`) — it never
  auto-recovers, preventing runaway loops.
- **Order-flow (ICT-style) hypotheses are coded but DISABLED by default** (`analytics.order_flow.enabled: false`) —
  per §3.4 they are treated as hypotheses to validate, not acquired truth.
- **L3 self-correction is strictly asynchronous** and only emits `REVIEW`/`ANOMALY` events; it has
  no write access to the execution path.

### Validation gate (architecture §7.3 / §8)

Before any strategy goes live it must pass, **in order**:

1. Backtest on historical data (vectorized reference)
2. Walk-forward validation (out-of-sample)
3. Monte-Carlo stress test (tail risk / ruin probability)
4. Paper trading (the live paper broker)
5. Small live capital deployment

The `cryptonaute --validate` command runs 1–3 and prints the gate verdict.

---

## Configuration

`config/default.yaml` — paper/sim defaults. `config/live.example.yaml` — live template
(MT5/crypto/Redis/Mistral wiring). Copy to `config/live.yaml` and edit; secrets go in `.env`
(see `.env.example`) — **never commit `.env`**.

Key tunables: broker selection, symbol list, fusion weights/threshold, EV min-expectancy,
risk hard/soft caps, drawdown limit, correlation threshold, circuit-breaker limit, ICT toggle.

---

## Infra

`infra/` contains a `Dockerfile`, `docker-compose.yml` (redis + app + prometheus + grafana),
and `infra/monitoring/` (prometheus scrape config + grafana dashboard JSON). The platform
exposes Prometheus metrics via `core.metrics`.

---

## Project layout

```
cryptonaute-hft/
├── core/                 L0 contracts
├── data_ingestion/       L1 market + news feeds, normalizer
│   ├── market_feeds/  news_feeds/
├── analytics/            L2 quant / regime / classic(macro) / order_flow / nlp
├── self_correction/      L3 Mistral client + reviewer + anomaly detector
├── strategy_engine/      L4 FSM, fusion, EV engine
├── execution/            L5 broker adapter, paper broker, order manager
├── risk_management/      L6 veto + portfolio optimizer + circuit breaker
├── validation/           backtest / walk-forward / monte carlo / replay harness
├── config/               default.yaml, live.example.yaml
├── infra/                Docker, compose, monitoring
├── tests/                deterministic pytest suite
├── main.py               CLI entry (console script: `cryptonaute`)
├── orchestrator.py       wires all layers
└── pyproject.toml
```

---

## Notes

- Deterministic by design: the sim feed, normalizer, and replay harness are seeded so replays
  are byte-for-byte reproducible (the `test_replay_is_deterministic` test enforces this).
- Live adapters degrade gracefully: if MT5/Redis/Mistral are unavailable, the platform falls
  back to sim/paper/local and logs a warning — it never crashes on a missing optional dep.
- This is a research/trading framework, not financial advice. Validate thoroughly and trade
  at your own risk.
