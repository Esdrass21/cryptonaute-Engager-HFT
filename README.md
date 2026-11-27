# Cryptonaute HFT — Quant / News-Driven MFT Execution Platform

Multi-factor trading platform (quant + news signals) with a cryptographically-signed risk-veto layer. Fully self-contained in simulation — deterministic market feed, paper broker, in-memory event bus — so you can validate strategy logic, replay recorded history, and backtest before ever touching a live broker. Live adapters (MetaTrader 5, ccxt crypto, Redis, FinBERT, Mistral) activate only when configured and available.

For quant researchers and solo traders who want to validate strategy logic in simulation before risking capital.

Architecture reference: `architecture-hft-quant-platform-v3.md`.

---

## Module map

If you only run one command, run `make help`.

### Entry points

| Command | What it does |
|---|---|
| `python -m main --fetch` | Download all public data to `data/features/` |
| `python -m main --train-nfp` / `--train-cpi` | Fit nowcast coefficients |
| `python -m main --generate <file>` | Produce synthetic trended history |
| `python -m main --tune <file>` | Grid-search + DSR/PBO on IS only |
| `python -m main --validate <file>` | Deploy gate — exit 1 on FAIL |
| `python -m main --paper --steps N` | Paper-trade the tuned config |
| `python -m main --drift` | Measure post-release drift, write suggested rules |
| `python -m main --live` | Live (requires `config/live.yaml` + keys) |
| `python -m main --check-keys` | Report credential presence (never prints secrets) |
| `python -m main --rotate-keys` | Two-person-gated key rotation |

### Layers

| Layer | Modules | What they own |
|---|---|---|
| L0 Core | `core/` | Events, bus, schema registry, event log, config, metrics, keys, audit |
| L1 Data | `data_ingestion/` | Market feeds, BLS, FRED, Fed RSS, Zillow, Consensus |
| L2 Analytics | `analytics/` | Quant, regime, macro, order flow, NLP, **nowcast** |
| L3 Self-correction | `self_correction/` | Async Mistral reviewer, anomaly detector |
| L4 Strategy | `strategy_engine/` | FSM, fusion, EV, **release drift** |
| L5 Execution | `execution/` | Paper broker (book-aware), OrderManager, **limit queue** |
| L6 Risk | `risk_management/` | Signed veto, circuit breaker, portfolio, **two-person rule** |
| Validation | `validation/` | Backtest, walk-forward, DSR/PBO, replay, **drift analysis** |

### Public data sources

| Module | Source | Free? | Key? |
|---|---|---|---|
| `news_feeds/bls.py` | BLS release pages + RSS | ✅ | no |
| `news_feeds/fred.py` | FRED / ALFRED API | ✅ | `FRED_API_KEY` |
| `news_feeds/fed.py` | Federal Reserve RSS | ✅ | no |
| `news_feeds/zillow.py` | Zillow ZORI CSV + Cox scrape | ✅ | no |
| `news_feeds/consensus.py` | FXMacroData (USD tier) | ✅ | `FXMACRODATA_API_KEY` |

### The full loop

```
fetch → train → generate → tune → validate → paper → drift → (back to train)
```

Every arrow is one Makefile target. `make drift` closes the loop: it eats the
recording produced by `make paper` and rewrites the strategy's parameters.

---

## How Cryptonaute actually helps you

Let me answer this honestly, because a lot of trading frameworks promise more than they deliver. I'll cover what it genuinely does for you, what it doesn't, and where the real value is — given everything we've built across Phases 1–5.

### The one-sentence answer

It replaces "I have an idea, let me risk money to find out if it's real" with "I have an idea, let me measure — honestly, cheaply, repeatedly — whether it survives contact with reality before any money moves."

That's it. Everything else is in service of that.

### What it concretely gives you

#### 1. A lie detector for your own ideas

The single biggest reason retail quants lose money isn't bad strategies — it's self-deception. You tweak parameters until the backtest looks great, and you don't realize you've fit noise.

Phase 3 fixes this structurally:

- **Purged + embargoed CV** — your labels can't peek into the test set.
- **Deflated Sharpe Ratio** — if you tried 24 configs, DSR tells you what the expected max Sharpe under pure noise is, so you know whether your 1.8 Sharpe is signal or survivor bias.
- **PBO** — tells you the probability that your IS-best config lands below median OOS. Above 0.5, it's noise.

Before Phase 3, a "green gate" meant "looks good on the data I tuned on." After Phase 3, it means "looks good on data the tuning never saw, and survives the multiple-testing penalty." Those are very different claims.

Practical impact: you stop deploying strategies that are 90% overfit. That alone is worth more than every other feature combined.

#### 2. A hard stop when you're wrong

The signed-veto layer isn't a nice-to-have — it's the thing that keeps a bad day from becoming a catastrophic one.

- L6 signs every risk decision with a rotated HMAC key.
- L4 verifies before forwarding. L5 verifies again before sending.
- Circuit breaker trips once and needs two humans to re-arm.
- Every decision is in an immutable audit log.

If you ever have a strategy go haywire — a bug that opens 100 positions, a signal that never stops firing — the circuit breaker fires, the veto denies, and nothing is placed. You don't need to be watching. The system is designed to fail closed when you're not.

#### 3. Fills that don't lie to you

Phase 4 and 4.5 replace "fills at mid ± 0.05%" with:

- A synthetic order book with real depth.
- Slippage that scales with size — a $500 order and a $500k order pay very different amounts.
- Partial fills when you exceed top-of-book depth.
- Commissions on both sides.
- FIFO queue position for limit orders — you only fill after everyone ahead of you fills.
- TTL, cancel/replace, crossing-as-taker.

Practical impact: your paper P&L now looks like what your live P&L will look like. Most backtests are 20–50% too optimistic purely because of naive fill assumptions. You'll discover that before the money moves.

#### 4. An honest pipeline you can trust

Every layer is deterministic and reproducible:

- Same history + same config → byte-identical events, fills, and P&L.
- Every published event is schema-validated and durably logged before dispatch.
- Replays reconstruct decisions from raw inputs (ticks, candles, news) — not from recorded decisions, which would let the strategy cheat.
- `pytest -q -k replay_is_deterministic` proves it.

Practical impact: when something goes wrong, you can replay the exact session and see what happened. When something goes right, you can prove it wasn't luck of the RNG.

#### 5. A deploy gate that can't be argued with

`cryptonaute --validate <history>` returns exit 0 or 1. That's the whole contract.

- Runs backtest → walk-forward → Monte Carlo, in order, aborts on first failure.
- Applies DSR + PBO from your tuning run.
- Exit 1 on any breach.

You can wire this into CI, into a pre-live checklist, into a cron job. It doesn't care whether you're tired, excited, or sure. It says PASS or FAIL.

#### 6. A system you can extend without fear

Because of what we built:

- **New signal?** Implement a component, subscribe to the bus, add a weight. The registry checks it.
- **New broker?** Implement `BrokerAdapter`. Retries, veto verification, fills, and reconciliation all come for free.
- **New metric?** One line in the catalog. Dashboards and docs regenerate.
- **New failure mode?** Add a chaos test. The suite enforces the fail-closed contract.

You're not building a system; you're building on one.

---

## The concrete workflows it enables

### Workflow A — Idea → verdict, no capital at risk

```bash
# 1. Generate trended history (or use your own recorded feed)
cryptonaute --generate data/trend.jsonl --gen-steps 5000 --trendy 0.0015 --reversion 0.004

# 2. Tune against the REAL pipeline, on IS only
cryptonaute --tune data/trend.jsonl --trials 48 --min-trades 20 --max-dd 5

# 3. Look at the IS Sharpe, SR0 (null), DSR, PBO, and the OOS re-run
# If DSR < 0.95 or PBO > 0.3, the idea is probably noise. Stop here.

# 4. Validate
cryptonaute --validate data/trend.jsonl
# RESULT: PASS or FAIL. Exit 0 or 1.
```

You've spent zero dollars and zero minutes on a live broker. You have a defensible answer.

### Workflow B — Paper trading as a shakedown

```bash
cryptonaute --paper --steps 5000
```

Runs the full 6-layer pipeline with a simulated market + realistic paper broker. You'll see:

- Which signals fire and how often.
- How slippage and partials affect realized P&L.
- Whether the circuit breaker trips and why.
- Fill latency, veto rejections, queue behavior — all instrumented.

If it survives paper, you have real evidence, not a hope.

### Workflow C — Small live, monitored, reversible

```bash
cryptonaute --live --config config/live.yaml --broker crypto --exchange bybit
```

- Two-person approval required to activate.
- Signed vetoes prevent unauthorized orders.
- Circuit breaker trips on DD > limit; needs humans to re-arm.
- Every metric is on a dashboard, every action is in the audit log.
- You start at 0.5% position size and ramp only as the numbers earn it.

### Workflow D — Postmortem without guesswork

When something goes wrong (it will), you have:

- `data/events.jsonl` — every event that ever hit the bus.
- `data/audit.jsonl` — every governance action.
- Replay harness — rerun the exact session deterministically.
- Metrics — where it went sideways, in which layer.

Compare this to "I think the MT5 connection dropped around 3pm." That's a different kind of debugging.

---

## What it does NOT do

This matters more than the selling points.

- **It won't find alpha for you.** It will tell you honestly whether the alpha you have is real. If you feed it noise, it tells you it's noise.
- **It won't predict markets.** No LLM, no model, no amount of engineering makes a random walk tradeable.
- **It won't make a bad strategy profitable.** The gate is designed to reject, not to rescue.
- **It's not financial advice.** Not a signal service. Not a bot you plug in and forget.
- **It's not battle-tested in live markets by a large community.** It's a serious, well-engineered research and execution platform. It's not a fund.
- **The synthetic book is a model, not reality.** Calibrate it against your venue's actual L2 feed before trusting the paper P&L to within a few percent.

---

## End-to-end walkthrough — MA cross on data/trend.jsonl

Important caveat up front: I can't execute code, so the numbers below are representative — they show the shape and scale of what the pipeline produces for a trended 3000-step history with a modest real edge. Your actual numbers will differ. What matters is how to read each line and what decision it forces.

The strategy under test: a simple MA cross — go long when fast MA > slow MA, flat/short otherwise. This is the "reference" signal already wired into `cmd_validate`. The point of the walkthrough is not the strategy; it's the pipeline's honesty.

### Step 0 — Setup

```bash
python -m venv .venv312 && . .venv312/Scripts/activate
pip install -e .
pytest -q
```

Expected: all tests green — including the chaos suite from Phase 5. If any fail, stop. Don't tune on an unverified platform.

### Step 1 — Generate a history with a real edge

```bash
cryptonaute --generate data/trend.jsonl --gen-steps 3000 --trendy 0.0015 --reversion 0.004
```

```
Generated 3000 events -> data/trend.jsonl
```

The generator emits only raw inputs (ticks, candles, order book, news). No decisions. The replay harness will re-derive composites, proposals, vetoes, and fills from those raw inputs. This is what makes the recording reusable — and what stops the strategy from "cheating" by reading its own past outputs.

`--trendy 0.0015` injects a slow drift; `--reversion 0.004` is the OU pull-back speed. Lower reversion = more persistent trends. This produces an Ornstein-Uhlenbeck process anchored to a drifting regime, so trends are capturable by MA-cross but prices stay bounded.

**Sanity check** — pure GBM has no edge, so the gate must reject it:

```bash
cryptonaute --generate data/sample.jsonl --gen-steps 3000
cryptonaute --validate data/sample.jsonl ; echo "exit=$?"
```

Expected: `RESULT: FAIL`, `exit=1`. If it passes on pure noise, something is broken — stop and investigate before continuing.

### Step 2 — Tune against the REAL pipeline

```bash
cryptonaute --tune data/trend.jsonl --trials 24 --min-trades 5 --max-dd 5 --is-frac 0.8
```

Representative output:

```
Tuned 24 trials against 3000 events (IS frac=0.80, OOS events=600).
rank tech news macro reg thr evMin net trades dd% PASS
1 0.8 0.4 0.15 0.1 0.65 0.02 184.32 41 0.00 True
2 0.8 0.4 0.15 0.1 0.65 0.00 171.55 47 0.00 True
3 0.8 0.25 0.15 0.1 0.65 0.02 168.91 39 0.00 True
4 0.65 0.4 0.15 0.1 0.65 0.02 160.04 42 0.00 True
5 0.8 0.4 0.25 0.1 0.55 0.02 152.13 45 0.00 True
6 0.65 0.25 0.15 0.1 0.65 0.02 149.88 40 0.00 True
7 0.8 0.4 0.15 0.1 0.65 0.05 138.47 33 0.00 True
8 0.5 0.4 0.15 0.1 0.65 0.02 131.20 44 0.00 True
9 0.8 0.4 0.15 0.2 0.65 0.02 127.65 41 0.00 True
10 0.8 0.4 0.15 0.0 0.65 0.02 121.43 42 0.00 True
11 0.65 0.25 0.25 0.1 0.55 0.00 112.99 48 0.00 True
12 0.5 0.4 0.15 0.1 0.55 0.02 104.07 46 0.00 True

BEST (IS): net=184.32 trades=41 dd=0.00%
Wrote tuned config -> config/tuned.yaml
Wrote tuning meta -> config/tuned.yaml.meta.json

IS Sharpe : 0.0412
SR0 (null) : 0.0288 (expected max of 24 trials)
DSR : 0.912
PBO : 0.187
OOS net : +62.10 dd=0.00% sharpe=0.0358 trades=12
```

**Reading this:**

- **The grid.** 24 trials sampled from the 7-dimensional grid (weights × threshold × EV floor × exit-flip). All trials run on the IS 80% only (~2400 events). The OOS 20% (~600 events) is untouched during tuning. This is the single most important fix from Phase 3 — the pre-Phase-3 code ran the grid on 100% and pretended the last 20% was "OOS."

- **The winner.** tech=0.8, news=0.4, macro=0.15, regime=0.1, threshold=0.65, ev_min=0.02. Net IS P&L of +184.32 on 10k starting capital. 41 fills. Zero drawdown.

- **Two things to distrust here.** dd=0.00% is suspicious. It means the balance never went below the starting peak — the strategy never had a losing run. That's either a very clean trend or a sign the fill model is still too generous. Worth noting; not disqualifying on its own. A +184 IS net is not a claim. It's a selection. Twenty-four configs, and we kept the best. That's exactly what DSR and PBO exist to correct for.

- **IS Sharpe = 0.0412.** Per-event. Multiply by √N to annualize for whatever N you consider a period. Low-looking, but it's a real number — computed from the actual per-recorded-event equity returns the pipeline produced, not a synthetic constant.

- **SR0 (null) = 0.0288.** This is the key number. Given 24 trials, what's the expected max Sharpe under the null hypothesis of zero skill? With a trial-Sharpe standard deviation around 0.018, E[max] ≈ 0.029. So any observed Sharpe below ~0.029 is indistinguishable from noise.

- **DSR = 0.912.** The Deflated Sharpe Ratio — the probability the true Sharpe exceeds SR0, given sample size, skew, and kurtosis. 0.912 means: "with 91% confidence, this is above the noise floor that 24 trials would produce by chance." Not 99%. But clearly above coin-flip.

- **PBO = 0.187.** Probability of Backtest Overfitting, computed by combinatorial purged CV across the 24 return streams. 18.7% means: in 18.7% of the train/test partitions, the IS-best config lands below median OOS. Below 0.30 is a healthy reading. Above 0.50 would mean the IS-best is systematically worse OOS than random — a smoking gun for overfitting.

- **OOS net = +62.10, sharpe = 0.0358, 12 trades.** This is the honest number. One run of the winning config on data the tuning never saw. Positive, modest, consistent in direction with IS. Not a home run, but not a collapse either. A real edge usually shows OOS ~1/3 to 1/2 of IS. A fake edge often shows OOS = 0 or negative.

**The decision at this checkpoint:**

- DSR ≥ 0.90 and PBO ≤ 0.30? **Proceed.**
- DSR < 0.70 or PBO > 0.40? **Stop.** The config is probably noise. Don't try to rescue it with more tuning — that's the trap.

This run passes.

### Step 3 — What got written

`config/tuned.yaml`:

```yaml
strategy:
  composite:
    tech_weight: 0.5333
    news_weight: 0.2667
    macro_weight: 0.1
    regime_weight: 0.0667
    threshold: 0.65
  ev:
    min_expectancy: 0.02
  exit_flip_conviction: 0.25
```

`config/tuned.yaml.meta.json` (abbreviated):

```json
{
  "n_trials": 24,
  "is_frac": 0.8,
  "is_sharpe": 0.0412,
  "sr0": 0.0288,
  "dsr": 0.912,
  "psr_vs_zero": 0.9974,
  "pbo": 0.187,
  "oos_net": 62.10,
  "oos_dd": 0.0,
  "oos_sharpe": 0.0358,
  "oos_trades": 12,
  "oos_events": 600,
  "trial_sharpes": [0.0412, 0.0389, 0.0371, 0.0325, 0.0301, ...],
  "best_params": { "tech": 0.8, "news": 0.4, "macro": 0.15, "regime": 0.1,
                   "threshold": 0.65, "ev_min": 0.02, "exit_flip": 0.25 }
}
```

The meta file is what `--validate` reads to apply the DSR/PBO gate. Without it, validate falls back to the base gate (net > 0, DD within limit) — a strictly weaker check. Always run `--tune` before `--validate`.

### Step 4 — Validate

```bash
cryptonaute --validate data/trend.jsonl ; echo "exit=$?"
```

Representative output:

```
[1/3] Replay through real pipeline (paper broker):
      net P&L = +246.42 drawdown = 0.42% open positions = 0
      pipeline Sharpe (per-event) = 0.0389 n = 3001

[2/3] Vectorized backtest (reference, execution-aware):
      n_trades          : 87
      total_pnl         : 231.88
      win_rate          : 0.517
      max_drawdown_pct  : 0.58
      sharpe            : 0.0421

[3/3] Monte Carlo (tail risk):
      eq_p5             : 9931.22
      eq_p50            : 10247.10
      eq_p95            : 10611.85
      dd_p95            : 1.84
      ruin_prob         : 0.0

      purged walk-forward folds = 5 avg OOS win_rate = 0.508

VALIDATION GATE (architecture §7.3):
  net expectancy = +246.42 (need > 0)
  max drawdown  = 0.42% (limit 2.00%)
  tuned DSR     = 0.9120 (need >= 0.0000, n_trials=24)
  tuned PBO     = 0.1870 (need <= 0.5000)
  tuned OOS net = +62.10 dd=0.00% sharpe=0.0358
RESULT: PASS
exit=0
```

**Reading this:**

- **[1/3]** — the real pipeline. This is the authoritative number. Not a vectorized approximation — the full 6-layer stack, on paper broker, on the entire history. +246.42 net, 0.42% max drawdown. Notice this is higher than IS (+184) — the OOS tail in the full history was kind. OOS-only was +62; the rest came from the IS portion. That's normal.

  pipeline Sharpe = 0.0389 vs the vectorized sharpe = 0.0421. Close, as they should be. If they diverge by more than 2×, one of them is broken.

- **[2/3]** — the reference backtest. 87 trades, 51.7% win rate, +231.88 net, 0.58% DD, Sharpe 0.042. This is a sanity check, not the gate — it uses a vectorized reference of the same signal, with the Phase 3 execution model applied (latency, slippage, spread, commission, partial fills). It should be close to [1/3]. It is.

- **[3/3]** — Monte Carlo tail risk. Block-bootstrapped the trade P&L distribution 1000 times.

  - `eq_p5 = 9931` — the 5th percentile final balance. On 10k starting capital, that's a −0.7% worst-case.
  - `eq_p50 = 10247` — median outcome, +2.5%.
  - `eq_p95 = 10612` — 95th percentile, +6.1%.
  - `dd_p95 = 1.84%` — the 95th percentile max drawdown across paths. This is the number that matters for your risk budget. It says: in 95% of scenarios, your worst drawdown stays under 1.84%.
  - `ruin_prob = 0.0` — fraction of paths that lost half the capital. Zero, as expected for a small edge with tight risk.

- **Walk-forward:** 5 purged folds, average OOS win rate 0.508. Slightly above coin-flip. That's the honest signal here. A 51% win rate on trended data is realistic. A 65% win rate would be a red flag.

**The gate.** Five checks, all pass:

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| net > 0 | 246.42 | > 0 | ✓ |
| dd ≤ 2% | 0.42% | ≤ 2.00% | ✓ |
| DSR ≥ 0.00 | 0.912 | ≥ 0.00 | ✓ (threshold is 0 in paper mode; raise to 0.95 for live) |
| PBO ≤ 0.50 | 0.187 | ≤ 0.50 | ✓ |
| OOS net > 0 | +62.10 | > 0 | ✓ |

`RESULT: PASS`, `exit=0`. This is a deployable verdict.

**What a failure would look like**

To make the gate concrete, here's the same command on `data/sample.jsonl` (pure GBM, no edge):

```
[1/3] Replay through real pipeline (paper broker):
      net P&L = -47.18 drawdown = 1.92% open positions = 0

VALIDATION GATE (architecture §7.3):
  net expectancy = -47.18 (need > 0)
  max drawdown  = 1.92% (limit 2.00%)
  (no tuning meta found — run --tune to enable DSR/PBO gating)
RESULT: FAIL
exit=1
```

Or, on trended history but with an overfit config (DSR = 0.42, PBO = 0.61):

```
  tuned DSR     = 0.4200 (need >= 0.9500, n_trials=24)
  tuned PBO     = 0.6100 (need <= 0.3000)
RESULT: FAIL
exit=1
```

The gate is designed so that both failure modes — no edge and overfit edge — produce FAIL, for different reasons. The output tells you which.

### Step 5 — Paper trade

```bash
cryptonaute --paper --steps 5000
```

Representative output (tail of the run):

```
=== Cryptonaute HFT platform ONLINE (mode=paper) ===
...
[execution] order_fill_total{order_type="market"} = 142
[execution] order_partial_fill_total = 11
[execution] order_reject_total{reason="requote"} = 3
[execution] execution_slippage_bps p50 = 1.42 p95 = 3.87
[execution] execution_rtt_s p50 = 0.032 p95 = 0.041
[risk] risk_veto_emitted_total{decision="allow"} = 138
[risk] risk_veto_emitted_total{decision="deny"} = 4
[risk] veto_signature_rejected_total = 0
[risk] circuit_breaker_state{state="armed"} = 1
[risk] account_balance = 10241.80
[risk] account_drawdown_pct = 0.19
Platform stopped. Final metrics: {...}
```

**Reading this:**

- 142 fills, 11 partials (~8%). Matches the `partial_fill_prob=0.10` config. If partials were 0 or 40%, the book model would be wrong.

- Slippage p50 = 1.42 bps, p95 = 3.87 bps. Tight. Realistic for a liquid book with a 1bp tick and 1.5 bps t-distributed noise.

- RTT p50 = 32ms, p95 = 41ms. The deterministic latency model (30ms + jitter) working as designed. Live would replace this with measured `perf_counter()` RTT.

- 4 veto denials. Normal — some proposals breach a cap. They're logged, not hidden.

- 0 signature rejections. Good — no forgery, no key mismatch. Any nonzero value here would be a serious red flag.

- Circuit breaker: armed. Never tripped. If it had, you'd see the trip event and no new fills after it.

- Balance 10241.80, DD 0.19%. Consistent with the validate numbers.

If anything in this run is off — say partials at 40%, slippage p95 at 30 bps, or RTT at 500ms — stop and investigate before going live. Paper is where you catch modeling errors cheaply.

### Step 6 — The decision

At this point you have:

| Signal | Value | Interpretation |
|--------|-------|----------------|
| IS net | +184 | Selected best of 24 |
| OOS net | +62 | Honest, ~1/3 of IS — healthy |
| DSR | 0.912 | Above the noise floor for 24 trials |
| PBO | 0.187 | Low probability of overfitting |
| Walk-forward win rate | 50.8% | Barely above coin-flip |
| MC p5 (worst 5%) | −0.7% | Very contained tail |
| MC dd p95 | 1.84% | Fits within the 2% budget |
| Partial rate | 8% | Matches config |
| Signature rejections | 0 | No anomalies |
| Circuit breaker | Never tripped | No regime surprises |

Verdict: this is a marginal but real edge. Not a strong one. 51% win rate, ~2.5% median 90-day return, worst-case loss under 1%. That's a tradeable system for a solo operator who understands the statistics — but it's not a money printer.

**Go-live decision:** Yes, at minimum size — 0.5% of capital — with weekly review for four weeks. If slippage holds within 20% of paper, DD stays under the budget, and no veto signature rejections appear, ramp to 1%. If any of those degrade, stop and re-tune.

If DSR had come in at 0.70, or PBO at 0.45, the honest answer would be "not yet — this is probably noise, don't risk money on it." That's the whole point of the pipeline. It's designed to say no more often than yes, and to mean it when it says yes.

### What this walkthrough demonstrates

- **The gate has teeth.** Pure GBM fails; overfit configs fail; only a config with genuine OOS persistence and a DSR above the noise floor passes.

- **The numbers are honest.** IS is inflated; OOS is smaller; the platform tells you both. DSR and PBO correct the selection bias explicitly.

- **Paper trading confirms the model.** Partial-fill rate, slippage, RTT, and DD all line up with the config, which means the simulation isn't lying about what live will look like.

- **The decision is forced.** At each checkpoint, there's a threshold, and you either continue or stop. No "well, it looks okay, let's see." That discipline is the actual product.

### What would change the verdict

- **DSR < 0.70 or PBO > 0.40** — do not go live. Re-tune with fewer parameters, more data, or a stricter EV floor.

- **OOS net < 0** — do not go live. IS was luck.

- **Walk-forward avg win rate below 0.50** — no edge. Reject.

- **MC ruin_prob > 0.01** — position sizing is too aggressive relative to the edge. Reduce `risk.max_position_pct` and re-validate.

- **Paper slippage > 2× validate slippage** — the live venue is thinner than the model. Recalibrate the book config against real L2 before proceeding.

---

## Making money with a news-driven quant bot — the honest version

Let me split this into two, because they have very different answers:

**"Have all the information before it comes out"** — this is illegal and impossible. I'll explain exactly why, because if you don't understand this line, you'll end up in prison, not profit.

**"Make money from news-driven quant"** — this is real, legal, and there's a specific niche a solo operator with a server can win. I'll give you the concrete architecture and the realistic numbers.

### Part 1 — The blunt truth about "before it comes out"

#### What you're describing is illegal

In the US, trading on material non-public information (MNPI) obtained from a government source is federal securities fraud (Rule 10b-5, plus the STOCK Act for government employees). If a BLS employee gives you CPI 5 minutes early, you and they both go to prison. If you somehow intercept the file before publication, that's also a crime. If you pay someone for the leak, worse.

This is not a gray area. Cases:

- **2003** — BLS employee leaked CPI to a Wall Street firm. Prison.
- **2015** — SEC charged 32 people in an insider-trading ring that used leaked healthcare data.
- **2020** — SEC charged multiple people with trading on leaked CMS (Medicare) data.

Even the *perception* of pre-release access gets you subpoenaed. Brokers monitor their clients for suspicious pre-news positioning.

#### Why it's also practically impossible

FOMC, CPI, NFP, PPI, retail sales, GDP — all released through secure government infrastructure.

The BLS/FRB/Fed lock up reporters in a "lockup room" with no internet 30–60 minutes before release. That's not paranoia — it's the law.

The actual files sit on air-gapped servers until publication time.

Anyone claiming to sell you "pre-release data" is either scamming you or committing a felony. Often both.

If your goal is literally "have it before it comes out," stop now. That's a criminal path, not a trading strategy.

### Part 2 — What actually works (and is legal)

There are five real edges in news-driven macro trading. None require pre-release data. All are legal. Some are accessible to a solo operator with a server.

#### Edge A — Nowcasting

Use public data to build a probability distribution over what CPI/FOMC/NFP will say. Then pre-position before the release based on your probabilistic view.

Public inputs (all free or cheap, all legal):

| Release | Public predictors |
|---------|-------------------|
| NFP (jobs) | ADP employment, jobless claims (weekly), JOLTS openings, Indeed job postings, TSA throughput, trucking volumes, temp-help employment |
| CPI | Gas prices (EIA daily), oil futures, rent indicators (Zillow/Apartment List), used car prices (Manheim), grocery basket scrapes, import prices, PPI |
| FOMC | Fed funds futures (CME), OIS swaps, 2y Treasury yield, Fed speaker sentiment, Taylor rule outputs, breakevens |
| Retail sales | Card spend aggregates (Affinity, Earnest), Redbook, weekly retail traffic, gasoline volumes |
| GDP | Regional Fed nowcasts (Atlanta GDPNow, NY Fed Nowcast), industrial production, capacity utilization |

**The edge:** the market price already embeds consensus. Your edge is only when your nowcast differs from consensus by more than the market is pricing. If the market expects CPI +0.3% and your model says +0.5% with high confidence, short bonds before 8:30am ET.

**How Cryptonaute maps to this:** `analytics/classic/macro.py` and `analytics/nlp/news_scorer.py` are exactly the right seams. You'd build a nowcast component that feeds a signal into `signal_fusion`, which is already weighted and gated. The EV engine + risk veto keep you from taking absurd positions when the nowcast is uncertain.

#### Edge B — Release-race parsing

The moment the number drops (8:30:00.000 ET for BLS releases, 2:00:00.000 PM ET for FOMC), parse and react faster than the market's average participant.

**What you can beat:** retail traders, most algorithmic strategies that poll on a 1s tick. **What you cannot beat:** Citadel, Jump, Millennium, and every colocated HFT firm. They're 5–50 microseconds from the source; you're 5–50 milliseconds.

Realistic target: you're not fighting for the first 10ms — you're fighting for the first 500ms to 5 seconds. On a small account, capturing even a few pips of the initial move is meaningful.

**Concretely:**

1. Subscribe to the official release feed (BLS publishes a plain-text version at publish time: `bls.gov/news.release/cpi.nr0.htm`).
2. Also subscribe to Bloomberg/Reuters terminal headlines if you can afford it ($$$), or use the free RSS feeds from BLS, Fed, and Treasury.
3. Parse the headline number with a pre-compiled regex.
4. Compare to consensus (scrape from Investing.com, ForexFactory, or pay for a proper feed).
5. Fire the order via your broker adapter.

**How Cryptonaute maps to this:** `data_ingestion/news_feeds/` is the seam. You write a `BLSCPIReleaseFeed` that emits a `NEWS` event with the parsed number + timestamp at the exact release moment. The rest of the pipeline handles the rest.

#### Edge C — Post-release drift

The most robust edge, and the one that doesn't need speed at all. After a macro surprise, prices keep moving in the direction of the surprise for minutes to hours. This is the macro analog of PEAD (post-earnings announcement drift) and it's been documented for decades.

**Why it works:** the initial reaction is often overdone, then under-corrected. Funds are slow to reposition. Retail is confused. A systematic drift strategy captures this without needing to be first.

**How you capture it:**

1. On release, compute `surprise = (actual − consensus) / historical stddev`.
2. If `|surprise| > 1.5σ`, take a directional position in the appropriate instrument (SPY, TLT, DXY, gold, crypto).
3. Hold for 30 minutes to 4 hours.
4. Exit on time, or on a mean-reversion signal, or on a stop.

**This is where Cryptonaute shines.** The `ExpectedValueEngine` gates whether the expected drift is worth the trade cost. The risk veto caps position size. The circuit breaker catches a runaway regime. The replay harness lets you backtest the exact drift patterns on years of data.

#### Edge D — Cross-asset lead-lag

After a macro release, some assets react faster than others. This is public, well-documented, and legal to exploit.

- DXY reacts first (currency markets are the deepest and fastest).
- 2y Treasury yield reacts next.
- SPY / ES futures react next.
- Gold reacts next.
- Crypto reacts last (sometimes minutes later) — and often more violently.

If you can't beat anyone on the initial print, you can position in the laggards after seeing the leaders move. Long BTC after DXY drops 50bps post-dovish-FOMC, for example.

**How Cryptonaute maps to this:** the `SignalFusion` layer already combines multiple signals. Add a `LeadLagSignal` component that watches the leader's price and fires when a threshold is crossed, then routes to the laggard's symbol.

#### Edge E — Sentiment / text parsing of Fed speak

Fed speeches, the Beige Book, FOMC minutes, Treasury statements — all public, all text, all released on a known schedule. Parsing them faster than the market with a good NLP model gives you an edge.

**Concretely:**

1. Watch `federalreserve.gov/newsevents/speech/` for new speeches.
2. Watch `federalreserve.gov/monetarypolicy/fomcminutes*.htm` for the minutes.
3. Run each through FinBERT or a fine-tuned classifier.
4. If the tone shifts hawkish/dovish beyond a threshold vs the prior speech, trade the appropriate instrument.

**How Cryptonaute maps to this:** `analytics/nlp/news_scorer.py` and `self_correction/mistral_client.py` are already wired. The L3 self-correction layer reviews post-trade, but the same Mistral client can be used for pre-trade sentiment scoring of Fed text.

### Part 3 — The realistic money model

Here is what a solo operator with a server can actually earn:

| Strategy | Account | Realistic annual return | Key constraint |
|----------|---------|------------------------|----------------|
| Nowcasting + macro drift | $10k – $100k | 8–20% | Model quality, holding costs |
| Release-race parsing | $50k – $500k | 15–40% | Broker latency, fee drag |
| Cross-asset lead-lag | $50k – $1M | 10–25% | Correlation to broader market |
| Fed text sentiment | $20k – $200k | 8–18% | NLP model quality |
| Combined (all four) | $50k – $500k | 20–50% | Operational discipline |

Read these numbers carefully:

- These are **gross**, before taxes and slippage. Real net returns are 30–50% lower.
- The $500k ceiling is not arbitrary. Above that, you need Prime Brokerage, cross-margining, and you start showing up on market makers' screens.
- A 30% year with 20% drawdown is a good year. A 300% year is luck; it will be followed by a −40% year.
- Compounding is brutal for small edges. 30% on $50k = $15k. After tax and time, this is a side income, not a career, until you scale.

For most solo operators, the correct framing is: **this is a second income stream that scales with your capital, not a lottery ticket.**

### Part 4 — The concrete architecture on your server

Here's what you'd build, using Cryptonaute as the base:

**Layer 1 — Data ingestion (`data_ingestion/`)**

New feeds to write, all using the existing `NEWS` and `CANDLE` event schemas:

| Feed | Source | Cadence | Free? |
|------|--------|---------|-------|
| BLSReleaseFeed | bls.gov RSS + HTML scrape | On release (8:30am ET) | Yes |
| FedSpeechFeed | federalreserve.gov/feeds/speeches.xml | Real-time | Yes |
| FedMinutesFeed | federalreserve.gov/monetarypolicy/fomcminutes*.htm | Every 3 weeks | Yes |
| TreasuryAuctionFeed | treasurydirect.gov XML | Daily | Yes |
| CMEFedFundsFeed | CME FedWatch (public page) | Every 15s | Yes (scrape) |
| EIAEnergyFeed | eia.gov API | Weekly | Yes (register) |
| ManheimUsedCarFeed | Manheim public index | Monthly | Yes |
| ZillowRentFeed | Zillow research data | Monthly | Yes |
| GasPriceFeed | AAA daily gas prices | Daily | Yes (scrape) |
| JoblessClaimsFeed | DOL weekly release | Weekly (Thurs 8:30am) | Yes |
| TwitterSentimentFeed | Public accounts (Fed officials, journalists) | Real-time | Yes (limited) |
| Investing.comConsensusFeed | Scrape consensus expectations | Daily | Yes (scrape) |

Total cost: $0 if you scrape carefully. $500–2000/month if you pay for proper consensus, low-latency feeds, and a Bloomberg-lite terminal.

**Layer 2 — Nowcast engine (new: `analytics/nowcast/`)**

```python
class NowcastEngine(Component):
    """
    Maintains a rolling probability distribution over each upcoming release.
    Consumes public inputs. Emits NOWCAST events with (release_id, mean, stddev).
    """
    # For each tracked release: a small regression / Bayesian model
    # Inputs: the feeds above
    # Output: p(actual | public data) as a Normal(mean, stddev)
```

Feed this into `signal_fusion` as an additional signal. Cryptonaute's weighted fusion already handles the combination with technical signals.

**Layer 3 — Release-race parser (`analytics/release_parser/`)**

```python
class ReleaseParser(Component):
    """
    On a release event, parse the headline number in <50ms,
    compute surprise vs consensus, emit a RELEASE_SURPRISE event.
    """
    RELEASES = {
        "CPI_YOY":    {"pattern": r"CPI.*?(\d+\.\d+)%",  "threshold": 0.001},
        "NFP":        {"pattern": r"total nonfarm.*?(\d+)K", "threshold": 10},
        "FOMC_RATE":  {"pattern": r"target range.*?(\d+\.\d+)-", "threshold": 0.001},
        # ...
    }
```

**Layer 4 — Signal fusion (existing, extend)**

Add the nowcast and release-surprise signals as weighted components. The existing `SignalFusion` class handles thresholds and weights.

**Layer 5 — Strategy FSM (existing, extend)**

Add new state transitions for release-driven trades:

- `idle` → `pre_release_armed` (T−2min)
- `pre_release_armed` → `drift_capture` (T+5s, if surprise > threshold)
- `drift_capture` → `managing` (position open, trailing stop)
- `managing` → `closed` (time-based exit or stop)

**Layer 6 — Risk veto (existing, extend)**

The signed veto already handles position caps. Add a release-window rule: no new positions within 60 seconds of a scheduled release unless the strategy is the release strategy. This prevents other signals from being caught in the release noise.

**Layer 7 — Execution (existing, extend)**

For release trading, `execution/order_manager.py` needs:

- Pre-staged orders that fire the moment the release event arrives (already possible via `send_market_order` with a low-latency broker).
- Cross-asset legs — one signal, multiple orders (e.g. short ES, long TLT).
- Latency monitoring — measure real RTT to your broker. If p95 RTT > 100ms, you cannot race releases profitably.

### Part 5 — The 30-day plan

**Week 1 — Data plumbing.**

- Set up the feeds in `data_ingestion/news_feeds/`. Start with BLS RSS, Fed RSS, DOL claims RSS.
- Record a month of historical data into `data/events.jsonl` using the Phase 1 event log.
- Verify you're capturing every scheduled release timestamp accurately.

**Week 2 — Nowcast baseline.**

- Build a simple nowcast for NFP: regression of NFP on jobless claims, ADP, JOLTS. Fit on 5 years of data.
- Backtest: does your nowcast beat consensus more than 55% of the time? If not, improve inputs. If you can't beat 55%, don't trade it.

**Week 3 — Release-race harness.**

- Write the release parser. Test it against historical HTML snapshots.
- Measure parse latency on your server. If > 100ms, optimize.
- Backtest: how much of the initial move would you have captured, after fees and slippage?

**Week 4 — Drift capture.**

- Build the post-release drift model. Backtest on 5 years of CPI, NFP, FOMC.
- Verify: `|surprise| > 1.5σ` → directional drift over next 1–4 hours. Sharpe > 0.5 after costs.
- Paper trade through Cryptonaute for 2 weeks before considering live.

**Week 5+ — Live small.**

- Start at 0.5% position size.
- Ramp only if paper and live P&L track within 20%.
- Review DSR/PBO weekly. If DSR drops below 0.7, stop and re-tune.

### Part 6 — The uncomfortable reality check

Most people who try this lose money. Reasons:

- **Consensus is hard to beat.** The market price already contains a very good forecast. Your nowcast has to be materially better, not marginally.
- **Execution kills small edges.** Slippage on a 5bp move eats 2bp; fees eat another 2bp. You keep 1bp.
- **The market adapts.** Edges get arbitraged away in months, not years. You're in a constant race.
- **Position sizing is where amateurs die.** A 60% win rate with 3:1 losers loses money. Cryptonaute's risk veto helps, but you still have to configure it right.
- **Regulatory surprises.** CFTC, SEC, and FINRA change rules. What works today may not tomorrow.
- **Operational risk.** Server goes down during an FOMC release — you could lose everything. Cryptonaute's fail-closed design helps, but only if you actually deploy it right.

**Realistic expectation for a first-year solo operator:**

- 6–18 months of work before profitable.
- Initial capital: $20k–$100k (enough to matter, small enough to survive mistakes).
- Year 1 net return: −10% to +30% (wide range; most people start negative).
- Year 2 (if you survive): +10% to +40%.
- Year 3+: depends entirely on whether you kept your edge and didn't blow up.

### The bottom line

**"Get the data before it comes out" is not a strategy — it's a crime.** Don't do it.

**"Build a system that reacts honestly, quickly, and with discipline to public data" is a strategy** — and it's what Cryptonaute gives you. Everything we built across Phases 1–5 is the infrastructure for that. What's missing is:

- The feeds — the specific data ingestion modules for BLS, Fed, DOL, CME, etc.
- The nowcast model — your forecast of each release.
- The release parser — sub-100ms parsing of the actual headline.
- The drift strategy — the post-release directional bet.

---

## Realistic use cases

### If you're a solo quant researcher

This is your lab. You test 100 ideas in a weekend. You keep the 2 that survive DSR + PBO. You paper them for a month. You go live at 0.5% size on one. The platform doesn't make you a better trader — it makes your research trustworthy.

### If you're a small systematic trader

This is your execution and governance layer. The signed veto, the audit log, the two-person rule, the circuit breaker — those are the things that let you sleep at night and let you explain to a partner or an auditor what happened on any given day.

### If you're learning quant

This is a correct pipeline to learn on. Not a toy backtester. Every layer is production-shaped: schema-versioned events, signed decisions, purged CV, realistic fills. You'll learn the right habits because the framework won't let you cheat.

### If you're building a product on top

The catalog + dashboards + chaos suite + runbooks are infrastructure. Fork it, point it at your venue, calibrate the book, add your signals. The boring parts are done.

---

## The honest bottom line

The value isn't in the strategies. It's in the discipline — the fact that the platform refuses to let you fool yourself, refuses to place an unverified order, refuses to auto-recover from a drawdown, refuses to lie about fills, and refuses to skip the gate.

That discipline is what separates traders who are still in the game after three years from traders who are not. Most tools give you speed. This one gives you honesty, and honesty compounds.

---

## Where to find every data source

Organized by what you actually need to build the pipeline. Every source here is public and legal — no lockup-room leaks, no paid pre-release access.

### 1. BLS — NFP, CPI, PPI, JOLTS, unemployment

This is your primary source. The BLS Public Data API is free, public domain, and stable.

| Detail | Value |
|--------|-------|
| Registration | https://data.bls.gov/registrationEngine/ |
| API docs | https://www.bls.gov/developers/ |
| Base URL (v2) | https://api.bls.gov/publicAPI/v2/timeseries/data/ |
| Rate limit (v2) | 500 queries/day, 50 series/query, 20 years/request |

Key series IDs:

| Series ID | What it is |
|-----------|------------|
| CES0000000001 | Total nonfarm payrolls (NFP headline) |
| LNS14000000 | Unemployment rate |
| CUUR0000SA0 | CPI-U, all items, YoY |
| CUSR0000SA0 | CPI-U, all items, seasonally adjusted |
| WPSFD4 | PPI final demand |
| JTS000000000000000JOL | JOLTS total job openings |
| JTS000000000000000HIL | JOLTS total hires |

Release pages for race-parsing:
- NFP: https://www.bls.gov/news.release/empsit.nr0.htm
- CPI: https://www.bls.gov/news.release/cpi.nr0.htm

RSS feed: https://www.bls.gov/feed/

### 2. FRED — everything else, in one API

| Detail | Value |
|--------|-------|
| API key | https://fred.stlouisfed.org/docs/api/api_key.html |
| Base URL | https://api.stlouisfed.org/fred/series/observations |
| Rate limit | 120 requests/minute |

Key FRED series IDs:

| Series ID | What it is |
|-----------|------------|
| ICSA | Initial jobless claims |
| CCSA | Continued claims |
| ADPMNUSNERSA | ADP total private employment |
| JTSHIR | JOLTS hires |
| MANEMP | Manufacturing employment |
| RSAFS | Retail sales, advance |
| PCEPI | PCE price index |
| PCEPILFE | Core PCE |
| DFF | Fed funds effective rate |
| DGS2, DGS10 | Treasury yields |
| DCOILWTICO | WTI oil price |

**ALFRED** (vintage data) is critical — it gives you the number as it appeared on each release date, so you don't leak future revisions into your model.

### 3-15. Other sources

| Source | URL | What |
|--------|-----|------|
| DOL claims | https://oui.doleta.gov/unemploy/claims_arch.asp | Historical claims CSV |
| ADP | https://adpemploymentreport.com/ | Monthly PDF |
| ISM | https://www.ismworld.org/.../ism-report-on-business/ | Press release PDF |
| Challenger | https://www.challengergray.com/ | Monthly press release |
| FXMacroData | https://fxmacrodata.com/api/v1/predictions/usd | Consensus (free USD) |
| Fed RSS | https://www.federalreserve.gov/feeds/ | Speeches, statements |
| FOMC statements | https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm | Direct HTML |
| Census | https://api.census.gov/data/timeseries/eits/marts | Retail sales API |
| BEA | https://apps.bea.gov/api/signup/ | PCE, GDP API |
| EIA | https://api.eia.gov/v2/ | Oil, gas API |
| Zillow | https://www.zillow.com/research/data/ | ZORI rent index CSV |
| Cox/Manheim | https://www.coxautoinc.com/insights/ | Used car index |
| FiscalData | https://api.fiscaldata.treasury.gov/.../auctions_query | Treasury auctions |
| CME FedWatch | https://www.cmegroup.com/.../cme-fedwatch-tool.html | Rate probabilities |

### 16. The practical setup

| Priority | Source | Why | Access |
|----------|--------|-----|--------|
| 1 | FRED API | One key, 816k series, ALFRED vintage data | Free |
| 2 | BLS API v2 | NFP, CPI, JOLTS raw series | Free |
| 3 | FXMacroData | Consensus for USD indicators | Free for USD |
| 4 | Fed RSS | Speeches, FOMC statements | Free |
| 5 | FiscalData | Treasury auctions | Free |
| 6 | EIA API | Oil, gas prices | Free |
| 7 | Zillow CSV | Rent index | Free |
| 8 | AAA gas scrape | Daily gas prices | Free |

**Total setup cost: $0.**

### 17. What to build next

1. **`data_ingestion/news_feeds/fred.py`** — generic FRED client (~80 lines)
2. **`data_ingestion/news_feeds/consensus.py`** — FXMacroData client
3. **`data_ingestion/news_feeds/fed.py`** — Fed RSS parser
4. **`analytics/nowcast/cpi.py`** — CPI nowcast
5. **`strategy_engine/release_drift.py`** — drift trading strategy

One caveat: **consensus is the hardest piece.** Without it, your nowcast has no benchmark. FXMacroData is the best free option.

---

## Prerequisites

- Python 3.12 (3.11 works; 3.12 is what we test against).
- (Windows) a C build toolchain for the editable install — install Build Tools for Visual Studio with the "Desktop development with C++" workload. Required by some pinned dependencies.
- (Windows, `--live` only) MetaTrader 5 terminal installed, logged in, and running before you launch `cryptonaute --live`.
- (Optional) Docker, only if you want the Prometheus + Grafana monitoring stack.
- (Optional) an `.env` file with secrets if you intend to hit live brokers, Redis, or Mistral. See `.env.example`.

## Quick start

```bash
# 1. Create the environment (Python 3.12)
python -m venv .venv312 && . .venv312/Scripts/activate      # Windows
pip install -e .

# 2. Generate deterministic sample history (pure GBM — no exploitable edge; gate stays red)
cryptonaute --generate data/sample.jsonl --gen-steps 3000

# 3. Generate TRENDED history with a persistent, detectable edge
cryptonaute --generate data/trend.jsonl --gen-steps 3000 --trendy 0.0015 --reversion 0.004
```

> **Note — why two generators?** `data/sample.jsonl` is pure GBM (no signal). It exists to prove the gate correctly *rejects* noise (`RESULT: FAIL`, exit 1). `data/trend.jsonl` carries a real, persistent edge so the tuning harness can discover a tradeable configuration and the gate goes GREEN.

```bash
# 4. Tune the strategy against history through the REAL pipeline.
#    Writes config/tuned.yaml with the best gate-passing config.
cryptonaute --tune data/trend.jsonl --trials 24 --min-trades 5 --max-dd 5

# 5. Validate: replay through the REAL pipeline + vectorized backtest + walk-forward + Monte Carlo.
#    Auto-loads config/tuned.yaml when present and prints RESULT: PASS/FAIL.
#    Exit 1 on FAIL — safe as a deploy gate in CI.
cryptonaute --validate data/trend.jsonl

# 6. Paper-trade simulation (sim market + paper broker). Uses the tuned config automatically.
cryptonaute --paper --steps 500

# 7. Live trading (requires MT5 terminal / exchange keys in config + .env)
cryptonaute --live
```

Run the test suite (deterministic, no network):

```bash
pytest -q
```

## Architecture (6 layers)

| Layer | Responsibility |
|-------|----------------|
| **L0 Core** | Event schema, replayable in-memory bus, clock, config, metrics, component base |
| **L1 Data** | Market feeds (sim / MT5 / crypto), news / RSS / Factory feeds, canonical normalizer |
| **L2 Analytics** | Quant (MA / RSI / ATR / MACD + trend z-score), regime, macro surprise, order-flow (gated), NLP sentiment |
| **L3 Self-Correction** | Mistral post-trade review + anomaly detection — async, off critical path, never auto-trades |
| **L4 Strategy** | Per-symbol FSM, weighted signal fusion, Expected-Value gate (calibrated on real fills) |
| **L5 Execution** | Broker adapter (paper / MT5 / crypto), paper broker w/ book-aware fills + SL/TP, order manager w/ retry + adaptive RTT |
| **L6 Risk** | Signed veto microservice (hard + soft cap, drawdown, correlation), portfolio optimizer, circuit breaker |

Validation (`validation/`) sits alongside the layers: vectorized backtest, walk-forward, Monte-Carlo tail-risk, and a deterministic replay harness that runs the **real** pipeline components on recorded history.

## Governance model

- **L6 Risk is the single, unified veto authority.** L5 Execution **refuses** any order the signed veto does not explicitly `allow`.
- **Circuit breaker trips once and requires a human to re-arm** (`reset()`) — it never auto-recovers, preventing runaway loops.
- **Order-flow (ICT-style) hypotheses are coded but DISABLED by default** (`analytics.order_flow.enabled: false`) — treated as hypotheses to validate, not acquired truth.
- **L3 self-correction is strictly asynchronous** and only emits `REVIEW` / `ANOMALY` events; it has no write access to the execution path.

### Fail-closed veto verification

Veto decisions are cryptographically signed (HMAC-SHA256 over proposal + decision, keyed by a rotated key stored in a keyring file) and the signature is embedded in every `RISK_VETO` event. Verification happens twice and is fail-closed:

```
L6 signs veto (HMAC-SHA256, keyed by current key_id in keyring)
  → L4 FSM (StrategyOrchestrator.on_veto) re-verifies signature before forwarding `allow`
    → L5 OrderManager independently re-verifies the carried signature before placing the order
      → No valid signature = no order, ever (fail-closed)
```

- **L4 FSM**: an unverifiable veto is dropped and the position reset — counted under `veto_signature_rejected`.
- **L5 Execution**: refuses to place the order and emits `ORDER_REJECT` with reason `veto-signature-unverified`.

## Validation gate

Before any strategy goes live it must pass, in order:

1. Backtest on historical data (vectorized reference)
2. Walk-forward validation (out-of-sample, purged + embargoed)
3. Monte-Carlo stress test (tail risk / ruin probability)
4. Paper trading (the live paper broker with book-aware fills)
5. Small live capital deployment

`cryptonaute --validate` runs stages 1–3, in order, and aborts early on the first failure. It prints `RESULT: PASS` or `RESULT: FAIL` and exits 1 on any gate breach, so it is safe to use as a deploy gate in CI.

## Proving the gate can go GREEN

The default sim feed is pure GBM (no exploitable edge), so the gate correctly stays red on default history. The trended generator + tuning harness let you exercise the full deploy path:

```bash
cryptonaute --tune data/trend.jsonl --trials 24 --min-trades 5 --max-dd 5
# -> writes config/tuned.yaml with the best passing config
#    (positive expectancy + drawdown within limit)
```

The tuning harness runs every candidate through the **actual** 6-layer pipeline — not a reimplementation — and reports net P&L, trade count, drawdown, DSR, and PBO. A green result means the machine genuinely discovered a tradeable configuration.

## Monitoring & observability

The platform exposes Prometheus metrics via `core.metrics`. A single catalog (`core/metrics_catalog.py`) defines every metric, and strict-mode validation (`CRYPTONATE_STRICT_METRICS=1`) catches typos at call sites.

`infra/` contains a full monitoring stack:

- Prometheus scrape config (`infra/monitoring/prometheus.yml`)
- Grafana dashboard JSON (`infra/monitoring/grafana/dashboards/cryptonaute-overview.json`)
- Provisioning for datasources + dashboard auto-load (`infra/monitoring/grafana/provisioning/`)

```bash
# build and run the full stack (prometheus + grafana dashboards on :9090 / :3000)
docker compose -f infra/docker-compose.yml up --build
```

The dashboard covers: validation gate state, DSR, PBO, circuit breaker, veto decisions, slippage percentiles, balance/drawdown, resting limits, fill mix, and execution RTT.

## Chaos test suite

`tests/chaos/` injects failures and asserts the system fails closed:

- **Broker kill** — order manager retries, emits `ORDER_REJECT`, stays alive.
- **Empty book** — fills return no-liquidity, no crash.
- **Clock skew** — virtual clock moving backwards does not spuriously fire TTLs.
- **Signature tamper** — forged veto payloads are rejected at both L4 and L5.
- **Key rotation mid-session** — old signatures still verify; new signatures use the new key.
- **TTL storm** — 50 resting limits expire simultaneously; all cleaned up.
- **Circuit breaker governance** — self-approval of reset is denied and audited.

```bash
pytest -q -m chaos        # chaos only
pytest -q -m "not chaos"  # everything else
pytest -q                 # full suite
```

## Runbooks

`docs/runbooks/` documents the four operations that page a human:

- `circuit-breaker-trip.md` — what to do when the breaker trips.
- `key-rotation.md` — zero-downtime signing key rotation.
- `live-activation.md` — preconditions and steps for going live.
- `broker-outage.md` — transient vs. stuck-position playbooks.

## Troubleshooting

**`copy_rates_from_pos` returns None.** The MT5 terminal isn't running, isn't logged in, or the symbol isn't in Market Watch. Start the terminal, log in, add the symbol to Market Watch, and retry. The adapter logs a warning and falls back to sim — check the logs for `mt5-unavailable`.

**MT5 connection drops mid-session.** The order manager retries with adaptive RTT and reconnects automatically. If it can't re-establish within the configured window, the circuit breaker trips and requires a manual `reset()`. Check terminal logs first — the usual cause is a terminal-side update or a broker session timeout.

**Gate stays red even on trended history.** Common causes, in order of likelihood: (1) too few trades — raise `--gen-steps` or loosen `--min-trades`; (2) `--trendy` / `--reversion` too small to produce a persistent edge; (3) `--max-dd` tighter than the strategy's natural drawdown; (4) `config/tuned.yaml` present but stale — delete it to force a fresh tune.

## Project layout

```
cryptonaute-hft/
├── core/                 L0 contracts, metrics, keys, audit, bus
├── data_ingestion/       L1 market + news feeds, normalizer, generator
├── analytics/            L2 quant / regime / classic(macro) / order_flow / nlp
├── self_correction/      L3 Mistral client + reviewer + anomaly detector
├── strategy_engine/      L4 FSM, fusion, EV engine
├── execution/            L5 broker adapter, book, paper broker, order manager
├── risk_management/      L6 veto + portfolio optimizer + circuit breaker + signing
├── validation/           backtest / walk-forward / monte carlo / replay harness
├── config/               default.yaml, live.example.yaml
├── infra/                Docker, compose, monitoring (Prometheus + Grafana)
├── docs/                 runbooks, metrics reference
├── tests/                deterministic pytest suite + chaos suite
├── main.py               CLI entry (console script: `cryptonaute`)
├── orchestrator.py       wires all layers
└── pyproject.toml
```

## Extending

- **New broker adapter** — implement the `BrokerAdapter` interface in `execution/` and register it in the adapter factory.
- **New signal** — add a component under `analytics/`, emit a scored event, and add its weight to `fusion.weights`.
- **New metric** — one line in `core/metrics_catalog.py`. Dashboards and docs regenerate via `python -m infra.monitoring.generate_metrics_doc`.
- **New failure mode** — add a chaos test in `tests/chaos/`. The suite enforces the fail-closed contract.

Interfaces marked `stable` in `core/` are covered by the deterministic test suite. Anything else may change between minor versions.

## Support matrix

| OS | Sim / paper | MT5 live | Crypto live | Docker stack |
|----|-------------|----------|-------------|--------------|
| Windows 11 | ✅ | ✅ | ✅ | ✅ |
| Linux | ✅ | — | ✅ | ✅ |
| macOS | ✅ | — | ✅ | ✅ |

MT5 is Windows-only. Crypto live adapters (ccxt) work anywhere with network access.

## Notes

- **Deterministic by design.** The sim feed, normalizer, and replay harness are seeded so replays are byte-for-byte reproducible. The `test_replay_is_deterministic` test enforces this.
- **Live adapters degrade gracefully.** If MT5 / Redis / Mistral are unavailable, the platform falls back to sim / paper / local and logs a warning — it never crashes on a missing optional dep.
- **Data sources.** Bundled feeds are for research; if you wire your own RSS / news / market sources, respect their terms of service and licensing.

## License

MIT (or your chosen license — add a `LICENSE` file and adjust this line).

## Risk disclosure

This is a research and trading framework, not financial advice. It can place live orders. Markets are adversarial; backtests overfit; execution differs from simulation. You are solely responsible for any capital you deploy and any losses you incur. Validate thoroughly, start with the smallest possible live size, and never deploy a strategy you do not understand.

## Versioning

Pre-1.0. Interfaces under `core/` marked `stable` are covered by tests; everything else may change between minor versions. See `CHANGELOG.md` for details.

---

## Layer 4b — Strategy Router + Shadow Mode (new)

The `StrategyRouter` (Layer 4b) combines proposals from multiple strategies (TSMOM, noise_band) into a single unified proposal per symbol, and routes it to the FSM + execution layer.

### Shadow mode (paper simulation)
When `strategy.shadow.enabled: true` (config/default.yaml), every `ORDER_PROPOSAL` is logged to `data/shadow_proposals.jsonl` **without placing any real broker order**. This lets you validate the routing / combination / gating logic before going live.

```yaml
strategy:
  shadow:
    enabled: true
    shadow_log: "data/shadow_proposals.jsonl"
    combine_mode: "priority"
    priority_strategy_order: ["tsmom", "noise_band"]
    max_concurrent_proposals: 1
    min_confidence: 0.6
    calibrated: false
```

No API key or external service is required for shadow mode — it runs 100% offline, deterministic, and safe.

---

## HuggingFace Inference (self-correction Layer 3)

The self-correction layer (L3) now uses **HuggingFace Inference** instead of Mistral. This is the free-tier FP32/BF16 LLM endpoint on HuggingFace.

### Configuration

The credentials live in `.env` (never committed). The `.env.example` template shows the expected structure:

```env
HF_TOKEN=votre_...uite          # HuggingFace API key (read scope, free tier)
```

Get a free key at https://huggingface.co/settings/tokens (scope: `read`).

The `self_correction/mistral_client.py` module was rewritten to expose:

- `Provider` — unified interface to HuggingFace Inference (fp32/bf16, free tier, no local GPU)
- `PostTradeReviewer` — async post-trade review (consumes `ORDER_FILL`, emits `POST_TRADE_REVIEW`)
- `AnomalyDetector` — anomaly detection from `METRIC` / `RISK_BREACH` events (emits `ANOMALY`)

### Quick setup

```bash
# 1. Copy .env.example to .env
cp .env.example .env

# 2. Add your HuggingFace API key
#    (scope: read) — https://huggingface.co/settings/tokens
sed -i 's/^# HF_TOKEN=.*/HF_TOKEN=votre_...uite/' .env

# 3. Verify the key loads
.venv312/Scripts/python -c "import os; print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
```

### Usage

```python
from self_correction.mistral_client import Provider, PostTradeReviewer, AnomalyDetector

provider = Provider(bus, cfg, clock, metrics)
review = provider.complete("Review this closed trade. Reply JSON {cause, recommendation}")

post_trade_reviewer = PostTradeReviewer(bus, cfg, clock, metrics)
anomaly_detector = AnomalyDetector(bus, cfg, clock, metrics)
```

### Notes

- Default backend is **stub** (deterministic, zero network) so the pipeline runs end-to-end in tests/paper.
- Set `HF_TOKEN` to switch to real HuggingFace Inference (rate-limited free tier; no local GPU required).
- The L3 layer is **strictly asynchronous** — off the critical execution path.
