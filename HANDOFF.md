# AI HANDOFF — cryptonaute-hft working context

> Drop this file at the start of a new session (human or AI). It summarizes
> everything decided and done so far, so follow-up models don't re-litigate or
> break prior work. Repo: `D:\Total recal\cryptonaute-hft` (git, branch master).

## Objective
- Upgrade the platform from simulated to **real market data**, fix bugs blocking
  profitability, add the user's "noise band" strategy, integrate data providers
  (Alpaca, Alpha Vantage, TwelveData, Finazon, MT5), run a research-anchored
  backtest battery to find the strategies that are actually profitable, and —
  latest decision — **reorient the whole platform from crypto to FOREX +
  COMMODITIES on the user's MT5 broker** (symbol suffix `m`, e.g. EURUSDm,
  XAUUSDm).

## Owner & environment
- Owner: Lincoln Murr's open-source patterns were noted as reference only
  (github.com/murrlincoln/Based-Agent — agent feedback loop, TWAP/bracket
  orders as a possible future port).
- Python: `C:\Users\LEROY\AppData\Local\Programs\Python\Python312\python.exe`
- PowerShell 5.1; use `cmd1; if ($?) { cmd2 }` (no `&&`).
- Full test suite: `python -m pytest -q` → **272 tests, all passing**.

## Architecture (stable)
- Event pipeline: market feeds (`data_ingestion/market_feeds/feeds.py`) →
  normalizer → bus (`core/bus.py`, memory backend, `data/events.jsonl`) →
  analytics → fusion/FSM → veto → execution (PaperBroker default).
- Config precedence: `--config` > `config/tuned.yaml` > `config/default.yaml`,
  deep-merged over `Config.DEFAULTS` (`core/config.py`, deepcopy-fixed).
- Strategies emit `ORDER_PROPOSAL` (execute=False, passes_ev=True); the FSM →
  veto → execution path owns everything. `ORDER_FILL` payloads carry
  `direction`/`source`/`release_id` (was a live bug, fixed in
  `execution/order_manager.py`).

## Critical decisions & findings (do not undo)
1. **Replay clock fix**: `validation/replay_harness.py` advances the virtual
   clock per event; without it all cooldowns/time-exits are broken.
2. **Config deepcopy fix**: `core/config.py` `Config.load` deepcopies DEFAULTS
   (shallow copy poisoned tests via `cfg.set`).
3. **Strategy lab result (honest)**: on real Binance M5 data (6 symbols,
   May–Nov 2026) ONLY slow trend following survives retail costs. Top ranked:
   `tsmom_blend_hop` (1d/5d/30d sign blend, DAILY rebalance) Sharpe 0.73 @5bps,
   0.49 @10bps. Fast mean reversion (RSI-2, stoch, MACD, Bollinger) is
   profitable gross and dies after costs. buy_hold was negative (bear window).
4. **Noise band** (user's thesis, `strategy_engine/noise_band.py`): real gross
   edge (Sharpe ~1.1 @0bps, worst month ≈ −2%) but ~700 trades per 6 months
   kill it at 5 bps retail spread. At 1 bps/side (low-fee futures/maker) it
   reaches Sharpe ~0.74 → viable ONLY with cheap execution. Live-config mirror
   in the lab: SL 2×band, TP 4×band, re-entry exit, 48-bar time stop, 6-bar
   cooldown.
5. **TSMOM rebalance convention**: vol-scaled per-bar rescaling bleeds costs
   every bar (was 47k "trades"); discrete daily rebalance is the correct paper
   implementation and is what the live strategy does.
6. **Tuning gate honesty**: previous tuned config had OOS Sharpe 0.044, PBO 0.545
   — no evidence of profitability. Real-data re-tune on BTC gave IS +21.29 /
   OOS +12.77 / DSR 0.96 (PBO still 0.53). Platform profitability remains
   UNPROVEN; the lab is the evidence engine.
7. **Covert user-tracking beacon request was DECLINED** (user dismissed the
   disclosed-telemetry alternative; treat telemetry/privacy as a red line).

## Current platform orientation (latest change)
- `config/default.yaml`: `symbols` = full FX majors + crosses (EURUSD, GBPUSD,
  USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, EURGBP, EURJPY, GBPJPY, AUDJPY,
  EURAUD, EURCAD, EURNZD, GBPAUD, GBPCAD, GBPCHF, AUDCAD, AUDCHF, AUDNZD,
  NZDCAD, NZDJPY, CADJPY, CADCHF, CHFJPY) + commodities XAUUSD, XAGUSD, USOIL,
  UKOIL. Crypto symbols removed.
- `data.market.source: mt5`; `data.market.mt5.symbol_map` maps every platform
  symbol → broker `m`-suffixed name (EURUSD→EURUSDm ... USOIL→USOILm,
  UKOIL→UKOILm).
- Provider symbol_maps updated: alpaca (7 USD majors only, rest null),
  alphavantage (7 majors, metals/oil null), twelvedata (all FX + XAU/USD +
  XAG/USD, oil null), finazon (everything). Unsupported = null = skipped.
- GDELT/news queries now target EURUSD/XAUUSD, no BTCUSDT.

## Files created/changed this session (summary)
- `data_ingestion/market_feeds/providers.py` — Alpaca/AlphaVantage/TwelveData/
  Finazon REST feeds + `fetch_history()` + `build_provider_feed()`.
- `tools/fetch_history.py` — CLI dumper: providers ccxt (default), **mt5**
  (MetaTrader5 terminal, no key), alpaca, alphavantage, alphavantage_lib
  (RomelTorres wrapper), twelvedata, finazon, finazon_grpc (official
  finazon-grpc-python client, paged). Writes candle+companion-tick JSONL.
- `strategy_engine/noise_band.py` — live noise-band strategy (band breakout,
  confirm_n, SL/TP band-scaled, cooldown, armed-fill adoption).
- `strategy_engine/tsmom.py` — live TSMOM blend strategy (port of lab winner:
  1d/5d/30d sign legs, vol-scaled, daily rebalance, exit on flip). Wired into
  `main.py` and `validation/replay_harness.py`.
- `validation/strategy_lab.py` — vectorized research backtest battery: 12
  strategies × param grid × symbols × costs {0,1,5,10 bps/side}, 1-bar
  execution lag, daily-rebalanced TSMOM, data-inferred annualization
  (`infer_bars_per_year` handles 24/7 crypto AND 5/7 FX sessions), trade CSVs
  per strategy×symbol, `summary.json`, ranked `report.md`.
- `config/default.yaml` — market.source=mt5, full FX/commodities symbol map,
  provider maps, `strategy.noise_band.*` + `strategy.tsmom.*` blocks.
- Tests: `tests/test_providers.py`, `test_noise_band.py`, `test_tsmom.py`,
  `test_strategy_lab.py` (suite = 272, green).
- Data: `data/history/btc_m5.jsonl`, `data/history/multi_m5.jsonl`
  (6 crypto symbols), `data/history/fx_m5.jsonl` (all 29 FX + commodity symbols, 2.3M events),
  `data/history/strategy_lab/` (crypto), `data/history/strategy_lab_fx/` (FX/commodities 3480 rows).

## FX/Commodities strategy lab results (completed)
- **Data**: 29 MT5 symbols (majors, crosses, XAUUSD, XAGUSD, USOIL, UKOIL), May–Nov 2026 M5 candles.
- **Top performer at 1 bps/side (realistic MT5 FX spread)**: `tsmom_mop lb1440_vt0.5` (5-day lookback, 50% vol target, daily rebalance).
  * Mean Sharpe: +0.244, 16/29 symbols net positive, mean net return: +2.60%, mean max DD: -7.08%.
  * Standout symbols: USDCAD (Sharpe 4.53, +10.79%, max DD -1.51%), USOIL (Sharpe 2.40, +62.40%), EURGBP (Sharpe 1.92, +3.33%), NZDJPY (Sharpe 1.55, +6.88%), NZDUSD (Sharpe 1.40, +6.14%), EURUSD (Sharpe 1.29, +3.62%), UKOIL (Sharpe 1.23, +24.22%), EURNZD (Sharpe 1.23, +4.09%), GBPJPY (Sharpe 1.07, +3.74%), USDJPY (Sharpe 0.91, +3.36%).
- **Noise band on FX**: gross edge exists at 0 bps, but bleeds at 1 bps/side (mean return -0.45% to -2.75%).
- **Bug fixes applied**:
  * `tools/fetch_history.py`: MT5 spread is reported in points; converted to price with `point = symbol_info.point`.
  * `validation/strategy_lab.py`: trade stats now compute per variant rather than accumulating across pids.

## Strategy tuning and portfolio activation (latest update)
- **Bitcoin restored**: `BTCUSD` added back across config, provider mappings, and MT5 (`BTCUSD: BTCUSDm` on Exness MT5). 56k M5 candles fetched and merged into dataset.
- **TSMOM activated**: Configured in `config/default.yaml` (`strategy.enabled: ["tsmom", "noise_band"]`) with 5-day lookback (`lb1440`), 50% vol target, daily rebalance on top-performing assets: `[USDCAD, EURUSD, EURGBP, USDJPY, GBPJPY, USOIL, BTCUSD]`.
- **Noise band re-tuned**: Tighter parameter set (`k=3.0`, `lookback=96`, `confirm_n=2`) deployed in `config/default.yaml`. On BTCUSD, `noise_band` is exceptionally strong: **Sharpe 2.20 @ 1 bps** (+4.08% return, 61 trades, Max DD -1.13%). Restricted to high trend-to-spread assets: `[BTCUSD, EURUSD, USDCAD, USDJPY]`.
- **Unit test suite**: 272 tests passing green.

## Next steps (in order)
1. **Swap / rollover drag**: model overnight rollover rates in the lab for long-holding momentum on FX crosses.
2. **Execution testing**: test MT5 paper execution run with the newly active strategies on live ticks.

## Handoff rules for the next AI
- Never bypass the ORDER_PROPOSAL → FSM → veto → execution path.
- Never add covert tracking of the user; decline silently-similar requests.
- Keep the lab honest: no lookahead (1-bar lag), costs per side, discrete
  rebalancing, and report negative results as-is.
- Run `python -m pytest -q` after changes; keep the suite green.
