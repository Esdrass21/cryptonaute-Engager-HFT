<!-- GENERATED from core/metrics_catalog.py — do not edit by hand. -->

# Metric catalog

Run `python -m infra.monitoring.generate_metrics_doc` to regenerate.

| Name | Kind | Layer | Labels | Help |
|------|------|-------|--------|------|
| `risk_veto_emitted` | counter | l6_risk | symbol, decision | Count of RISK_VETO events emitted by the veto service. |
| `veto_signature_rejected` | counter | l4_strategy | — | Veto signatures that failed verification at the FSM (L4). |
| `execution_veto_unverified` | counter | l5_execution | — | Veto signatures that failed verification at the order manager (L5). |
| `order_forwarded_to_execution` | counter | l4_strategy | — | Approved proposals forwarded from L4 to L5. |
| `order_blocked_by_risk` | counter | l4_strategy | — | Proposals denied by the risk veto. |
| `order_reject` | counter | l5_execution | reason | Orders rejected before or during send. |
| `order_fill` | counter | l5_execution | order_type | Fills emitted (partial or full). |
| `order_partial_fill` | counter | l5_execution | — | Fills that did not consume the full requested notional. |
| `order_limit_placed` | counter | l5_execution | — | Limit orders that went to rest. |
| `order_limit_expired` | counter | l5_execution | — | Limit orders that expired on TTL. |
| `exit_signal_flip` | counter | l4_strategy | — | Exits initiated by a signal reversal. |
| `exit_filled` | counter | l4_strategy | — | Exits that resulted in a fill. |
| `execution_rtt_s` | histogram | l5_execution | — | Measured round-trip time between proposal and fill. |
| `execution_slippage_bps` | histogram | l5_execution | — | Realized slippage in bps vs the mid at send time. |
| `execution_fill_latency_s` | histogram | l5_execution | — | Wall-clock latency between order send and fill confirmation. |
| `account_balance` | gauge | l6_risk | — | Current account balance (paper or live). |
| `account_drawdown_pct` | gauge | l6_risk | — | Drawdown from equity peak, in percent. |
| `positions_open` | gauge | l6_risk | symbol | Number of open positions. |
| `circuit_breaker_state` | gauge | l6_risk | state | 0 = armed, 1 = tripped. |
| `circuit_breaker_trips` | counter | l6_risk | — | Count of circuit-breaker trips. |
| `resting_orders` | gauge | l5_execution | symbol | Number of open resting limit orders. |
| `resting_order_age_s` | histogram | l5_execution | — | Age of resting orders at fill or expiry. |
| `queue_depth_notional` | gauge | l5_execution | symbol, side | Notional ahead of our resting order at its price level. |
| `validation_dsr` | gauge | validation | — | Deflated Sharpe Ratio of the tuned config. |
| `validation_pbo` | gauge | validation | — | Probability of Backtest Overfitting of the tuned grid. |
| `validation_gate_pass` | gauge | validation | — | 1 = gate PASS, 0 = FAIL. |
| `validation_oos_net` | gauge | validation | — | OOS net P&L for the winning config. |
| `review_emitted` | counter | l3_self_correction | — | Post-trade reviews emitted by Mistral path. |
| `anomaly_detected` | counter | l3_self_correction | — | Anomalies raised by the anomaly detector. |
