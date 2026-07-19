# Runbook — Live activation

## Preconditions (all must be true)

1. `cryptonaute --validate <history>` prints `RESULT: PASS` and exit code 0.
2. `config/tuned.yaml.meta.json` exists with:
   - `dsr >= validation.gate.min_dsr`
   - `pbo <= validation.gate.max_pbo`
3. `config/live.yaml` exists and differs from `default.yaml` only in the approved keys.
4. `cryptonaute --check-keys` reports all required credentials present and authenticated.
5. The keyring is on disk, not env (`core.keys.provider: file`).
6. `RISK_SIGNING_KEY` is **unset** in the environment.

## Steps

1. Two-person approval for `live.activate`:
   ```python
   from risk_management.two_person import TwoPersonRule, Approval, ACTION_LIVE_ACTIVATE
   rule.check(Approval(actor="<you>", approvers=("<a1>", "<a2>"),
                       action=ACTION_LIVE_ACTIVATE, payload={"capital": 1000}))
   ```
2. Deploy with the live config:
   ```bash
   cryptonaute --live --config config/live.yaml --broker crypto --exchange <venue>
   ```
3. Watch the first five minutes:
   - `validation_gate_pass == 1`
   - `circuit_breaker_state{state="armed"} == 1`
   - `rate(order_fill_total[1m]) > 0` and `rate(order_reject_total[1m])` low
   - `histogram_quantile(0.95, rate(execution_slippage_bps_bucket[5m]))` within tolerance
4. Ramp: start with `risk.max_position_pct` at 0.5%, double weekly up to the validated ceiling.

## Abort

If any of: drawdown > 1%, slip p95 > 2× paper, or two consecutive veto signature rejections.

Abort by stopping the process; do not disable the veto service.

## Never

- Never run `--live` with `RISK_ALLOW_ENV_KEYS=1` unless it's a drill.
- Never skip paper trading. The gate is the contract.
