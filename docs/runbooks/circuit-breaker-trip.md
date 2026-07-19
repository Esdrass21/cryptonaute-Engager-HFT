# Runbook — Circuit breaker tripped

## Symptom

- `circuit_breaker_state{state="armed"} == 0` and `{state="tripped"} == 1`
- `circuit_breaker_trips_total` incremented
- Log line: `CIRCUIT BREAKER TRIPPED at X.XX% drawdown — trading halted`
- `RISK_BREACH` event on the bus with `reason="daily-drawdown-limit"`
- No new `ORDER_FILL` events; L4/L5 stop forwarding proposals

## Immediate actions

1. **Do not restart the process.** The breaker is intentionally non-auto-recoverable. A restart
   loses the audit trail and re-arms silently.
2. Inspect `data/audit.jsonl` — last 50 lines — for the trip record and any prior approvals.
3. Inspect `data/events.jsonl` — find the `RISK_BREACH` event; note the drawdown value.
4. Capture state for post-mortem:
   ```bash
   curl -s localhost:9100/metrics | grep -E 'account_|positions_open|circuit_breaker'
   ```

## Decision

Trading is halted. Only a two-person-approved `reset()` re-arms it.

Root cause must be understood before reset. Common causes:

- A correlated position was admitted under a regime the model does not handle.
- A broker outage caused a phantom position to persist.
- A config change widened the risk envelope without re-validation.

## Recovery

1. Notify approvers out-of-band. Get two distinct approver IDs.
2. In-process reset (never from a shell — the CLI has no `--reset-breaker` by design):
   ```python
   orchestrator.circuit.reset(actor="<your-id>", approvers=("<approver-1>", "<approver-2>"))
   ```
3. Verify audit record:
   ```bash
   tail -1 data/audit.jsonl | jq .
   ```
   Expected: `{"action": "circuit_breaker.reset", "outcome": "approved", ...}`
4. Confirm metric: `circuit_breaker_state{state="armed"} == 1`
5. Write a post-mortem in `docs/postmortems/YYYY-MM-DD-<slug>.md` with: what triggered, what was
   approved, what changes to risk config are being made to prevent recurrence.

## Never

- Never bypass the breaker by editing `risk.max_daily_drawdown_pct` in-place and restarting.
- Never reset without approvers. The rule refuses self-approval.
