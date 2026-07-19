# Runbook — Broker outage

## Symptom

- `order_reject_total{reason=~"send-failed|timeout|no-liquidity|requote"}` spiking
- `execution_rtt_s` p95 climbing past the retry backoff window
- `positions_open` frozen while `order_fill_total` stalls

## Immediate actions

1. Do not restart. The order manager retries with exponential backoff by design.
2. Check broker status page / terminal logs (MT5) / exchange status API.
3. Confirm paper vs live: `cryptonaute --check-keys` (safe — no orders).

## If the outage is transient (< 5 min)

- Let `_send_with_retry` run. Backoff caps at `base_backoff_s * 2^max_retry`.
- Watch `order_reject_total` for the reason breakdown. `requote` is normal, not failure.

## If the outage exceeds the drawdown window

- The circuit breaker will trip on the drawdown that accumulates.
- Follow `circuit-breaker-trip.md`. Do not bypass it.

## If positions are stuck open at the broker

1. Read the real state: `broker.get_open_positions()`.
2. Reconcile against `orchestrator.execution._open` (in-process).
3. If they diverge, reconcile by hand and record the discrepancy in the audit log:
   ```python
   orchestrator.audit.write(AuditRecord(ts=now(), action="broker.reconcile",
                                        actor="<you>", approvers=("<a1>",),
                                        outcome="executed", details={"symbols": [...]}))
   ```
4. Do not force-close positions from a script. Close them via the broker UI/API if needed.

## Post-outage

Post-mortem mandatory. Record: duration, cause, P&L impact, position reconciliation outcome,
whether the circuit breaker fired.
