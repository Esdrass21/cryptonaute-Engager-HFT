# core/metrics_catalog.py
"""Single source of truth for every metric the platform emits."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass(frozen=True)
class AlertRule:
    name: str
    expr: str
    for_: str = "5m"
    severity: str = "warning"
    summary: str = ""
    runbook: str = ""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: MetricKind
    layer: str
    help: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()
    alerts: tuple[AlertRule, ...] = ()


# --------------------------------------------------------------------------- #
# Catalog                                                                     #
# --------------------------------------------------------------------------- #

_CATALOG: list[MetricSpec] = [
    # --- L4 / L5 / L6 core flow ------------------------------------------ #
    MetricSpec("risk_veto_emitted", MetricKind.COUNTER, "l6_risk",
               "Count of RISK_VETO events emitted by the veto service.",
               labels=("symbol", "decision")),
    MetricSpec("veto_signature_rejected", MetricKind.COUNTER, "l4_strategy",
               "Veto signatures that failed verification at the FSM (L4).",
               alerts=(
                   AlertRule(
                       name="VetoSignatureRejectedAtL4",
                       expr='increase(veto_signature_rejected_total[5m]) > 0',
                       for_="0s",
                       severity="critical",
                       summary="L4 rejected a veto signature",
                       runbook="docs/runbooks/key-rotation.md",
                   ),
               )),
    MetricSpec("execution_veto_unverified", MetricKind.COUNTER, "l5_execution",
               "Veto signatures that failed verification at the order manager (L5).",
               alerts=(
                   AlertRule(
                       name="VetoSignatureRejectedAtL5",
                       expr='increase(execution_veto_unverified_total[5m]) > 0',
                       for_="0s",
                       severity="critical",
                       summary="L5 refused an order",
                       runbook="docs/runbooks/live-activation.md",
                   ),
               )),
    MetricSpec("order_forwarded_to_execution", MetricKind.COUNTER, "l4_strategy",
               "Approved proposals forwarded from L4 to L5."),
    MetricSpec("order_blocked_by_risk", MetricKind.COUNTER, "l4_strategy",
               "Proposals denied by the risk veto."),
    MetricSpec("order_reject", MetricKind.COUNTER, "l5_execution",
               "Orders rejected before or during send.",
               labels=("reason",),
               alerts=(
                   AlertRule(
                       name="HighOrderRejectRate",
                       expr='sum(rate(order_reject_total[5m])) > 0.1',
                       for_="10m",
                       severity="warning",
                       summary="Order rejects above 0.1/s sustained",
                       runbook="docs/runbooks/broker-outage.md",
                   ),
               )),
    MetricSpec("order_fill", MetricKind.COUNTER, "l5_execution",
               "Fills emitted (partial or full).",
               labels=("order_type",)),
    MetricSpec("order_partial_fill", MetricKind.COUNTER, "l5_execution",
               "Fills that did not consume the full requested notional."),
    MetricSpec("order_limit_placed", MetricKind.COUNTER, "l5_execution",
               "Limit orders that went to rest."),
    MetricSpec("order_limit_expired", MetricKind.COUNTER, "l5_execution",
               "Limit orders that expired on TTL."),
    MetricSpec("exit_signal_flip", MetricKind.COUNTER, "l4_strategy",
               "Exits initiated by a signal reversal."),
    MetricSpec("exit_filled", MetricKind.COUNTER, "l4_strategy",
               "Exits that resulted in a fill."),

    # --- Execution quality ----------------------------------------------- #
    MetricSpec("execution_rtt_s", MetricKind.HISTOGRAM, "l5_execution",
               "Measured round-trip time between proposal and fill.",
               buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)),
    MetricSpec("execution_slippage_bps", MetricKind.HISTOGRAM, "l5_execution",
               "Realized slippage in bps vs the mid at send time.",
               buckets=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0),
               alerts=(
                   AlertRule(
                       name="SlippageP95High",
                       expr='histogram_quantile(0.95, rate(execution_slippage_bps_bucket[15m])) > 10',
                       for_="15m",
                       severity="warning",
                       summary="Slippage p95 above 10 bps",
                       runbook="docs/runbooks/broker-outage.md",
                   ),
               )),
    MetricSpec("execution_fill_latency_s", MetricKind.HISTOGRAM, "l5_execution",
               "Wall-clock latency between order send and fill confirmation.",
               buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0)),

    # --- Account / risk -------------------------------------------------- #
    MetricSpec("account_balance", MetricKind.GAUGE, "l6_risk",
               "Current account balance (paper or live)."),
    MetricSpec("account_drawdown_pct", MetricKind.GAUGE, "l6_risk",
               "Drawdown from equity peak, in percent.",
               alerts=(
                   AlertRule(
                       name="DrawdownWarning",
                       expr='account_drawdown_pct > 1.0',
                       for_="5m",
                       severity="warning",
                       summary="Drawdown above 1%",
                       runbook="docs/runbooks/circuit-breaker-trip.md",
                   ),
               )),
    MetricSpec("positions_open", MetricKind.GAUGE, "l6_risk",
               "Number of open positions.",
               labels=("symbol",)),
    MetricSpec("circuit_breaker_state", MetricKind.GAUGE, "l6_risk",
               "0 = armed, 1 = tripped.",
               labels=("state",)),
    MetricSpec("circuit_breaker_trips", MetricKind.COUNTER, "l6_risk",
               "Count of circuit-breaker trips.",
               alerts=(
                   AlertRule(
                       name="CircuitBreakerTripped",
                       expr='increase(circuit_breaker_trips_total[1m]) > 0',
                       for_="0s",
                       severity="critical",
                       summary="Circuit breaker tripped",
                       runbook="docs/runbooks/circuit-breaker-trip.md",
                   ),
               )),
    MetricSpec("resting_orders", MetricKind.GAUGE, "l5_execution",
               "Number of open resting limit orders.",
               labels=("symbol",)),
    MetricSpec("resting_order_age_s", MetricKind.HISTOGRAM, "l5_execution",
               "Age of resting orders at fill or expiry.",
               buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)),
    MetricSpec("queue_depth_notional", MetricKind.GAUGE, "l5_execution",
               "Notional ahead of our resting order at its price level.",
               labels=("symbol", "side")),

    # --- Data feeds ------------------------------------------------------ #
    MetricSpec("fred_poll_failed", MetricKind.COUNTER, "l1_data",
               "FRED poll attempts that returned an error.",
               labels=("series",),
               alerts=(
                   AlertRule(
                       name="FredPollingFailing",
                       expr='increase(fred_poll_failed_total[15m]) > 5',
                       for_="15m",
                       severity="warning",
                       summary="FRED poll failures",
                   ),
               )),

    # --- Validation ------------------------------------------------------ #
    MetricSpec("validation_dsr", MetricKind.GAUGE, "validation",
               "Deflated Sharpe Ratio of the tuned config."),
    MetricSpec("validation_pbo", MetricKind.GAUGE, "validation",
               "Probability of Backtest Overfitting of the tuned grid."),
    MetricSpec("validation_gate_pass", MetricKind.GAUGE, "validation",
               "1 = gate PASS, 0 = FAIL."),
    MetricSpec("validation_oos_net", MetricKind.GAUGE, "validation",
               "OOS net P&L for the winning config."),

    # --- L3 self-correction (async) ------------------------------------- #
    MetricSpec("review_emitted", MetricKind.COUNTER, "l3_self_correction",
               "Post-trade reviews emitted by Mistral path."),
    MetricSpec("anomaly_detected", MetricKind.COUNTER, "l3_self_correction",
               "Anomalies raised by the anomaly detector."),
]


def all_specs() -> list[MetricSpec]:
    return list(_CATALOG)


def by_layer() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for s in _CATALOG:
        out.setdefault(s.layer, []).append(s)
    return out


def names() -> list[str]:
    return [s.name for s in _CATALOG]


def spec(name: str) -> MetricSpec:
    for s in _CATALOG:
        if s.name == name:
            return s
    raise KeyError(f"unknown metric {name!r}")


def validate_usage(name: str, kind: MetricKind, labels: Iterable[str] = ()) -> None:
    """Assert-time helper. Raises if a metric is used with the wrong kind/labels."""
    s = spec(name)
    if s.kind != kind:
        raise AssertionError(f"metric {name} is {s.kind}, called as {kind}")
    if s.labels and set(labels) != set(s.labels):
        raise AssertionError(
            f"metric {name} expects labels {s.labels}, got {tuple(labels)}"
        )
