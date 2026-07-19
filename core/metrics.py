"""Observability primitives: in-process metrics collector + optional Prometheus export.

Every layer emits metrics (latency, reject rate, fill ratio, slippage, P&L) so the
Prometheus/Grafana dashboard (infra/monitoring) can scrape them. When prometheus-client
is not installed, the collector simply retains the latest values in memory — no failure.

Architecture §8: "explicit end-to-end latency, order reject rate, fill ratio, realized
slippage, P&L per strategy — alerts before execution problems become losses."
"""

from __future__ import annotations

import os
import threading
from time import monotonic
from typing import Dict, List, Optional

try:  # pragma: no cover - optional
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _HAVE_PROM = True
except Exception:  # noqa: BLE001
    _HAVE_PROM = False

# --- Phase 5: strict catalog validation (dev/CI) ---
from core.metrics_catalog import MetricKind, spec, validate_usage
_STRICT = os.environ.get("CRYPTONAUTE_STRICT_METRICS", "0") == "1"


class Metrics:
    """Lightweight metrics registry. Drop-in whether or not Prometheus is present."""

    def __init__(self, enable_prometheus: bool = False, port: int = 9100) -> None:
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._prom: Dict[str, object] = {}
        self._prom_enabled = enable_prometheus and _HAVE_PROM
        if self._prom_enabled:
            start_http_server(port)

    # ---- counters -----------------------------------------------------
    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        # --- Phase 5: strict catalog validation (dev/CI) ---
        if _STRICT:
            validate_usage(name, MetricKind.COUNTER, labels.keys() if labels else ())
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value
        if self._prom_enabled and name not in self._prom:
            self._prom[name] = Counter(name, name, list(labels or {})) if labels else Counter(name, name)
        if self._prom_enabled:
            self._prom[name].inc(value)  # type: ignore[attr-defined]

    # ---- gauges -------------------------------------------------------
    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        # --- Phase 5: strict catalog validation (dev/CI) ---
        if _STRICT:
            validate_usage(name, MetricKind.GAUGE, labels.keys() if labels else ())
        with self._lock:
            self._gauges[name] = value
        if self._prom_enabled and name not in self._prom:
            self._prom[name] = Gauge(name, name, list(labels or {})) if labels else Gauge(name, name)
        if self._prom_enabled:
            self._prom[name].set(value)  # type: ignore[attr-defined]

    # ---- histograms ---------------------------------------------------
    def observe(self, name: str, value: float) -> None:
        # --- Phase 5: strict catalog validation (dev/CI) ---
        if _STRICT:
            validate_usage(name, MetricKind.HISTOGRAM, ())
        with self._lock:
            self._histograms.setdefault(name, []).append(value)
            if len(self._histograms[name]) > 10_000:
                self._histograms[name] = self._histograms[name][-5000:]
        if self._prom_enabled:
            if name not in self._prom:
                self._prom[name] = Histogram(name, name)
            self._prom[name].observe(value)  # type: ignore[attr-defined]

    def timer(self, name: str):
        """Context manager / decorator-free latency timer."""
        return _Timer(self, name)

    # ---- snapshots ----------------------------------------------------
    def snapshot(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(self._counters)
        out.update(self._gauges)
        for k, v in self._histograms.items():
            if v:
                out[f"{k}_mean"] = sum(v) / len(v)
                out[f"{k}_p95"] = sorted(v)[int(len(v) * 0.95) - 1]
        return out

    # --- Phase 5: catalog helpers -------------------------------------- #
    def catalog_snapshot(self) -> dict:
        """Return the values of every catalog metric. Used by tests + /metrics."""
        from core.metrics_catalog import all_specs
        out = {}
        for s in all_specs():
            out[s.name] = self.snapshot().get(s.name, 0.0)
        return out

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class _Timer:
    def __init__(self, metrics: Metrics, name: str) -> None:
        self._m = metrics
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self._m.observe(self._name, monotonic() - self._start)
