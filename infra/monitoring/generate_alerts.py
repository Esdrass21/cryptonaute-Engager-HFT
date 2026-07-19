# infra/monitoring/generate_alerts.py
"""
Generate Prometheus Alertmanager rules from core/metrics_catalog.py.

The catalog is the single source of truth. Any metric with an `alerts` tuple
becomes one or more rules here. Do not edit alerts.yml by hand.

Usage:
    python -m infra.monitoring.generate_alerts
"""
from __future__ import annotations

from pathlib import Path

from core.metrics_catalog import all_specs


def render() -> str:
    lines = [
        "# GENERATED from core/metrics_catalog.py — do not edit by hand.",
        "# Regenerate: python -m infra.monitoring.generate_alerts",
        "groups:",
        "  - name: cryptonaute",
        "    rules:",
    ]
    for spec in all_specs():
        for a in spec.alerts:
            lines.extend([
                f"      - alert: {a.name}",
                f"        expr: {a.expr}",
                f"        for: {a.for_}",
                "        labels:",
                f"          severity: {a.severity}",
                f"          layer: {spec.layer}",
                "        annotations:",
                f"          summary: \"{a.summary or a.name}\"",
            ])
            if a.runbook:
                lines.append(f"          runbook: \"{a.runbook}\"")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path("infra/monitoring/alerts.yml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
