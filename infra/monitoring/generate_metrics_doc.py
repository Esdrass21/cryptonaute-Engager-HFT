"""
Generate docs/metrics.md from core/metrics_catalog.py.

Usage: python -m infra.monitoring.generate_metrics_doc
"""
from pathlib import Path
from core.metrics_catalog import all_specs


def render() -> str:
    lines = [
        "<!-- GENERATED from core/metrics_catalog.py — do not edit by hand. -->",
        "",
        "# Metric catalog",
        "",
        "Run `python -m infra.monitoring.generate_metrics_doc` to regenerate.",
        "",
        "| Name | Kind | Layer | Labels | Help |",
        "|------|------|-------|--------|------|",
    ]
    for s in all_specs():
        labels = ", ".join(s.labels) if s.labels else "—"
        lines.append(f"| `{s.name}` | {s.kind.value} | {s.layer} | {labels} | {s.help} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/metrics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")
