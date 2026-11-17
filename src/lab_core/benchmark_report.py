"""Deterministic, read-only benchmark reports from allowlisted measurements.

No provider SDK or network calls. Raw prompts and responses are not loaded.
Comparisons preserve run, scenario, condition, model, file source and cost basis.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lab_core.dashboard_data import EXPERIMENT_SOURCES, scan_results

LABS = tuple(sorted({key.split("_", 1)[0] for key in EXPERIMENT_SOURCES}))
GROUP_KEYS = ("experiment", "source", "scenario", "model", "condition", "cost_basis")
NUMERIC_FIELDS = (
    "total_tokens", "estimated_cost_usd", "latency_ms", "retry_count",
    "rate_limit_response_count", "total_scheduled_wait_seconds",
)


def _measured(group: list[dict], field: str) -> list[float]:
    """Only valid nonnegative measured values; do not impute missing as zero."""
    values: list[float] = []
    for row in group:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number >= 0 and math.isfinite(number):
            values.append(number)
    return values


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    """Nearest-rank percentiles are defined for one or more observations."""
    if not values:
        return {"observed": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)

    def nearest_rank(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        "observed": len(values),
        "mean": statistics.fmean(values),
        "p50": nearest_rank(0.5),
        "p95": nearest_rank(0.95),
    }


def _rate(group: list[dict], field: str) -> dict[str, float | int | None]:
    known = [row[field] for row in group if type(row.get(field)) is bool]
    return {"evaluated": len(known), "rate": sum(known) / len(known) if known else None}


def build_benchmark(
    records: Iterable[dict[str, Any]], *, lab: str,
    run_id: str, scenario: str | None = None,
) -> dict[str, Any]:
    """Summarize one lab/run; never pool distinct source files or workloads."""
    if lab not in LABS:
        raise ValueError(f"Unknown lab: {lab}")
    selected = [
        row for row in records if row.get("lab") == lab and row.get("run_id") == run_id
    ]
    if not selected:
        raise ValueError(f"No saved measurements for lab={lab!r}, run_id={run_id!r}")
    scenarios = sorted({str(row["scenario"]) for row in selected})
    if lab == "lab07" and scenario is None and len(scenarios) > 1:
        raise ValueError(
            "Lab07 contains multiple scenarios: specify --scenario as one of "
            + ", ".join(scenarios)
        )
    if scenario is not None:
        selected = [row for row in selected if row["scenario"] == scenario]
        if not selected:
            raise ValueError(f"No measurements for scenario={scenario!r} in the selected run")

    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in selected:
        groups[tuple(str(row.get(key) or "Unspecified") for key in GROUP_KEYS)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        summaries.append({
            **dict(zip(GROUP_KEYS, key)),
            "requests": len(group),
            "simulated": all(bool(row.get("simulated")) for row in group),
            "success": _rate(group, "succeeded"),
            "answer_accuracy": _rate(group, "answer_correct"),
            "metrics": {field: _metric_stats(_measured(group, field)) for field in NUMERIC_FIELDS},
        })

    return {
        "schema_version": 1,
        "lab": lab,
        "run_id": run_id,
        "scenario": scenario if scenario is not None else (scenarios[0] if len(scenarios) == 1 else None),
        "measurement_count": len(selected),
        "group_count": len(summaries),
        "groups": summaries,
    }


def _safe(value: Any) -> str:
    """Bound local identifiers and render them as inert Markdown."""
    raw = "".join(char if char.isprintable() else " " for char in str(value))[:100]
    return (raw.replace("\\", "\\\\").replace("|", "\\|")
            .replace("<", "&lt;").replace(">", "&gt;").replace(chr(96), "'"))


def _number(value: float | int | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def _mean_n(metric: dict, digits: int = 2) -> str:
    return f"{_number(metric['mean'], digits)} (n={metric['observed']})"


def _rate_n(rate: dict) -> str:
    return (f"{_number(100 * rate['rate'], 1)}% (n={rate['evaluated']})"
            if rate["rate"] is not None else "N/A (n=0)")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a stable Markdown report with denominators and no raw prompts."""
    lines = [
        "# LLM Token Economics — Benchmark Report",
        "",
        f"- Lab: **{_safe(report['lab'])}**",
        f"- Run: **{_safe(report['run_id'])}**",
        f"- Scenario: **{_safe(report['scenario']) if report['scenario'] is not None else 'mixed (separate groups)'}**",
        f"- Saved measurements: **{report['measurement_count']}**",
        f"- Groups: **{report['group_count']}**",
        "",
        "All metrics use recorded, valid values only. Each n is the number of rows "
        "where that particular metric exists, not necessarily the request count.",
        "P50/P95 use the nearest-rank method. Small samples are descriptive. "
        "Costs are estimated, not invoices. Workloads and file sources are not pooled.",
        "",
    ]
    if report["lab"] == "lab07":
        lines += [
            "## Offline retry simulation",
            "",
            "| Scenario | Strategy | Source | Clients | Success | Mean retries | Mean simulated 429 | Mean virtual wait (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for group in report["groups"]:
            m = group["metrics"]
            lines.append(
                f"| {_safe(group['scenario'])} | {_safe(group['condition'])} | {_safe(group['source'])}"
                f" | {group['requests']} | {_rate_n(group['success'])}"
                f" | {_mean_n(m['retry_count'])} | {_mean_n(m['rate_limit_response_count'])}"
                f" | {_mean_n(m['total_scheduled_wait_seconds'])} |"
            )
        lines.extend([
            "", "These are synthetic HTTP/client outcomes and virtual wait times. "
            "There is no real API usage, billable cost, model-answer accuracy or real latency."
        ])
    else:
        lines += [
            "## Recorded provider measurements",
            "",
            "| Experiment | Source | Scenario | Model | Condition | Cost basis | Requests | Success | Answer accuracy | Mean tokens/request | Mean estimated USD/request | Latency p50 / p95 (ms) |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for group in report["groups"]:
            m = group["metrics"]
            latency = m["latency_ms"]
            pcts = f"{_number(latency['p50'])} / {_number(latency['p95'])} (n={latency['observed']})"
            lines.append(
                f"| {_safe(group['experiment'])} | {_safe(group['source'])} | {_safe(group['scenario'])}"
                f" | {_safe(group['model'])} | {_safe(group['condition'])} | {_safe(group['cost_basis'])}"
                f" | {group['requests']} | {_rate_n(group['success'])} | {_rate_n(group['answer_accuracy'])}"
                f" | {_mean_n(m['total_tokens'])} | {_mean_n(m['estimated_cost_usd'], 7)} | {pcts} |"
            )
        lines.extend([
            "", "Latency semantics vary by experiment. These descriptive values do not "
            "establish causal or statistically significant differences."
        ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic, offline benchmark reports from saved measurements."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--lab", required=True, choices=LABS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario", help="Required for Lab07 runs with multiple scenarios")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--output", type=Path, help="Write report to path (otherwise stdout)")
    args = parser.parse_args(argv)
    records, warnings = scan_results(args.results_dir)
    try:
        report = build_benchmark(records, lab=args.lab, run_id=args.run_id, scenario=args.scenario)
    except ValueError as exc:
        parser.error(str(exc))
    if warnings:
        print(f"Warning: skipped {len(warnings)} invalid measurement file(s).", file=sys.stderr)
    content = (
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.format == "json" else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Saved report: {args.output}")
    else:
        sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
