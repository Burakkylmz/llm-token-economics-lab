"""Read-only, matched-context comparison of two saved experiment conditions.

No provider calls or raw prompts. Descriptive group differences are NOT paired
request deltas, causal effects, confidence intervals, or proof of a best model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from lab_core.benchmark_report import (
    GROUP_KEYS,
    LABS,
    NUMERIC_FIELDS,
    build_benchmark,
)
from lab_core.dashboard_data import scan_results
from lab_core.experiment_validation import validate_run

COMPARISON_REPORT_VERSION = 1
MATCH_KEYS = ("experiment", "source", "scenario", "cost_basis")
VALUE_UNITS = {
    "success_rate": "fraction",
    "answer_accuracy": "fraction",
    "total_tokens": "tokens / measured record",
    "estimated_cost_usd": "estimated USD / measured record",
    "latency_ms": "milliseconds / measured record",
    "latency_p50_ms": "milliseconds (nearest-rank p50)",
    "latency_p95_ms": "milliseconds (nearest-rank p95)",
    "retry_count": "simulated retries / client",
    "rate_limit_response_count": "simulated HTTP 429 responses / client",
    "total_scheduled_wait_seconds": "virtual seconds / client",
}


def _pick(report: dict[str, Any], condition: str, filters: dict[str, str | None]) -> dict[str, Any]:
    matches = [
        group for group in report["groups"]
        if group["condition"] == condition
        and all(value is None or group[key] == value for key, value in filters.items())
    ]
    if not matches:
        raise ValueError(
            f"No group for condition {condition!r}; inspect the saved run, scenario and optional filters"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous condition {condition!r}: {len(matches)} groups. "
            "Supply --experiment, --source, --model or --cost-basis to select exactly one"
        )
    return matches[0]


def _is_finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _difference(
    baseline: float | None,
    candidate: float | None,
    baseline_n: int,
    candidate_n: int,
    unit: str,
) -> dict[str, Any]:
    """Candidate minus baseline; undefined differences remain null."""
    present = (
        baseline_n > 0 and candidate_n > 0
        and _is_finite(baseline) and _is_finite(candidate)
    )
    delta = (candidate - baseline) if present else None
    relative = 100 * delta / abs(baseline) if present and baseline != 0 else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    return {
        "unit": unit,
        "baseline": baseline if present or _is_finite(baseline) else None,
        "candidate": candidate if present or _is_finite(candidate) else None,
        "baseline_observed": baseline_n,
        "candidate_observed": candidate_n,
        "absolute_delta": delta,
        "relative_delta_percent": relative,
        "percentage_point_delta": (delta * 100 if present and unit == "fraction" else None),
    }


def compare_benchmark(
    report: dict[str, Any],
    *,
    baseline_condition: str,
    candidate_condition: str,
    experiment: str | None = None,
    source: str | None = None,
    model: str | None = None,
    cost_basis: str | None = None,
    allow_model_difference: bool = False,
) -> dict[str, Any]:
    """Compare two unique groups from one lab/run and matched data context.

    Different models must be explicitly acknowledged. Even then, differences
    are descriptive: the tool cannot verify matched prompts or price dates.
    """
    if baseline_condition == candidate_condition:
        raise ValueError("Baseline and candidate conditions must be different")
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark report schema version")

    filters = {
        "experiment": experiment,
        "source": source,
        "model": model,
        "cost_basis": cost_basis,
    }
    baseline = _pick(report, baseline_condition, filters)
    candidate = _pick(report, candidate_condition, filters)

    for key in MATCH_KEYS:
        if baseline[key] != candidate[key]:
            raise ValueError(
                f"Incompatible comparison: {key} differs. "
                "Compare the same experiment, source, scenario and cost basis"
            )

    model_differs = baseline["model"] != candidate["model"]
    if model_differs and not allow_model_difference:
        raise ValueError(
            "Model identifiers differ; supply --allow-model-difference to "
            "acknowledge an exploratory cross-model comparison"
        )
    if bool(baseline["simulated"]) != bool(candidate["simulated"]):
        raise ValueError("Simulated and real provider measurements cannot be compared")
    if report["lab"] == "lab07" and not baseline["simulated"]:
        raise ValueError("Lab07 must use synthetic measurements only")
    if report["lab"] != "lab07" and baseline["simulated"]:
        raise ValueError("Real-model labs cannot use simulated Lab07 measurements")

    metrics: dict[str, dict[str, Any]] = {}
    for key in ("success_rate", "answer_accuracy"):
        name = "success" if key == "success_rate" else "answer_accuracy"
        left, right = baseline[name], candidate[name]
        metrics[key] = _difference(
            left["rate"], right["rate"], left["evaluated"], right["evaluated"], "fraction"
        )
    for key in NUMERIC_FIELDS:
        left, right = baseline["metrics"][key], candidate["metrics"][key]
        metrics[key] = _difference(
            left["mean"], right["mean"], left["observed"], right["observed"], VALUE_UNITS[key]
        )
    for key, percentile in (("latency_p50_ms", "p50"), ("latency_p95_ms", "p95")):
        left, right = baseline["metrics"]["latency_ms"], candidate["metrics"]["latency_ms"]
        metrics[key] = _difference(
            left[percentile], right[percentile], left["observed"], right["observed"], VALUE_UNITS[key]
        )

    notes = [
        "Descriptive unpaired group summaries; not matched-request effects or statistical significance.",
        "Each metric has its own observed denominator; missing data is not zero.",
        "Relative percent difference is undefined when the baseline value is zero.",
        "Recorded workload, prompts, latency semantics and pricing provenance must be verified independently.",
    ]
    if baseline["simulated"]:
        notes.append(
            "Lab07 values are synthetic client/retry outcomes and virtual time; "
            "no real LLM calls, provider token billing, real latency or answer quality are measured."
        )
    if model_differs:
        notes.append(
            "Models differ by explicit opt-in; results are exploratory and may reflect "
            "different capabilities, configurations or token pricing."
        )
    if baseline["requests"] != candidate["requests"]:
        notes.append("Unequal record counts; denominators and potential selection bias require review.")
    if baseline["model"] in ("Unknown", "Unspecified"):
        notes.append("Model provenance is unknown; check the original run configuration.")

    return {
        "comparison_report_version": COMPARISON_REPORT_VERSION,
        "lab": report["lab"],
        "run_id": report["run_id"],
        "context": {key: baseline[key] for key in MATCH_KEYS},
        "baseline": {"condition": baseline_condition, "model": baseline["model"],
                     "requests": baseline["requests"]},
        "candidate": {"condition": candidate_condition, "model": candidate["model"],
                      "requests": candidate["requests"]},
        "model_difference_acknowledged": bool(model_differs and allow_model_difference),
        "simulated": bool(baseline["simulated"]),
        "metrics": metrics,
        "notes": notes,
    }


def _escape(value: Any) -> str:
    raw = "".join(c if c.isprintable() else " " for c in str(value))[:100]
    return (raw.replace("\\", "\\\\").replace("|", "\\|")
            .replace("<", "&lt;").replace(">", "&gt;").replace(chr(96), "'"))


def _fmt(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:,.6g}"


def render_markdown(comparison: dict[str, Any]) -> str:
    """Render deterministic Markdown without untrusted raw prompt/answer data."""
    left, right = comparison["baseline"], comparison["candidate"]
    ctx = comparison["context"]
    lines = [
        "# LLM Token Economics — Condition Comparison",
        "",
        f"- Lab: **{_escape(comparison['lab'])}**; run: **{_escape(comparison['run_id'])}**",
        f"- Experiment: **{_escape(ctx['experiment'])}**; source: **{_escape(ctx['source'])}**",
        f"- Scenario: **{_escape(ctx['scenario'])}**; cost basis: **{_escape(ctx['cost_basis'])}**",
        f"- Baseline: **{_escape(left['condition'])}** ({_escape(left['model'])}, {left['requests']} records)",
        f"- Candidate: **{_escape(right['condition'])}** ({_escape(right['model'])}, {right['requests']} records)",
        "",
        "Absolute delta = candidate minus baseline. For rates, percentage points (pp) are shown.",
        "Relative delta % is undefined when the baseline is zero. N/A means a missing observation, **not** zero.",
        "",
        "| Metric | Unit | Baseline (n) | Candidate (n) | Delta | Relative % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in comparison["metrics"].items():
        change = metric["percentage_point_delta"] if metric["unit"] == "fraction" else metric["absolute_delta"]
        suffix = " pp" if metric["unit"] == "fraction" and change is not None else ""
        lines.append(
            f"| {_escape(name)} | {_escape(metric['unit'])}"
            f" | {_fmt(metric['baseline'])} (n={metric['baseline_observed']})"
            f" | {_fmt(metric['candidate'])} (n={metric['candidate_observed']})"
            f" | {_fmt(change)}{suffix} | {_fmt(metric['relative_delta_percent'])} |"
        )
    lines += ["", "## Interpretation constraints", ""]
    lines += [f"- {_escape(note)}" for note in comparison["notes"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two conditions in one saved run, offline and without provider calls."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--lab", required=True, choices=LABS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario", help="Required when Lab07 run contains multiple scenarios")
    parser.add_argument("--baseline-condition", required=True)
    parser.add_argument("--candidate-condition", required=True)
    parser.add_argument("--experiment", help="Restrict to one experiment family")
    parser.add_argument("--source", help="Restrict to one measurement filename")
    parser.add_argument("--model", help="Restrict both groups to the same model")
    parser.add_argument("--cost-basis", help="Restrict both groups to one cost basis")
    parser.add_argument("--allow-model-difference", action="store_true")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--output", type=Path, help="Write report (outside selected run); else stdout")
    args = parser.parse_args(argv)

    rows, warnings = scan_results(args.results_dir)
    try:
        report = build_benchmark(rows, lab=args.lab, run_id=args.run_id, scenario=args.scenario)
        comparison = compare_benchmark(
            report, baseline_condition=args.baseline_condition,
            candidate_condition=args.candidate_condition, experiment=args.experiment,
            source=args.source, model=args.model, cost_basis=args.cost_basis,
            allow_model_difference=args.allow_model_difference,
        )
        # The comparison is restricted to modern saved runs to ensure that
        # the opt-in validation gate applies to its exact source directory.
        experiment = comparison["context"]["experiment"]
        if experiment not in ("legacy",) and experiment:
            validation = validate_run(args.results_dir, experiment=experiment, run_id=args.run_id)
            if not validation.to_report()["valid"]:
                raise ValueError("Selected experiment data failed validation; repair the saved run before comparing")
        else:
            raise ValueError("Legacy flat files cannot be compared without a run-specific manifest")
    except ValueError as exc:
        parser.error(str(exc))

    if warnings:
        print(f"Warning: {len(warnings)} invalid measurement file(s) were skipped by the loader.", file=sys.stderr)
    rendered = (
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.format == "json" else render_markdown(comparison)
    )
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = args.output.resolve()
        input_run = (args.results_dir / comparison["context"]["experiment"] / args.run_id).resolve()
        if output == input_run or output.is_relative_to(input_run):
            parser.error("Output cannot overwrite any file in the selected saved run")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Saved comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
