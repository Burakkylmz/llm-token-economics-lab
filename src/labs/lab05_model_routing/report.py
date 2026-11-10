"""Markdown analysis report for one Lab05 experiment run.

Mirrors every other redesigned lab's report module structure and "restate
only what the data shows" rule, with sections specific to Lab05's research
questions (routing decisions, quality regression, cost efficiency,
latency) and an explicit, required limitations section.
"""

from __future__ import annotations

from typing import Optional, Sequence

from labs.lab05_model_routing.models import RoutingMeasurement
from labs.lab05_model_routing.stats import CostComparison, StrategySummary, TaskComparison

MIN_TASKS_FOR_ROBUST_CLAIM = 30


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value:.1f}%" if value is not None else "N/A"


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def build_analysis_report_markdown(
    manifest: dict,
    measurements: Sequence[RoutingMeasurement],
    summaries: Sequence[StrategySummary],
    comparisons: Sequence[TaskComparison],
    cost_comparison: Optional[CostComparison],
) -> str:
    config = manifest.get("experiment_configuration", {})
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]

    lines: list[str] = []
    lines.append(f"# Lab05 Model Routing Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Profile: `{manifest.get('profile')}`")
    lines.append(f"- Small model: `{config.get('small_model')}`")
    lines.append(f"- Large model: `{config.get('large_model')}`")
    lines.append(f"- reasoning_effort: `{config.get('reasoning_effort')}`")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append("")
    prices = manifest.get("pricing_snapshot") or {}
    if prices:
        lines.append("### Pricing snapshot (USD per 1 million tokens)")
        lines.append("")
        lines.append("| model | input | cached input | output | price verified |")
        lines.append("|---|---:|---:|---:|---|")
        for model, price in sorted(prices.items()):
            if price is None:
                lines.append(f"| `{model}` | N/A | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| `{model}` | {price['input_per_1m_usd']} | "
                f"{price['cached_input_per_1m_usd']} | {price['output_per_1m_usd']} | "
                f"{price['price_verified_date']} |"
            )
        lines.append("")
        lines.append(
            "These are configuration-time estimates, not provider invoices. "
            "Prices and availability may have changed since the verification date."
        )
        lines.append("")

    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Measurements recorded: {len(measurements)}")
    lines.append(f"- Successful: {len(successes)} / Failed: {len(failures)}")
    lines.append("")

    lines.append("## 3. Strategy Summary (RQ2-RQ4)")
    lines.append("")
    lines.append("| strategy | total | correct | accuracy | small calls | large calls | total cost | mean latency |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.strategy} | {s.total_tasks} | {s.correct_tasks} | "
            f"{_fmt_pct(s.accuracy * 100) if s.accuracy is not None else 'N/A'} (n={s.total_tasks}) | "
            f"{s.small_model_calls} | {s.large_model_calls} | {_fmt_cost(s.total_estimated_cost_usd)} | "
            f"{_fmt(s.latency_ms.mean)} ms |"
        )
    lines.append("")
    lines.append(
        "**Accuracy is always shown alongside dataset size (n)** -- 100% on a handful of tasks is a "
        "descriptive observation about THIS run, not a generalizable production-reliability claim."
    )
    lines.append("")

    lines.append("## 4. Routing Decisions (RQ1)")
    lines.append("")
    router_summary = next((s for s in summaries if s.strategy == "rule_based_routing"), None)
    if router_summary is not None and router_summary.routing_rule_distribution:
        lines.append("| matched rule | tasks routed to large model |")
        lines.append("|---|---|")
        for rule, count in sorted(router_summary.routing_rule_distribution.items()):
            lines.append(f"| {rule} | {count} |")
    else:
        lines.append("No large-model routing rule matched any task in this run (every task stayed on the small model).")
    lines.append("")

    lines.append("## 5. Task-Level Quality Regression Analysis (RQ2, RQ6)")
    lines.append("")
    if comparisons:
        lines.append("| task_id | baseline model | routed model | baseline correct | routed correct | quality outcome | cost delta | latency delta |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c.task_id} | {c.baseline_model} | {c.routed_model} | {c.baseline_correct} | "
                f"{c.routed_correct} | **{c.quality_outcome}** | {_fmt_cost(c.cost_delta)} | {_fmt(c.latency_delta)} ms |"
            )
    else:
        lines.append("No task-level baseline-vs-routed comparison is available for this run's strategy set.")
    lines.append("")
    lines.append(
        "A routed answer being wrong while the baseline was right is reported as "
        "`ROUTING_QUALITY_REGRESSION` regardless of any cost or latency savings -- quality regressions "
        "are never hidden behind a favorable cost number."
    )
    lines.append("")

    lines.append("## 6. Cost Efficiency Analysis (RQ3, RQ7)")
    lines.append("")
    if cost_comparison is not None:
        lines.append(f"- Baseline total cost: {_fmt_cost(cost_comparison.baseline_total_cost_usd)}")
        lines.append(f"- Routing total cost: {_fmt_cost(cost_comparison.routing_total_cost_usd)}")
        lines.append(f"- Absolute cost delta (routing - baseline): {_fmt_cost(cost_comparison.absolute_cost_delta_usd)}")
        lines.append(f"- Relative cost delta: {_fmt(cost_comparison.relative_cost_delta_percent)}%")
        if cost_comparison.routing_cost_savings_percent is not None:
            if cost_comparison.routing_cost_savings_percent < 0:
                lines.append(
                    f"- **Routing Cost Increase: {_fmt(-cost_comparison.routing_cost_savings_percent)}%** "
                    "(routing was more expensive than the always-small baseline in this run)."
                )
            else:
                lines.append(f"- Routing Cost Savings: {_fmt(cost_comparison.routing_cost_savings_percent)}%")
    else:
        lines.append("No baseline-vs-routing cost comparison is available for this run's strategy set.")
    lines.append("")
    lines.append(
        "A negative savings percentage (routing costing MORE) is reported exactly as measured -- it is "
        "never hidden, clamped to zero, or reframed as a success."
    )
    lines.append("")

    lines.append("## 7. Experimental Limitations")
    lines.append("")
    lines.append(f"- This run evaluated {len(set(m.task_id for m in measurements))} distinct task(s). Below {MIN_TASKS_FOR_ROBUST_CLAIM} tasks, these statistics describe this one run only -- no statistical significance or confidence interval is claimed.")
    lines.append("- **A single successful/failed observation per task is not proof that a model is always necessary or always unnecessary** for that task category.")
    lines.append("- Paired/interleaved execution order reduces, but does not eliminate, time-dependent confounders (e.g. provider load drifting over the run).")
    lines.append("- `structural_validation_only` (code-generation) results are never counted as `answer_correct=True` -- a matching function signature is not verified functional correctness.")
    lines.append("- Latency differences are reported as observations, not asserted as causally attributable to model size alone.")
    lines.append("- The routing policy is a fixed, deterministic rule set, not a learned or adaptive model -- it may route some tasks sub-optimally in ways this dataset does not expose.")
    lines.append("")

    return "\n".join(lines)
