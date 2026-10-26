"""Markdown analysis report for one `effort` experiment run -- the
single-factor analogue of labs.lab01_token_anatomy.factorial_report, with
the temperature/top_p main-effect and interaction sections removed (there
is only one factor to compare now: reasoning_effort).
"""

from __future__ import annotations

from typing import Sequence

from labs.lab01_token_anatomy.effort_models import EffortCondition, EffortMeasurement
from labs.lab01_token_anatomy.effort_stats import (
    DescriptiveStats,
    EffortConditionSummary,
    compute_grouped_effect,
)

MIN_REPETITIONS_FOR_ROBUST_CLAIM = 10


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value) -> str:
    return f"${value:.6f}" if value is not None else "N/A"


def _stats_row(label, stats: DescriptiveStats) -> str:
    return (
        f"| {label} | {stats.count} | {stats.count_missing} | {_fmt(stats.mean)} | {_fmt(stats.median)} | "
        f"{_fmt(stats.minimum)} | {_fmt(stats.maximum)} | {_fmt(stats.range)} | "
        f"{_fmt(stats.sample_standard_deviation)} | {_fmt(stats.p95)} |"
    )


def _stats_table_header() -> list[str]:
    return [
        "| Metric | n | missing | mean | median (p50) | min | max | range | sample stdev | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]


def build_analysis_report_markdown(
    manifest: dict,
    conditions: Sequence[EffortCondition],
    measurements: Sequence[EffortMeasurement],
    summaries: Sequence[EffortConditionSummary],
) -> str:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.finish_reason == "length"]
    prompt_types = sorted({c.prompt_type for c in conditions})
    repetitions = manifest.get("experiment_configuration", {}).get("repetitions")

    lines: list[str] = []
    lines.append(f"# Lab01 Reasoning Effort Comparison Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    # 1. Experiment configuration
    lines.append("## 1. Experiment Configuration")
    lines.append("")
    config = manifest.get("experiment_configuration", {})
    lines.append(f"- Model: `{manifest.get('model')}`")
    lines.append(f"- Prompts: {config.get('prompts')}")
    lines.append(f"- Reasoning efforts: {config.get('reasoning_efforts')}")
    lines.append(f"- Repetitions per condition: {repetitions}")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
    lines.append("- temperature/top_p: not sent -- provider default (not explicitly set, not known)")
    lines.append(f"- Planned conditions: {manifest.get('planned_conditions')}")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append(f"- configuration_fingerprint: `{manifest.get('configuration_fingerprint')}`")
    lines.append("")

    # 2. Total requests and successful measurements
    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Total measurements recorded: {len(measurements)}")
    lines.append(f"- Successful: {len(successes)}")
    lines.append(f"- Failed: {len(failures)}")
    if failures:
        lines.append("- Failure reasons observed (not necessarily exhaustive):")
        for err in sorted({f.error for f in failures if f.error})[:10]:
            lines.append(f"  - {err}")
    lines.append("")

    if not successes:
        lines.append(
            "_No successful measurements are available -- the remaining sections below "
            "have nothing to report on and are intentionally omitted rather than filled "
            "with placeholder numbers._"
        )
        return "\n".join(lines)

    # 3. Reasoning effort comparison (the only factor in this experiment)
    lines.append("## 3. Reasoning Effort Comparison")
    lines.append("")
    lines.append(
        "One table per prompt type (short_prompt and long_prompt differ in task scope as "
        "well as length, so their results are never pooled into one number)."
    )
    lines.append("")
    for prompt_type in prompt_types:
        subset = [m for m in successes if m.prompt_type == prompt_type]
        lines.append(f"### {prompt_type}")
        lines.append("")
        lines.append("**Reasoning Tokens by Reasoning Effort**")
        lines.append("")
        lines += _stats_table_header()
        for level, stats in compute_grouped_effect(subset, ("reasoning_effort",), "reasoning_tokens").items():
            lines.append(_stats_row(str(level), stats))
        lines.append("")
        lines.append("**Completion Tokens by Reasoning Effort**")
        lines.append("")
        lines += _stats_table_header()
        for level, stats in compute_grouped_effect(subset, ("reasoning_effort",), "completion_tokens").items():
            lines.append(_stats_row(str(level), stats))
        lines.append("")
        lines.append("**Non-Reasoning Completion Tokens by Reasoning Effort**")
        lines.append("")
        lines += _stats_table_header()
        for level, stats in compute_grouped_effect(
            subset, ("reasoning_effort",), "non_reasoning_completion_tokens"
        ).items():
            lines.append(_stats_row(str(level), stats))
        lines.append("")
        lines.append("**Latency (ms) by Reasoning Effort**")
        lines.append("")
        lines += _stats_table_header()
        for level, stats in compute_grouped_effect(subset, ("reasoning_effort",), "total_latency_ms").items():
            lines.append(_stats_row(str(level), stats))
        lines.append("")
        lines.append("**Estimated Cost (USD) by Reasoning Effort**")
        lines.append("")
        lines += _stats_table_header()
        for level, stats in compute_grouped_effect(subset, ("reasoning_effort",), "estimated_cost_usd").items():
            lines.append(_stats_row(str(level), stats))
        lines.append("")

    # 4. Token variability (per-condition)
    lines.append("## 4. Token Variability by Condition")
    lines.append("")
    lines += _stats_table_header()
    for summary in summaries:
        lines.append(_stats_row(summary.condition_id, summary.completion_tokens))
    lines.append("")
    lines.append(
        "Full per-condition detail (reasoning_tokens, non_reasoning_completion_tokens, "
        "latency, cost, reasoning ratios) is in `condition_summary.csv` next to this report."
    )
    lines.append("")

    # 5. Cost and latency comparisons
    lines.append("## 5. Cost and Latency Comparisons")
    lines.append("")
    total_cost = sum(s.total_estimated_cost_usd or 0 for s in summaries)
    lines.append(f"- Total estimated cost across all successful calls: {_fmt_cost(total_cost)}")
    lines.append("")
    lines.append("| condition_id | successful_calls | mean latency (ms) | total cost | mean cost |")
    lines.append("|---|---|---|---|---|")
    for summary in summaries:
        lines.append(
            f"| {summary.condition_id} | {summary.successful_calls} | "
            f"{_fmt(summary.latency_ms.mean)} | {_fmt_cost(summary.total_estimated_cost_usd)} | "
            f"{_fmt_cost(summary.mean_estimated_cost_usd)} |"
        )
    lines.append("")

    # 6. Truncation and missing data
    lines.append("## 6. Truncation and Missing Data")
    lines.append("")
    lines.append(f"- Calls ending with `finish_reason=\"length\"` (possible truncation): {len(truncated)}/{len(successes)}")
    reasoning_missing = sum(1 for m in successes if m.reasoning_tokens is None)
    lines.append(f"- Calls with no reasoning_tokens reported at all: {reasoning_missing}/{len(successes)}")
    lines.append("")

    # 7. Statistical limitations
    lines.append("## 7. Statistical Limitations")
    lines.append("")
    lines.append(
        f"- Each condition was repeated {repetitions} time(s). Below "
        f"{MIN_REPETITIONS_FOR_ROBUST_CLAIM} repetitions, these statistics are descriptive "
        "of this one run, not proof of the model's general behavior."
    )
    if repetitions == 1:
        lines.append(
            "- With exactly 1 repetition per condition, sample standard deviation and p95 are "
            "undefined (`N/A`) for every condition -- this is expected, not a bug."
        )
    lines.append(
        "- Sample standard deviation uses `statistics.stdev` (n-1 denominator) and is `N/A` "
        "below 2 observations; percentiles use `statistics.quantiles(..., method=\"inclusive\")` "
        "and are also `N/A` below 2 observations."
    )
    lines.append(
        "- short_prompt and long_prompt differ in task scope as well as length -- differences "
        "between them are never attributed to prompt length alone."
    )
    lines.append(
        "- `temperature`/`top_p` are not sent, so provider defaults apply -- this does NOT mean "
        "they have zero effect on the model's behavior, only that this experiment does not "
        "vary or observe them."
    )
    lines.append("")

    # 8. Research conclusions
    lines.append("## 8. Research Conclusions")
    lines.append("")
    lines.append(
        "This section restates only what the data above actually shows, re-derived from "
        "`measurements` -- it does not add claims beyond that."
    )
    for prompt_type in prompt_types:
        subset = [m for m in successes if m.prompt_type == prompt_type]
        by_effort = compute_grouped_effect(subset, ("reasoning_effort",), "reasoning_tokens")
        ordered = sorted((level, s.mean) for level, s in by_effort.items() if s.mean is not None)
        if ordered:
            summary_text = ", ".join(f"{level}={mean:.1f}" for level, mean in ordered)
            lines.append(f"- {prompt_type}: observed mean reasoning_tokens by reasoning_effort -- {summary_text}.")
    lines.append(
        "- Whether more reasoning tokens means a better final answer was evaluated only via a "
        "weak substring heuristic for short_prompt (`quality_heuristic_passed`, NOT a "
        "correctness guarantee); long_prompt has no evaluation rubric in this project and is "
        "always `quality_evaluation_method=\"not_evaluated\"`."
    )
    lines.append("")

    return "\n".join(lines)
