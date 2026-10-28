"""Markdown analysis report for one Lab04 experiment run.

Mirrors labs.lab02_context_growth.report/labs.lab03_prompt_caching.report's
structure and "restate only what the data shows" rule, with sections
specific to Lab04's research questions (output budget, truncation,
reasoning overhead, answer quality, invocation count, workflow latency,
quality-constrained cost) and an explicit, required limitations section.
"""

from __future__ import annotations

from typing import Optional, Sequence

from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement
from labs.lab04_output_control.stats import InvocationComparisonSummary, OutputLimitConditionSummary

MIN_REPETITIONS_FOR_ROBUST_CLAIM = 10


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _stats_table_header() -> list[str]:
    return [
        "| Metric | n | missing | mean | median (p50) | min | max | range | sample stdev | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]


def _stats_row(label, stats) -> str:
    return (
        f"| {label} | {stats.count} | {stats.count_missing} | {_fmt(stats.mean)} | {_fmt(stats.median)} | "
        f"{_fmt(stats.minimum)} | {_fmt(stats.maximum)} | {_fmt(stats.range)} | "
        f"{_fmt(stats.sample_standard_deviation)} | {_fmt(stats.p95)} |"
    )


def build_analysis_report_markdown(
    manifest: dict,
    output_limit_measurements: Sequence[OutputLimitMeasurement],
    invocation_measurements: Sequence[InvocationMeasurement],
    output_limit_summaries: Sequence[OutputLimitConditionSummary],
    invocation_comparison: Optional[InvocationComparisonSummary],
) -> str:
    config = manifest.get("experiment_configuration", {})
    ol_successes = [m for m in output_limit_measurements if m.succeeded]
    ol_failures = [m for m in output_limit_measurements if not m.succeeded]
    inv_successes = [m for m in invocation_measurements if m.succeeded]
    inv_failures = [m for m in invocation_measurements if not m.succeeded]

    lines: list[str] = []
    lines.append(f"# Lab04 Output Control Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Model: `{manifest.get('model')}`")
    lines.append(f"- reasoning_effort: `{config.get('reasoning_effort')}`")
    lines.append(f"- output_limit_values: {config.get('output_limit_values')}")
    lines.append(f"- output_limit_repetitions: {config.get('output_limit_repetitions')}")
    lines.append(f"- invocation_max_completion_tokens: {config.get('invocation_max_completion_tokens')}")
    lines.append(f"- invocation_execution_mode: `{config.get('invocation_execution_mode')}`")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append("")

    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Output-limit calls: {len(output_limit_measurements)} (successful: {len(ol_successes)}, failed: {len(ol_failures)})")
    lines.append(f"- Invocation-comparison calls: {len(invocation_measurements)} (successful: {len(inv_successes)}, failed: {len(inv_failures)})")
    lines.append("")

    # 3. Output control / truncation (RQ1-RQ2)
    lines.append("## 3. Output Limit & Truncation Detection (RQ1-RQ2)")
    lines.append("")
    lines.append("| condition_id | max_completion_tokens | successful | truncated | truncation rate |")
    lines.append("|---|---|---|---|---|")
    for s in output_limit_summaries:
        lines.append(
            f"| {s.condition_id} | {s.max_completion_tokens} | {s.successful_calls} | {s.truncated_calls} | "
            f"{_fmt_pct(s.truncation_rate)} |"
        )
    lines.append("")
    lines.append(
        "Truncation is detected strictly from the provider's own `finish_reason == \"length\"` -- "
        "`completion_tokens == max_completion_tokens` alone is necessary but not sufficient evidence."
    )
    lines.append("")
    lines.append("**Completion / Reasoning / Non-Reasoning Tokens by Condition**")
    lines.append("")
    lines += _stats_table_header()
    for s in output_limit_summaries:
        lines.append(_stats_row(f"{s.condition_id}.completion_tokens", s.completion_tokens))
        lines.append(_stats_row(f"{s.condition_id}.reasoning_tokens", s.reasoning_tokens))
        lines.append(_stats_row(f"{s.condition_id}.non_reasoning_completion_tokens", s.non_reasoning_completion_tokens))
    lines.append("")

    # 4. Answer quality (RQ3-RQ4)
    lines.append("## 4. Reasoning Overhead & Answer Quality (RQ3-RQ4)")
    lines.append("")
    lines.append("| condition_id | accepted | incomplete | missing_response | mean completeness | mean correctness |")
    lines.append("|---|---|---|---|---|---|")
    for s in output_limit_summaries:
        lines.append(
            f"| {s.condition_id} | {s.accepted_count} | {s.incomplete_count} | {s.missing_response_count} | "
            f"{_fmt_pct(s.mean_field_completeness_ratio)} | {_fmt_pct(s.mean_field_correctness_ratio)} |"
        )
    lines.append("")
    lines.append(
        "**The lowest output token count is not necessarily the best optimization result.** A condition "
        "that consumes fewer tokens but accepts fewer answers is not a successful optimization -- cost/"
        "token savings and answer acceptance are always reported together, never one without the other."
    )
    lines.append("")

    # 5. Invocation comparison (RQ5)
    lines.append("## 5. Invocation Comparison (RQ5)")
    lines.append("")
    if invocation_comparison is not None:
        single, multiple = invocation_comparison.single, invocation_comparison.multiple
        lines.append("| | Single Invocation | Multiple Invocations |")
        lines.append("|---|---|---|")
        lines.append(f"| API Calls | {single.api_call_count} | {multiple.api_call_count} |")
        lines.append(f"| Total Prompt Tokens | {single.total_prompt_tokens} | {multiple.total_prompt_tokens} |")
        lines.append(f"| Total Completion Tokens | {single.total_completion_tokens} | {multiple.total_completion_tokens} |")
        lines.append(f"| Total Reasoning Tokens | {single.total_reasoning_tokens} | {multiple.total_reasoning_tokens} |")
        lines.append(f"| Total Estimated Cost | {_fmt_cost(single.total_estimated_cost_usd)} | {_fmt_cost(multiple.total_estimated_cost_usd)} |")
        lines.append(f"| Correct Answers | {single.correct_answers}/{single.api_call_count} | {multiple.correct_answers}/{multiple.api_call_count} |")
        lines.append("")
        lines.append(
            f"- Absolute cost difference (multiple - single): {_fmt_cost(invocation_comparison.absolute_cost_difference_usd)}"
        )
        lines.append(
            f"- Relative cost difference: {_fmt(invocation_comparison.relative_cost_difference_percent)}%"
        )
    else:
        lines.append("Invocation comparison was not run in this experiment.")
    lines.append("")

    # 6. Workflow latency (RQ6)
    lines.append("## 6. Workflow Wall-Clock Latency (RQ6)")
    lines.append("")
    if invocation_comparison is not None:
        single, multiple = invocation_comparison.single, invocation_comparison.multiple
        lines.append(f"- Single invocation workflow wall-clock: {_fmt(single.workflow_wall_clock_ms)} ms")
        lines.append(f"- Multiple invocations workflow wall-clock: {_fmt(multiple.workflow_wall_clock_ms)} ms")
        lines.append(f"- Difference (multiple - single): {_fmt(invocation_comparison.workflow_latency_difference_ms)} ms")
        lines.append(
            f"- Single invocation per-request latencies (ms): {single.individual_request_latencies_ms}"
        )
        lines.append(
            f"- Multiple invocations per-request latencies (ms): {multiple.individual_request_latencies_ms}"
        )
        lines.append(
            "Per-request latency totals and workflow wall-clock are NOT asserted equal -- loop overhead "
            "between calls is real and is never hidden by summing per-request numbers instead."
        )
    else:
        lines.append("Invocation comparison was not run in this experiment.")
    lines.append("")

    # 7. Quality-constrained comparison (RQ7)
    lines.append("## 7. Quality-Constrained Comparison (RQ7)")
    lines.append("")
    if invocation_comparison is not None:
        lines.append(f"- `quality_comparison`: **{invocation_comparison.quality_comparison}**")
        lines.append(
            "A strategy is never declared optimal merely for being cheaper or faster if its answers "
            "were not all accepted."
        )
    else:
        lines.append("Invocation comparison was not run in this experiment.")
    lines.append("")

    # 8. Limitations (explicitly required)
    lines.append("## 8. Experimental Limitations")
    lines.append("")
    repetitions = config.get("output_limit_repetitions")
    lines.append(
        f"- Each output-limit condition was repeated {repetitions} time(s). Below "
        f"{MIN_REPETITIONS_FOR_ROBUST_CLAIM} repetitions, these statistics describe this one run only -- "
        "no statistical significance or confidence interval is claimed."
    )
    lines.append("- **n=1 standard deviation is never fabricated**: single-repetition groups report `sample_standard_deviation = N/A`.")
    lines.append("- The invocation comparison runs the single strategy exactly once and the multiple strategy exactly once per question -- no repetition-based variability statistics are computed for it.")
    lines.append("- A single accepted/failed observation is never generalized into a production-reliability claim.")
    lines.append("- Workflow wall-clock latency includes this process's own loop/orchestration overhead, not only network/provider time.")
    lines.append("- Output length (and therefore cost/latency) is not held perfectly fixed across repetitions even within one condition -- the model's own response length varies.")
    lines.append("")

    return "\n".join(lines)
