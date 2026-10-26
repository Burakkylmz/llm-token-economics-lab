"""Builds the Markdown analysis report for one factorial experiment run.

Pure function of (manifest, conditions, measurements) -- no I/O, so it is
directly unit-testable with synthetic/mock measurements. Every number in
the report comes from `measurements`; nothing here invents a plausible-
looking figure when data is missing (missing shows up as "N/A" or an
explicit "insufficient data" note, never a fabricated value).
"""

from __future__ import annotations

from typing import Sequence

from labs.lab01_token_anatomy.factorial_models import FactorialCondition, FactorialMeasurement
from labs.lab01_token_anatomy.factorial_stats import (
    ConditionSummary,
    DescriptiveStats,
    compute_grouped_effect,
    compute_level_difference,
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


def _main_effect_section(title: str, measurements: Sequence[FactorialMeasurement], factor: str, metric: str) -> list[str]:
    lines = [f"**{title}**", ""]
    lines += _stats_table_header()
    grouped = compute_grouped_effect(measurements, (factor,), metric)
    for level, stats in grouped.items():
        lines.append(_stats_row(str(level), stats))
    lines.append("")
    lines.append(
        "_Averaged over every other factor's levels for this prompt type -- a main "
        "effect, not a controlled, single-variable-at-a-time comparison._"
    )
    lines.append("")
    return lines


def _interaction_table(measurements: Sequence[FactorialMeasurement], factor_a: str, factor_b: str, metric: str) -> list[str]:
    grouped = compute_grouped_effect(measurements, (factor_a, factor_b), metric)
    lines = [f"| {factor_a} | {factor_b} | n | mean {metric} | sample stdev |", "|---|---|---|---|---|"]
    for key, stats in grouped.items():
        level_a, level_b = key
        lines.append(f"| {level_a} | {level_b} | {stats.count} | {_fmt(stats.mean)} | {_fmt(stats.sample_standard_deviation)} |")
    return lines


def build_analysis_report_markdown(
    manifest: dict,
    conditions: Sequence[FactorialCondition],
    measurements: Sequence[FactorialMeasurement],
    summaries: Sequence[ConditionSummary],
) -> str:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.finish_reason == "length"]
    prompt_types = sorted({c.prompt_type for c in conditions})
    temperatures = sorted({c.temperature for c in conditions})

    lines: list[str] = []
    lines.append(f"# Lab01 Factorial Analysis Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    # 1. Experiment configuration
    lines.append("## 1. Experiment Configuration")
    lines.append("")
    config = manifest.get("experiment_configuration", {})
    lines.append(f"- Model: `{manifest.get('model')}`")
    lines.append(f"- Prompts: {config.get('prompts')}")
    lines.append(f"- Reasoning efforts: {config.get('reasoning_efforts')}")
    lines.append(f"- Temperatures: {config.get('temperatures')}")
    lines.append(f"- Top-p values: {config.get('top_p_values')}")
    lines.append(f"- Repetitions per condition: {config.get('repetitions')}")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
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

    # 3-5. Reasoning effort / temperature / top-p comparisons, per prompt type
    lines.append("## 3-5. Reasoning Effort, Temperature, and Top-p Comparisons")
    lines.append("")
    lines.append(
        "Each table below is a **main effect**: the mean for one factor's levels, "
        "averaged across every other factor's levels, computed separately per prompt "
        "type (short_prompt and long_prompt differ in task scope, not only length -- "
        "see Section 10 -- so their results are never pooled into one number)."
    )
    lines.append("")
    for prompt_type in prompt_types:
        subset = [m for m in successes if m.prompt_type == prompt_type]
        lines.append(f"### {prompt_type}")
        lines.append("")
        lines += _main_effect_section("Reasoning Effort -> reasoning_tokens", subset, "reasoning_effort", "reasoning_tokens")
        lines += _main_effect_section("Temperature -> reasoning_tokens", subset, "temperature", "reasoning_tokens")
        lines += _main_effect_section("Top-p -> reasoning_tokens", subset, "top_p", "reasoning_tokens")
        lines += _main_effect_section("Reasoning Effort -> completion_tokens", subset, "reasoning_effort", "completion_tokens")

    # 6. Interaction effects
    lines.append("## 6. Interaction Effects")
    lines.append("")
    lines.append(
        "Two-way tables (marginalized over the remaining factors), per prompt type. "
        "Cell counts below 2 observations make sample stdev undefined (`N/A`) -- not 0."
    )
    lines.append("")
    for prompt_type in prompt_types:
        subset = [m for m in successes if m.prompt_type == prompt_type]
        lines.append(f"### {prompt_type}: reasoning_effort x temperature -> reasoning_tokens")
        lines.append("")
        lines += _interaction_table(subset, "reasoning_effort", "temperature", "reasoning_tokens")
        lines.append("")
        lines.append(f"### {prompt_type}: reasoning_effort x top_p -> reasoning_tokens")
        lines.append("")
        lines += _interaction_table(subset, "reasoning_effort", "top_p", "reasoning_tokens")
        lines.append("")
        lines.append(f"### {prompt_type}: temperature x top_p -> reasoning_tokens")
        lines.append("")
        lines += _interaction_table(subset, "temperature", "top_p", "reasoning_tokens")
        lines.append("")

        if len(temperatures) >= 2:
            lo, hi = min(temperatures), max(temperatures)
            lines.append(
                f"**Controlled contrast (held constant: prompt_type={prompt_type}, top_p, "
                f"reasoning_effort): mean reasoning_tokens difference from temperature={lo} "
                f"to temperature={hi}, one row per (top_p, reasoning_effort) cell:**"
            )
            lines.append("")
            lines.append("| top_p | reasoning_effort | diff (temp hi - temp lo) |")
            lines.append("|---|---|---|")
            top_ps = sorted({c.top_p for c in conditions if c.prompt_type == prompt_type})
            efforts = sorted({c.reasoning_effort for c in conditions if c.prompt_type == prompt_type})
            for top_p in top_ps:
                for effort in efforts:
                    cell = [m for m in subset if m.top_p == top_p and m.reasoning_effort == effort]
                    by_temp = compute_grouped_effect(cell, ("temperature",), "reasoning_tokens")
                    diff = compute_level_difference(by_temp, lo, hi)
                    lines.append(f"| {top_p} | {effort} | {_fmt(diff)} |")
            lines.append("")

    lines.append(
        "_These are descriptive comparisons of observed means, not a fitted "
        "statistical model -- see Section 10 for why no p-values are reported._"
    )
    lines.append("")

    # 7. Token variability (per-condition)
    lines.append("## 7. Token Variability by Condition")
    lines.append("")
    lines += _stats_table_header()
    for summary in summaries:
        lines.append(_stats_row(summary.condition_id, summary.completion_tokens))
    lines.append("")
    lines.append(
        f"Full per-condition detail (reasoning_tokens, non_reasoning_completion_tokens, "
        f"latency, cost, reasoning ratios) is in `condition_summary.csv` next to this report."
    )
    lines.append("")

    # 8. Cost and latency comparisons
    lines.append("## 8. Cost and Latency Comparisons")
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

    # 9. Truncation and missing data
    lines.append("## 9. Truncation and Missing Data")
    lines.append("")
    lines.append(f"- Calls ending with `finish_reason=\"length\"` (possible truncation): {len(truncated)}/{len(successes)}")
    reasoning_missing = sum(1 for m in successes if m.reasoning_tokens is None)
    lines.append(f"- Calls with no reasoning_tokens reported at all: {reasoning_missing}/{len(successes)}")
    lines.append(
        "- `completion_tokens` equal to `max_completion_tokens` is reported together with "
        "`finish_reason` everywhere in this project -- it is never treated as truncation "
        "proof on its own."
    )
    lines.append("")

    # 10. Statistical limitations
    lines.append("## 10. Statistical Limitations")
    lines.append("")
    lines.append(
        f"- Each condition was repeated {manifest.get('experiment_configuration', {}).get('repetitions')} "
        f"time(s). Below {MIN_REPETITIONS_FOR_ROBUST_CLAIM} repetitions, per-condition statistics here "
        "are descriptive of this one run, not proof of the model's general behavior."
    )
    lines.append(
        "- Sample standard deviation uses `statistics.stdev` (n-1 denominator) and is `N/A` "
        "below 2 observations; percentiles use `statistics.quantiles(..., method=\"inclusive\")` "
        "and are also `N/A` below 2 observations."
    )
    lines.append(
        "- No ANOVA or regression p-values are reported: with 10 repetitions per cell and "
        "54 cells, fitting a factorial model would need its residual distribution and "
        "variance-homogeneity assumptions checked before any p-value could be trusted, and "
        "a single descriptive run like this is not sized for that. Grouped means and "
        "difference contrasts above are observational, not inferential, claims."
    )
    lines.append(
        "- short_prompt and long_prompt differ in task scope as well as length (see Part G "
        "of this project's design notes) -- differences between them are never attributed "
        "to prompt length alone."
    )
    lines.append(
        "- Groq's `seed` parameter (when used) is explicitly documented by the provider as "
        "best-effort, not a determinism guarantee; identical parameters can still produce "
        "different outputs."
    )
    lines.append("")

    # 11. Research conclusions
    lines.append("## 11. Research Conclusions")
    lines.append("")
    lines.append(
        "This section intentionally only restates what the data above actually shows, "
        "re-derived from `measurements` -- it does not add claims beyond that."
    )
    for prompt_type in prompt_types:
        subset = [m for m in successes if m.prompt_type == prompt_type]
        by_effort = compute_grouped_effect(subset, ("reasoning_effort",), "reasoning_tokens")
        ordered = sorted(
            ((level, stats.mean) for level, stats in by_effort.items() if stats.mean is not None),
        )
        if ordered:
            summary_text = ", ".join(f"{level}={mean:.1f}" for level, mean in ordered)
            lines.append(f"- {prompt_type}: observed mean reasoning_tokens by reasoning_effort -- {summary_text}.")
    lines.append(
        "- More reasoning tokens were NOT evaluated against answer quality for long_prompt "
        "(no rubric exists in this project -- see `quality_evaluation_method=\"not_evaluated\"` "
        "in raw_measurements.csv); for short_prompt, `quality_heuristic_passed` is a weak "
        "substring heuristic, not a correctness guarantee, and is reported separately."
    )
    lines.append("")

    return "\n".join(lines)
