"""Markdown analysis report for one Lab02 experiment run.

Mirrors labs.lab01_token_anatomy.effort_report's structure and the same
"restate only what the data shows, never add a claim beyond it" rule, with
sections specific to Lab02's research questions (context growth/bloat,
answer correctness, multi-turn growth) and an explicit confounders section
(information position, relevant-fact repetition, output length, reasoning
variability, provider load) required by this lab's design task.
"""

from __future__ import annotations

from typing import Sequence

from labs.lab02_context_growth.models import ContextMeasurement, TurnMeasurement
from labs.lab02_context_growth.stats import ContextConditionSummary, compute_level_difference

MIN_REPETITIONS_FOR_ROBUST_CLAIM = 10


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value) -> str:
    return f"${value:.6f}" if value is not None else "N/A"


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
    context_measurements: Sequence[ContextMeasurement],
    turn_measurements: Sequence[TurnMeasurement],
    summaries: Sequence[ContextConditionSummary],
) -> str:
    successes = [m for m in context_measurements if m.succeeded]
    failures = [m for m in context_measurements if not m.succeeded]
    config = manifest.get("experiment_configuration", {})
    repetitions = config.get("repetitions")
    by_condition = {s.condition_id: s for s in summaries}

    lines: list[str] = []
    lines.append(f"# Lab02 Context Growth & Context Bloat Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    # 1. Configuration
    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Model: `{manifest.get('model')}`")
    lines.append(f"- Repetitions per context condition: {repetitions}")
    lines.append(f"- Multi-turn turns: {config.get('multi_turn_turns')}")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
    lines.append(f"- Planned context calls: {manifest.get('planned_context_calls')}")
    lines.append(f"- Planned multi-turn calls: {manifest.get('planned_turn_calls')}")
    lines.append(f"- Planned calls (total): {manifest.get('planned_calls')}")
    lines.append("")

    # 2. Requests and measurements
    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Context-condition measurements recorded: {len(context_measurements)}")
    lines.append(f"- Successful: {len(successes)} / Failed: {len(failures)}")
    lines.append(f"- Multi-turn measurements recorded: {len(turn_measurements)}")
    if failures:
        lines.append("- Failure reasons observed (not necessarily exhaustive):")
        for err in sorted({f.error for f in failures if f.error})[:10]:
            lines.append(f"  - {err}")
    lines.append("")

    # 3. Context growth / bloat comparison (RQ1, RQ2, RQ3)
    lines.append("## 3. Context Growth & Context Bloat (RQ1-RQ3)")
    lines.append("")
    lines.append("| condition_id | context_chars | successful_calls | mean prompt_tokens | mean latency (ms) | mean cost |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition_id} | {_fmt(s.context_chars, 0)} | {s.successful_calls} | "
            f"{_fmt(s.prompt_tokens.mean)} | {_fmt(s.latency_ms.mean)} | {_fmt_cost(s.mean_estimated_cost_usd)} |"
        )
    lines.append("")
    lines.append("**Reasoning Tokens by Condition**")
    lines.append("")
    lines += _stats_table_header()
    for s in summaries:
        lines.append(_stats_row(s.condition_id, s.reasoning_tokens))
    lines.append("")
    lines.append("**Completion Tokens by Condition**")
    lines.append("")
    lines += _stats_table_header()
    for s in summaries:
        lines.append(_stats_row(s.condition_id, s.completion_tokens))
    lines.append("")

    # Recommended pairwise comparisons.
    lines.append("**Pairwise Latency Differences (mean, ms; B - A)**")
    lines.append("")
    pairs = [("minimal_context", "expanded_context"), ("minimal_context", "irrelevant_context"), ("expanded_context", "irrelevant_context")]
    latency_by_condition = {cid: s.latency_ms for cid, s in by_condition.items()}
    for a, b in pairs:
        if a in latency_by_condition and b in latency_by_condition:
            diff = compute_level_difference(latency_by_condition, a, b)
            lines.append(f"- {a} -> {b}: {_fmt(diff)}")
    lines.append("")

    # 4. Answer correctness (Part C)
    lines.append("## 4. Answer Correctness")
    lines.append("")
    lines.append("| condition_id | correct | incorrect | format_invalid | missing_response |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition_id} | {s.correct_count} | {s.incorrect_count} | "
            f"{s.format_invalid_count} | {s.missing_response_count} |"
        )
    lines.append("")
    lines.append(
        "Evaluation is strict exact-integer matching (see labs.lab02_context_growth.evaluation) -- "
        "a response that merely mentions the expected digit without answering in the required "
        "format is scored `format_invalid`, never `correct`."
    )
    lines.append("")

    # 5. Multi-turn growth (RQ4)
    lines.append("## 5. Multi-Turn Growth (RQ4)")
    lines.append("")
    turn_successes = [t for t in turn_measurements if t.succeeded]
    if turn_successes:
        lines.append("| turn | messages_count | prompt_tokens | incremental_prompt_tokens | cumulative_prompt_tokens | cumulative_cost |")
        lines.append("|---|---|---|---|---|---|")
        for t in turn_successes:
            lines.append(
                f"| {t.turn_index} | {t.messages_count} | {_fmt(t.prompt_tokens, 0)} | "
                f"{_fmt(t.incremental_prompt_tokens, 0)} | {_fmt(t.cumulative_prompt_tokens, 0)} | "
                f"{_fmt_cost(t.cumulative_estimated_cost_usd)} |"
            )
    else:
        lines.append("_No successful multi-turn measurements were recorded for this run._")
    lines.append("")
    lines.append(
        "Incremental growth (turn-over-turn prompt_tokens increase) and cumulative consumption "
        "(the running total of prompt_tokens re-sent across the whole conversation) are distinct "
        "numbers: a prior turn's assistant reply is re-sent as input on every subsequent turn, so "
        "cumulative_prompt_tokens grows faster than any single turn's own prompt_tokens would suggest."
    )
    lines.append("")

    # 6. Statistical limitations
    lines.append("## 6. Statistical Limitations")
    lines.append("")
    lines.append(
        f"- Each context condition was repeated {repetitions} time(s). Below "
        f"{MIN_REPETITIONS_FOR_ROBUST_CLAIM} repetitions, these statistics are descriptive of this "
        "one run only, not proof of the model's general behavior."
    )
    if repetitions == 1:
        lines.append(
            "- With exactly 1 repetition per condition, sample standard deviation and p95 are "
            "undefined (`N/A`) for every condition -- this is expected, not a bug."
        )
    lines.append(
        "- No confidence intervals or hypothesis tests are computed; none of the numbers above "
        "should be read as a statistically significant difference."
    )
    lines.append("- The large irrelevant_context latency observed in exploratory runs of this lab is NOT explained by this report alone (see confounders below); it is not attributed to a single cause.")
    lines.append("")

    # 7. Confounders (explicitly required by this lab's design task).
    lines.append("## 7. Experimental Confounders (Not Isolated by This Design)")
    lines.append("")
    lines.append(
        "Changing context relevance necessarily changes its content, so this experiment cannot "
        "fully separate \"size\" from \"relevance\" as independent causes. Known, un-isolated "
        "confounders:"
    )
    lines.append("- **Information position**: the canonical fact sits at the START of expanded_context but at the END of irrelevant_context.")
    lines.append("- **Relevant-fact repetition count**: held at exactly 1 occurrence in both conditions, but the surrounding repeated filler differs in structure.")
    lines.append("- **Output length**: the model's reasoning/response length is not held fixed across conditions.")
    lines.append("- **Reasoning-token variability**: reasoning token usage can vary run-to-run independent of context.")
    lines.append("- **Provider load**: latency in particular can reflect momentary Groq-side load, not only this run's own context size.")
    lines.append("")

    # 8. Research conclusions
    lines.append("## 8. Research Conclusions")
    lines.append("")
    lines.append(
        "This section restates only what the data above actually shows, re-derived from the "
        "measurements -- it does not add claims beyond that."
    )
    ordered = sorted((s.condition_id, s.prompt_tokens.mean) for s in summaries if s.prompt_tokens.mean is not None)
    if ordered:
        summary_text = ", ".join(f"{cid}={mean:.1f}" for cid, mean in ordered)
        lines.append(f"- Observed mean prompt_tokens by condition -- {summary_text}.")
    lines.append(
        "- Any claim about WHY latency differs between conditions of similar input size is "
        "exploratory only (see Section 7); this report does not assert a single causal mechanism."
    )
    lines.append("")

    return "\n".join(lines)
