"""Markdown analysis report for one Lab06 experiment run.

Mirrors every other redesigned lab's report module structure and "restate
only what the data shows" rule, with sections specific to this lab's
timing semantics (Part C) and an explicit, required limitations section.
"""

from __future__ import annotations

from typing import Optional, Sequence

from labs.lab06_streaming.models import StreamMeasurement
from labs.lab06_streaming.stats import ConditionSummary, PairComparison

MIN_PAIRS_FOR_ROBUST_CLAIM = 30


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
    measurements: Sequence[StreamMeasurement],
    summaries: Sequence[ConditionSummary],
    comparisons: Sequence[PairComparison],
) -> str:
    config = manifest.get("experiment_configuration", {})
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]

    lines: list[str] = []
    lines.append(f"# Lab06 Streaming Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Profile: `{manifest.get('profile')}`")
    lines.append(f"- Model: `{config.get('model')}`")
    lines.append(f"- reasoning_effort: `{config.get('reasoning_effort')}`")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
    lines.append(f"- request_usage_when_supported: {config.get('request_usage_when_supported')}")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append("")

    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Measurements recorded: {len(measurements)}")
    lines.append(f"- Successful: {len(successes)} / Failed: {len(failures)}")
    lines.append("")

    lines.append("## 3. Timing Semantics by Condition (Part C)")
    lines.append("")
    lines.append("| condition | successful | mean first_stream_event_ms | mean first_visible_content_ms | mean total_latency_ms |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition} | {s.successful_calls} | {_fmt(s.first_stream_event_ms.mean)} | "
            f"{_fmt(s.first_visible_content_ms.mean)} | {_fmt(s.total_latency_ms.mean)} |"
        )
    lines.append("")
    lines.append(
        "`ttft_ms` is always `None` in this redesign: no verified provider per-token timing distinct "
        "from the first SSE event's own client-measured arrival time (`first_stream_event_ms`) was "
        "found for Groq -- see labs.lab06_streaming.runner's module docstring. Reporting the first SSE "
        "event as a true provider TTFT would be a mislabeling this redesign explicitly avoids."
    )
    lines.append("")

    lines.append("## 4. Streaming Token Usage Support")
    lines.append("")
    for s in summaries:
        lines.append(f"- {s.condition}: usage available on {s.usage_available_count}/{s.successful_calls} successful call(s)")
    lines.append("")
    lines.append(
        "Streaming usage was requested via `extra_body={\"stream_options\": {\"include_usage\": true}}` "
        "(not a typed SDK parameter -- verified by reading `Completions.create`'s full parameter list) "
        "when `request_usage_when_supported=true`. The provider is never guaranteed to populate it; "
        "when it doesn't, `usage_status` stays `USAGE_UNAVAILABLE` and every dependent field (tokens, "
        "cost, provider throughput) stays `None` rather than being estimated from chunk counts."
    )
    lines.append("")

    lines.append("## 5. Paired Latency Comparison (Part J)")
    lines.append("")
    if comparisons:
        lines.append("| pair | streaming first visible | streaming total | non-streaming total | perceived gain | perceived gain % | total latency diff |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c.pair_id} | {_fmt(c.streaming_first_visible_content_ms)} | {_fmt(c.streaming_total_latency_ms)} | "
                f"{_fmt(c.non_streaming_total_latency_ms)} | {_fmt(c.perceived_latency_gain_ms)} | "
                f"{_fmt_pct(c.perceived_latency_gain_percent)} | {_fmt(c.total_latency_difference_ms)} |"
            )
    else:
        lines.append("No paired streaming/non_streaming comparison is available for this run.")
    lines.append("")
    lines.append(
        "Perceived latency gain percentage is denominated against `non_streaming_total_latency_ms` "
        "(the baseline reference point). A negative gain (streaming's first visible content arrived "
        "LATER than the non-streaming full response) is reported exactly as measured, never hidden or "
        "clamped to zero."
    )
    lines.append("")

    lines.append("## 6. Answer Completeness & Quality Comparison")
    lines.append("")
    lines.append("| condition | completed | incomplete | truncated | no_visible_content | stream_interrupted | api_error |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition} | {s.completed_count} | {s.incomplete_count} | {s.truncated_count} | "
            f"{s.no_visible_content_count} | {s.stream_interrupted_count} | {s.api_error_count} |"
        )
    lines.append("")

    lines.append("## 7. Cost Comparison")
    lines.append("")
    for s in summaries:
        lines.append(f"- {s.condition}: total estimated cost {_fmt_cost(s.total_estimated_cost_usd)} ({s.successful_calls} successful call(s))")
    lines.append("")
    lines.append(
        "Reasoning tokens are already included in `completion_tokens` (the OpenAI-compatible usage "
        "convention Groq follows) and are never added a second time when computing cost."
    )
    lines.append("")

    lines.append("## 8. Experimental Limitations")
    lines.append("")
    pair_count = len({m.pair_id for m in measurements})
    lines.append(f"- This run executed {pair_count} pair(s). Below {MIN_PAIRS_FOR_ROBUST_CLAIM} pairs, these statistics describe this one run only -- no statistical significance, confidence interval, or p95/p99 latency benchmark is claimed.")
    lines.append("- **Streaming does not necessarily make the model generate the complete answer faster.** It can make useful content available to the user earlier -- these are two different claims, and only the second is what `perceived_latency_gain_ms` measures.")
    lines.append("- `first_event_to_visible_gap_ms` reflects SOME combination of reasoning generation, metadata-only events, and transport buffering -- this measurement alone does not prove which cause dominated in any given call.")
    lines.append("- Paired/interleaved execution order reduces, but does not eliminate, time-dependent confounders (e.g. provider load drifting over the run).")
    lines.append("- `effective_completion_tokens_per_second` includes network latency, prompt processing, and reasoning generation time -- it is an end-to-end rate, not a pure model-decoding-speed measurement.")
    lines.append("- A single request's token generation rate is not the same concept as system-level requests-per-second throughput (see LAB08 for concurrent-request throughput).")
    lines.append("")

    return "\n".join(lines)
