"""Markdown analysis report for one Lab03 experiment run.

Mirrors labs.lab02_context_growth.report's structure and "restate only
what the data shows" rule, with sections specific to Lab03's research
questions (exact-prefix caching, request hit rate vs token cache ratio,
cache-aware cost savings, latency) and an explicit, required limitations
section (cache hit is not guaranteed, cache state not fully controllable,
TTL/eviction, provider load, small sample size, eligibility vs observed
caching, correlation vs causal attribution).
"""

from __future__ import annotations

from typing import Sequence

from labs.lab03_prompt_caching.models import CacheMeasurement
from labs.lab03_prompt_caching.stats import CacheConditionSummary, compute_level_difference

MIN_REPETITIONS_FOR_ROBUST_CLAIM = 10


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
    measurements: Sequence[CacheMeasurement],
    summaries: Sequence[CacheConditionSummary],
) -> str:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    config = manifest.get("experiment_configuration", {})
    repetitions = config.get("repetitions")
    by_condition = {s.condition_id: s for s in summaries}

    lines: list[str] = []
    lines.append(f"# Lab03 Prompt Caching Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    # 1. Configuration
    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Model: `{manifest.get('model')}`")
    lines.append(f"- Repetitions (pairs): {repetitions}")
    lines.append(f"- max_completion_tokens: {config.get('max_completion_tokens')}")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append("")

    # 2. Requests and measurements
    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Measurements recorded: {len(measurements)}")
    lines.append(f"- Successful: {len(successes)} / Failed: {len(failures)}")
    if failures:
        lines.append("- Failure reasons observed (not necessarily exhaustive):")
        for err in sorted({f.error for f in failures if f.error})[:10]:
            lines.append(f"  - {err}")
    lines.append("")

    # 3. Cache observability (RQ1-RQ3)
    lines.append("## 3. Cache Observability (RQ1-RQ3)")
    lines.append("")
    lines.append(
        "| condition_id | successful_calls | reported_cache_usage_calls | "
        "observed request hit rate (Metric A) | token cache ratio (Metric B) |"
    )
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition_id} | {s.successful_calls} | {s.reported_cache_usage_calls} | "
            f"{_fmt_pct(s.observed_request_cache_hit_rate * 100) if s.observed_request_cache_hit_rate is not None else 'N/A'} | "
            f"{_fmt_pct(s.token_cache_ratio * 100) if s.token_cache_ratio is not None else 'N/A'} |"
        )
    lines.append("")
    lines.append(
        "**Metric A (observed request cache hit rate)** and **Metric B (token cache ratio)** answer "
        "different questions and are never the same number: A is the fraction of REQUESTS with any "
        "cache hit; B is the fraction of TOKENS, across all reporting requests, that were cached."
    )
    lines.append("")
    lines.append("**Cache Status Distribution (4-way, see labs.lab03_prompt_caching.cache_status)**")
    lines.append("")
    lines.append("| condition_id | CACHE_HIT | CACHE_MISS | CACHE_USAGE_UNAVAILABLE | INVALID_USAGE |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        counts = s.cache_status_counts
        lines.append(
            f"| {s.condition_id} | {counts.get('CACHE_HIT', 0)} | {counts.get('CACHE_MISS', 0)} | "
            f"{counts.get('CACHE_USAGE_UNAVAILABLE', 0)} | {counts.get('INVALID_USAGE', 0)} |"
        )
    lines.append("")
    lines.append(
        "CACHE_USAGE_UNAVAILABLE means the provider never included `prompt_tokens_details` in that "
        "call's response at all -- it is never counted as a CACHE_MISS (which requires the provider "
        "to have actually reported `cached_tokens == 0`)."
    )
    lines.append("")
    lines.append("**Prompt / Cached / Uncached Tokens by Condition**")
    lines.append("")
    lines += _stats_table_header()
    for s in summaries:
        lines.append(_stats_row(f"{s.condition_id}.prompt_tokens", s.prompt_tokens))
        lines.append(_stats_row(f"{s.condition_id}.cached_tokens", s.cached_tokens))
        lines.append(_stats_row(f"{s.condition_id}.uncached_prompt_tokens", s.uncached_prompt_tokens))
    lines.append("")

    # 4. Cost economics (RQ4)
    lines.append("## 4. Cost Economics (RQ4)")
    lines.append("")
    lines.append(
        "| condition_id | total cost | mean cost/call | total cache savings | mean savings/call | mean savings % |"
    )
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.condition_id} | {_fmt_cost(s.total_estimated_cost_usd)} | {_fmt_cost(s.mean_estimated_cost_usd)} | "
            f"{_fmt_cost(s.total_estimated_cache_savings_usd)} | {_fmt_cost(s.mean_estimated_cache_savings_usd)} | "
            f"{_fmt_pct(s.mean_estimated_cache_savings_percent)} |"
        )
    lines.append("")
    lines.append(
        "Savings figures are a counterfactual re-pricing of the SAME measured token usage under "
        "no-caching pricing -- never the measured cost of a second, actually-executed API call."
    )
    lines.append("")

    # 5. Latency (RQ5)
    lines.append("## 5. Latency by Condition (RQ5)")
    lines.append("")
    lines += _stats_table_header()
    for s in summaries:
        lines.append(_stats_row(s.condition_id, s.latency_ms))
    lines.append("")
    lines.append("**Pairwise Latency Difference (mean, ms; cache_unfriendly - repeated_prefix)**")
    lines.append("")
    latency_by_condition = {cid: s.latency_ms for cid, s in by_condition.items()}
    if "repeated_prefix" in latency_by_condition and "cache_unfriendly" in latency_by_condition:
        diff = compute_level_difference(latency_by_condition, "repeated_prefix", "cache_unfriendly")
        lines.append(f"- repeated_prefix -> cache_unfriendly: {_fmt(diff)}")
    lines.append(
        "Caching's documented benefit is on the input/prefill side; final answer generation still "
        "happens regardless of a cache hit, so a latency difference between conditions is not "
        "automatically attributable to caching alone (see Section 7)."
    )
    lines.append("")

    # 6. Answer correctness
    lines.append("## 6. Answer Correctness")
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
        "A condition is never judged a successful optimization merely for being cheaper or faster "
        "if its answers are not also correct -- cost/latency and correctness are reported separately."
    )
    lines.append("")

    # 7. Limitations (explicitly required)
    lines.append("## 7. Experimental Limitations")
    lines.append("")
    lines.append(f"- Each condition was repeated {repetitions} time(s) (paired). Below {MIN_REPETITIONS_FOR_ROBUST_CLAIM} repetitions, these statistics describe this one run only.")
    lines.append("- **Cache hit is not guaranteed.** Groq's prompt caching is provider-managed; a miss on repeated_prefix is not evidence of a bug.")
    lines.append("- **Cache state cannot be fully controlled by this experiment.** Prior, unrelated traffic to the same model may have already warmed or evicted the relevant cache entry before this run started.")
    lines.append("- **Cache expiration/eviction**: Groq documents an approximate multi-hour TTL; cache state at the start of a run is not observable or resettable by this code.")
    lines.append("- **Provider workload variability** can affect latency independent of caching.")
    lines.append("- **Small sample size**: no confidence intervals or hypothesis tests are computed here.")
    lines.append("- **Output length variability**: completion/reasoning token counts are not held fixed across calls.")
    lines.append("- **Cache eligibility vs. observed caching**: exceeding the documented minimum cacheable prompt length makes a hit POSSIBLE, never certain.")
    lines.append("- **Observed correlation vs. causal attribution**: any latency/cost difference between conditions is reported as an observation, not asserted as proof caching caused it.")
    lines.append("")

    lines.append("## 8. Research Conclusions")
    lines.append("")
    lines.append("This section restates only what the data above actually shows; it does not add claims beyond that.")
    ordered = sorted((s.condition_id, s.mean_estimated_cost_usd) for s in summaries if s.mean_estimated_cost_usd is not None)
    if ordered:
        summary_text = ", ".join(f"{cid}={_fmt_cost(cost)}" for cid, cost in ordered)
        lines.append(f"- Observed mean cost per call by condition -- {summary_text}.")
    lines.append("")

    return "\n".join(lines)
