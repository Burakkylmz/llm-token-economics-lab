"""Markdown analysis report for one Lab08 experiment run.

Mirrors every other redesigned lab's report module structure, with an
explicit, required limitations section restating this lab's non-negotiable
rules (Part T): concurrency is never claimed to reduce individual request
latency, configured concurrency is never equated with observed concurrency,
and eight requests are never presented as a statistically reliable p95
latency benchmark.
"""

from __future__ import annotations

from typing import Sequence

from labs.lab08_concurrency.models import ConcurrencyMeasurement
from labs.lab08_concurrency.stats import ConcurrencyComparison, ConditionSummary

MIN_REQUESTS_FOR_ROBUST_PERCENTILE = 30


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
    measurements: Sequence[ConcurrencyMeasurement],
    summaries: Sequence[ConditionSummary],
    comparisons: Sequence[ConcurrencyComparison],
) -> str:
    config = manifest.get("experiment_configuration", {})
    successes = [m for m in measurements if m.request_status == "SUCCESS"]
    failures = [m for m in measurements if m.request_status != "SUCCESS"]

    lines: list[str] = []
    lines.append(f"# Lab08 Async Processing & Concurrency Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append(f"- Profile: `{manifest.get('profile')}`")
    lines.append(f"- Model: `{config.get('model')}`")
    lines.append(f"- reasoning_effort: `{config.get('reasoning_effort')}`")
    lines.append(f"- SDK max_retries (this lab's own client only): {config.get('sdk_max_retries')}")
    lines.append(f"- Planned calls: {manifest.get('planned_calls')}")
    lines.append("")

    lines.append("## 2. Total Requests and Successful Measurements")
    lines.append("")
    lines.append(f"- Measurements recorded: {len(measurements)}")
    lines.append(f"- Successful: {len(successes)} / Failed: {len(failures)}")
    lines.append("")

    lines.append("## 3. Condition Summary by Concurrency Level")
    lines.append("")
    lines.append("| concurrency | mode | success rate | observed peak in-flight | utilization | batch wall-clock (ms) | successful RPS | mean queue wait (ms) | median service latency (ms) | p95 end-to-end (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.max_concurrent} | {s.execution_mode} | {_fmt_pct((s.success_rate or 0) * 100)} "
            f"({s.successful_requests}/{s.total_requests}) | {s.observed_peak_in_flight_requests} | "
            f"{_fmt_pct((s.concurrency_utilization_ratio or 0) * 100) if s.concurrency_utilization_ratio is not None else 'N/A'} | "
            f"{_fmt(s.batch_wall_clock_ms)} | {_fmt(s.successful_requests_per_second)} | {_fmt(s.queue_wait_ms.mean)} | "
            f"{_fmt(s.request_service_latency_ms.median)} | {_fmt(s.task_end_to_end_latency_ms.p95)} |"
        )
    lines.append("")
    lines.append(
        "**`observed_peak_in_flight_requests` is a measured value, never assumed equal to the "
        "configured concurrency limit** -- it can only ever be <= the configured limit (the semaphore "
        "enforces that), but is reported separately to make the distinction explicit."
    )
    lines.append("")

    lines.append("## 4. Baseline Comparisons")
    lines.append("")
    if comparisons:
        lines.append("| baseline | compared | throughput speedup | batch time reduction | median service latency delta | median end-to-end delta | success rate delta | cost delta |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c.baseline_concurrency} | {c.compared_concurrency} | "
                f"{_fmt(c.throughput_speedup_ratio)}x | {_fmt_pct(c.batch_time_reduction_percent)} | "
                f"{_fmt_pct(c.median_service_latency_delta_percent)} | {_fmt_pct(c.median_task_end_to_end_latency_delta_percent)} | "
                f"{_fmt(c.success_rate_delta)} | {_fmt_pct(c.cost_delta_percent)} |"
            )
    else:
        lines.append("No baseline comparison is available (only one concurrency level was run).")
    lines.append("")
    lines.append(
        "**Increasing concurrency can improve throughput while simultaneously increasing individual "
        "request latency** -- a positive `throughput speedup` and a positive `median service latency "
        "delta` (latency got WORSE) can both be true at once, and this report never collapses them "
        "into a single score."
    )
    lines.append("")

    lines.append("## 5. Cost & Quality Comparison")
    lines.append("")
    for s in summaries:
        lines.append(
            f"- concurrency={s.max_concurrent}: total cost {_fmt_cost(s.total_estimated_cost_usd)}, "
            f"cost/successful request {_fmt_cost(s.cost_per_successful_request_usd)}, correct answers {s.correct_count}/{s.successful_requests}"
        )
    lines.append("")

    lines.append("## 6. Experimental Limitations")
    lines.append("")
    total_requests = sum(s.total_requests for s in summaries)
    lines.append(f"- This run recorded {total_requests} total request(s) across all concurrency levels. Below {MIN_REQUESTS_FOR_ROBUST_PERCENTILE} requests per condition, P95/P99 latency is NOT a statistically reliable tail-latency benchmark -- it is reported as a descriptive observation of this one run only.")
    lines.append("- `concurrency_utilization_ratio` is a time-weighted average over this run's own batch span, not a claim about sustained production behavior.")
    lines.append("- A semaphore limits simultaneous in-flight requests; it does NOT guarantee compliance with the provider's actual RPM/TPM quota -- these are different concepts (Part H).")
    lines.append("- This lab's own AsyncGroq client disables the SDK's default automatic retries (max_retries=0) specifically so measured latency/attempt counts are not inflated by hidden retries -- a production client would typically NOT disable them.")
    lines.append("- No single batch run determines a production-optimal concurrency value; saturation, rate limits, and cost trade-offs can shift under different load patterns.")
    lines.append("")

    return "\n".join(lines)
