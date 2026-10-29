"""Builds Lab09's `analysis_report.md` -- the 14-section structure from
spec Part 14. If the Batch job has not finished, Batch-dependent sections
report `BATCH_PENDING` rather than inventing numbers, while the
already-available real-time sections stay fully populated.
"""

from __future__ import annotations

from typing import Optional

from labs.lab09_batch_processing.models import LOCAL_STATE_EVALUATED


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, float) else (f"{value:.1f}%" if value is not None else "N/A")


def build_analysis_report_markdown(
    manifest: dict,
    sequential_summary,
    concurrent_summary,
    realtime_comparison,
    batch_job: Optional[object],
    batch_counts: Optional[object],
    quality_summaries: list,
    pricing_comparison: Optional[object],
) -> str:
    lines: list[str] = []
    lines.append("# Lab09 -- Batch Processing: Analysis Report\n")

    lines.append("## 1. Experiment Configuration\n")
    lines.append(f"- Experiment Run ID: `{manifest.get('experiment_run_id')}`")
    lines.append(f"- Profile: `{manifest.get('profile')}`")
    lines.append(f"- Planned Real-Time Calls: {manifest.get('planned_realtime_calls')}")
    lines.append(f"- Batch Input Requests: {manifest.get('batch_input_requests')}\n")

    lines.append("## 2. Hypotheses\n")
    lines.append("- H1 (Workflow Latency): concurrent execution can reduce total wall-clock time vs sequential.")
    lines.append("- H2 (Per-Request Latency): higher concurrency does not necessarily reduce individual request latency.")
    lines.append("- H3 (Batch Economics): provider-managed batch processing may offer a lower price for the same workload.")
    lines.append("- H4 (Operational Trade-off): batch processing optimizes for cost-efficient deferred execution, not low latency.")
    lines.append("- H5 (Quality Preservation): faster/cheaper execution is not automatically successful if correctness drops.\n")

    lines.append("## 3. Workload and Controlled Variables\n")
    lines.append("Model, workload version, task IDs/prompts/expected answers, reasoning_effort, and "
                  "max_completion_tokens are held identical across sequential, concurrent, and batch "
                  "conditions. The only difference between conditions is the execution mechanism.\n")

    lines.append("## 4. Sequential Measurements\n")
    if sequential_summary is not None:
        lines.append(f"- Total Requests: {sequential_summary.total_requests}")
        lines.append(f"- Successful Requests: {sequential_summary.successful_requests}")
        lines.append(f"- Batch Wall-Clock: {_fmt(sequential_summary.batch_wall_clock_ms)} ms")
        lines.append(f"- Median Service Latency: {_fmt(sequential_summary.request_service_latency_ms.median)} ms")
        lines.append(f"- Correct Answers: {sequential_summary.correct_count}/{sequential_summary.successful_requests}\n")
    else:
        lines.append("Not yet executed.\n")

    lines.append("## 5. Concurrent Measurements\n")
    if concurrent_summary is not None:
        lines.append(f"- Total Requests: {concurrent_summary.total_requests}")
        lines.append(f"- Successful Requests: {concurrent_summary.successful_requests}")
        lines.append(f"- Observed Peak In-Flight: {concurrent_summary.observed_peak_in_flight_requests}")
        lines.append(f"- Batch Wall-Clock: {_fmt(concurrent_summary.batch_wall_clock_ms)} ms")
        lines.append(f"- Median Service Latency: {_fmt(concurrent_summary.request_service_latency_ms.median)} ms")
        lines.append(f"- Correct Answers: {concurrent_summary.correct_count}/{concurrent_summary.successful_requests}\n")
    else:
        lines.append("Not yet executed.\n")

    if realtime_comparison is not None:
        lines.append("### Sequential vs Concurrent\n")
        lines.append(f"- Throughput Speedup: {_fmt(realtime_comparison.throughput_speedup_ratio)}x")
        lines.append(f"- Wall-Clock Reduction: {_fmt_pct(realtime_comparison.batch_time_reduction_percent)}")
        lines.append(f"- Median Service Latency Change: {_fmt_pct(realtime_comparison.median_service_latency_delta_percent)}\n")

    lines.append("## 6. Batch Job Status\n")
    if batch_job is None or batch_job.local_lifecycle_state == "NOT_SUBMITTED":
        lines.append("BATCH_PENDING -- the Batch job has not been submitted for this run yet. "
                      "Run `lab09 submit_batch --experiment-run-id <id> --live` to submit it.\n")
    else:
        lines.append(f"- Local Lifecycle State: `{batch_job.local_lifecycle_state}`")
        lines.append(f"- Provider Status: `{batch_job.provider_status}`")
        lines.append(f"- Batch ID: `{batch_job.batch_id}`")
        lines.append(f"- Request Counts: total={batch_job.request_counts_total} "
                      f"completed={batch_job.request_counts_completed} failed={batch_job.request_counts_failed}\n")
        if batch_job.local_lifecycle_state != LOCAL_STATE_EVALUATED:
            lines.append("BATCH_PENDING -- results not yet imported/evaluated.\n")

    lines.append("## 7. Request-Level Quality Evaluation\n")
    if quality_summaries:
        for q in quality_summaries:
            lines.append(f"- {q.condition}: {q.correct_count}/{q.total_evaluated} correct "
                          f"({_fmt_pct(q.accuracy)})")
        lines.append("")
    else:
        lines.append("Not yet available.\n")

    lines.append("## 8. Token Economics\n")
    lines.append("See `token_consumption_breakdown` chart and per-condition CSVs for prompt/completion/"
                  "total token counts. Missing usage is reported as N/A, never as zero.\n")

    lines.append("## 9. Standard vs Batch Cost\n")
    if pricing_comparison is not None and pricing_comparison.estimated_standard_cost_usd is not None:
        lines.append(f"- Estimated Standard Cost: ${_fmt(pricing_comparison.estimated_standard_cost_usd, 8)}")
        lines.append(f"- Estimated Batch Cost: ${_fmt(pricing_comparison.estimated_batch_cost_usd, 8)}")
        lines.append(f"- Estimated Savings: {_fmt_pct(pricing_comparison.estimated_savings_percent)}")
        lines.append(f"- Pricing Source: {pricing_comparison.pricing_source} (verified {pricing_comparison.pricing_verified_date})\n")
    else:
        lines.append("BATCH_PENDING -- standard-vs-batch cost comparison requires completed Batch results.\n")

    lines.append("## 10. Workflow and Job Timing\n")
    lines.append("Real-time batch wall-clock time (this process's own measured duration) is reported "
                  "separately from any Batch job turnaround -- the latter is provider-managed and is "
                  "never presented as an individual request latency.\n")

    lines.append("## 11. Quality-Cost Trade-offs\n")
    lines.append("A cheaper or faster condition is not reported as superior unless its accuracy is also "
                  "shown alongside it -- see Section 7.\n")

    lines.append("## 12. Failures and Missing Measurements\n")
    if batch_counts is not None:
        lines.append(f"- Batch Missing Requests: {batch_counts.missing_requests}")
        lines.append(f"- Batch Failed Requests: {batch_counts.failed_requests}")
        lines.append(f"- Duplicate Result IDs: {batch_counts.duplicate_result_ids}")
        lines.append(f"- Unexpected Result IDs: {batch_counts.unexpected_result_ids}\n")
    else:
        lines.append("No Batch reconciliation data available yet.\n")

    lines.append("## 13. Experimental Limitations\n")
    lines.append("- Real-time latency measurements are client-side (this process's own wall-clock), "
                  "not independently verified against provider-side logs.")
    lines.append("- A single repetition per real-time condition means variability is descriptive only, "
                  "not a statistically powered comparison.")
    lines.append("- Batch job turnaround depends on Groq's own queue and completion window and is "
                  "never extrapolated or simulated when unobserved.\n")

    lines.append("## 14. Technical Conclusions\n")
    lines.append("- Concurrency changes wall-clock time, not necessarily per-request latency (H1/H2).")
    lines.append("- Batch processing trades immediate results for a lower price -- a deliberate shift "
                  "in optimization target, not a strictly better option (H3/H4).")
    lines.append("- A cheaper or faster execution strategy is only a genuine improvement if it preserves "
                  "answer correctness on the same golden workload (H5).\n")

    return "\n".join(lines)
