"""The ONE shared recompute pipeline for Lab09 -- used identically by
`lab09 all` (right after a live real-time run), `lab09 fetch_batch_results`
(right after Batch reconciliation), and `lab09 evaluate` (fully offline,
from whatever has already been persisted). A run backfilled this way
always produces byte-identical derived numbers to a freshly-executed one,
since there is only one code path that computes them.

Never calls the Groq API -- reads only what `persistence.py` has already
written for this run_dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataclasses import asdict

from lab_core.config import Lab09Config, ModelPricing
from labs.lab09_batch_processing.models import RECON_COMPLETED, STATUS_SUCCESS
from labs.lab09_batch_processing.persistence import (
    read_batch_job,
    read_batch_measurements,
    read_manifest,
    read_realtime_measurements,
    write_analysis_report,
    write_cost_comparison_csv,
    write_experiment_summary_json,
    write_quality_comparison_csv,
)
from labs.lab09_batch_processing.pricing import compare_standard_vs_batch
from labs.lab09_batch_processing.report import build_analysis_report_markdown
from labs.lab09_batch_processing.stats import (
    compare_realtime_conditions,
    summarize_quality_batch,
    summarize_quality_realtime,
    summarize_realtime_condition,
)


def _token_totals(measurements) -> dict:
    prompt = [m.prompt_tokens for m in measurements if m.prompt_tokens is not None]
    completion = [m.completion_tokens for m in measurements if m.completion_tokens is not None]
    return {
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
    }


def _wall_clock_from_measurements(measurements) -> Optional[float]:
    """Recomputed estimate for offline `evaluate`/backfill (no fresh timer
    available): the span between the first and last recorded timestamp is
    NOT reliable at this granularity, so this is intentionally left to the
    live run's own measured value, persisted in the manifest, when present.
    Offline recompute paths that lack it report batch_wall_clock_ms as None
    rather than approximate it."""
    return None


def compute_and_persist_summary(run_dir: Path, lab09_config: Lab09Config, model: str, pricing: Optional[ModelPricing]) -> dict:
    manifest = read_manifest(run_dir) or {}

    sequential_measurements = read_realtime_measurements(run_dir, "sequential")
    concurrent_measurements = read_realtime_measurements(run_dir, "concurrent")
    batch_measurements = read_batch_measurements(run_dir)
    batch_job = read_batch_job(run_dir)

    sequential_summary = None
    if sequential_measurements:
        wall_clock = manifest.get("sequential_wall_clock_ms")
        peak = manifest.get("sequential_peak_in_flight", 1)
        sequential_summary = summarize_realtime_condition("sequential", 1, sequential_measurements, wall_clock, peak)

    concurrent_summary = None
    if concurrent_measurements:
        wall_clock = manifest.get("concurrent_wall_clock_ms")
        peak = manifest.get("concurrent_peak_in_flight", 0)
        max_concurrent = concurrent_measurements[0].max_concurrent
        concurrent_summary = summarize_realtime_condition("concurrent", max_concurrent, concurrent_measurements, wall_clock, peak)

    realtime_comparison = None
    if sequential_summary and concurrent_summary:
        realtime_comparison = compare_realtime_conditions(sequential_summary, concurrent_summary)

    quality_summaries = []
    if sequential_measurements:
        quality_summaries.append(summarize_quality_realtime("sequential", sequential_measurements))
    if concurrent_measurements:
        quality_summaries.append(summarize_quality_realtime("concurrent", concurrent_measurements))
    if batch_measurements:
        quality_summaries.append(summarize_quality_batch(batch_measurements))

    token_totals = {}
    if sequential_measurements:
        token_totals["sequential"] = _token_totals(sequential_measurements)
    if concurrent_measurements:
        token_totals["concurrent"] = _token_totals(concurrent_measurements)
    if batch_measurements:
        completed = [m for m in batch_measurements if m.reconciliation_status == RECON_COMPLETED]
        token_totals["batch"] = _token_totals(completed)

    pricing_comparison = None
    batch_counts = None
    if batch_measurements:
        completed = [m for m in batch_measurements if m.reconciliation_status == RECON_COMPLETED]
        standard_costs = [m.estimated_standard_cost_usd for m in completed if m.estimated_standard_cost_usd is not None]
        batch_costs = [m.estimated_batch_cost_usd for m in completed if m.estimated_batch_cost_usd is not None]
        standard_total = sum(standard_costs) if standard_costs else None
        batch_total = sum(batch_costs) if batch_costs else None
        pricing_comparison = compare_standard_vs_batch(standard_total, batch_total)

        from collections import Counter

        status_counts = Counter(m.reconciliation_status for m in batch_measurements)
        batch_counts = {
            "planned_requests": len(batch_measurements) if not batch_job else manifest.get("batch_input_requests"),
            "completed_requests": status_counts.get(RECON_COMPLETED, 0),
            "failed_requests": status_counts.get("FAILED", 0),
            "missing_requests": status_counts.get("MISSING", 0),
            "duplicate_result_ids": status_counts.get("DUPLICATE", 0),
            "unexpected_result_ids": status_counts.get("UNKNOWN_ID", 0),
        }

    summary_payload = {
        "experiment_run_id": manifest.get("experiment_run_id", run_dir.name),
        "profile": manifest.get("profile"),
        "sequential_summary": asdict(sequential_summary) if sequential_summary else None,
        "concurrent_summary": asdict(concurrent_summary) if concurrent_summary else None,
        "realtime_comparison": asdict(realtime_comparison) if realtime_comparison else None,
        "batch_job": asdict(batch_job) if batch_job else None,
        "batch_counts": batch_counts,
        "quality_summaries": [asdict(q) for q in quality_summaries],
        "pricing_comparison": asdict(pricing_comparison) if pricing_comparison else None,
        "token_totals": token_totals,
    }
    write_experiment_summary_json(run_dir, summary_payload)

    if quality_summaries:
        write_quality_comparison_csv(run_dir, quality_summaries)
    if pricing_comparison is not None:
        write_cost_comparison_csv(run_dir, pricing_comparison)

    report_counts = None
    if batch_counts is not None:
        from labs.lab09_batch_processing.models import ReconciliationCounts

        report_counts = ReconciliationCounts(
            planned_requests=batch_counts.get("planned_requests") or 0,
            completed_requests=batch_counts["completed_requests"],
            failed_requests=batch_counts["failed_requests"],
            missing_requests=batch_counts["missing_requests"],
            duplicate_result_ids=batch_counts["duplicate_result_ids"],
            unexpected_result_ids=batch_counts["unexpected_result_ids"],
        )

    report_markdown = build_analysis_report_markdown(
        manifest, sequential_summary, concurrent_summary, realtime_comparison,
        batch_job, report_counts, quality_summaries, pricing_comparison,
    )
    write_analysis_report(run_dir, report_markdown)

    return summary_payload
