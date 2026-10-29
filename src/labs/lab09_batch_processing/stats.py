"""Descriptive statistics and condition summaries for Lab09.

`DescriptiveStats`/`compute_descriptive_stats` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats (same precedent as
labs.lab08_concurrency.stats).

Distinguishes, per spec Part 10's "Separate Comparisons":
  - real-time workflow performance (sequential vs concurrent): `summarize_realtime_condition`/`compare_realtime_conditions`
  - deferred processing economics (standard vs batch pricing): `pricing.compare_standard_vs_batch`
  - quality (same golden answers across all three conditions): `summarize_quality`
These are never collapsed into one combined ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
)
from labs.lab09_batch_processing.evaluation import STATUS_CORRECT
from labs.lab09_batch_processing.models import (
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_STARTED,
    RECON_COMPLETED,
    STATUS_SUCCESS,
    BatchRequestMeasurement,
    RealtimeMeasurement,
    RequestTimelineEvent,
)


@dataclass
class RealtimeConditionSummary:
    condition: str
    max_concurrent: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: Optional[float]

    observed_peak_in_flight_requests: int

    batch_wall_clock_ms: Optional[float]
    successful_requests_per_second: Optional[float]

    queue_wait_ms: DescriptiveStats
    request_service_latency_ms: DescriptiveStats
    task_end_to_end_latency_ms: DescriptiveStats

    total_estimated_cost_usd: Optional[float]
    correct_count: int


def summarize_realtime_condition(
    condition: str,
    max_concurrent: int,
    measurements: Sequence[RealtimeMeasurement],
    batch_wall_clock_ms: Optional[float],
    observed_peak_in_flight: int,
) -> RealtimeConditionSummary:
    total = len(measurements)
    successes = [m for m in measurements if m.request_status == STATUS_SUCCESS]
    costs = [m.estimated_standard_cost_usd for m in measurements if m.estimated_standard_cost_usd is not None]

    duration_s = (batch_wall_clock_ms / 1000) if batch_wall_clock_ms else None
    successful_rps = (len(successes) / duration_s) if duration_s else None

    return RealtimeConditionSummary(
        condition=condition,
        max_concurrent=max_concurrent,
        total_requests=total,
        successful_requests=len(successes),
        failed_requests=total - len(successes),
        success_rate=(len(successes) / total) if total else None,
        observed_peak_in_flight_requests=observed_peak_in_flight,
        batch_wall_clock_ms=batch_wall_clock_ms,
        successful_requests_per_second=successful_rps,
        queue_wait_ms=compute_descriptive_stats([m.queue_wait_ms for m in measurements]),
        request_service_latency_ms=compute_descriptive_stats([m.request_service_latency_ms for m in measurements]),
        task_end_to_end_latency_ms=compute_descriptive_stats([m.task_end_to_end_latency_ms for m in measurements]),
        total_estimated_cost_usd=sum(costs) if costs else None,
        correct_count=sum(1 for m in successes if m.evaluation_status == STATUS_CORRECT),
    )


@dataclass
class RealtimeComparison:
    throughput_speedup_ratio: Optional[float]
    batch_time_reduction_percent: Optional[float]
    median_service_latency_delta_percent: Optional[float]
    success_rate_delta: Optional[float]


def _percent_delta(baseline: Optional[float], compared: Optional[float]) -> Optional[float]:
    if baseline is None or compared is None or baseline == 0:
        return None
    return ((compared - baseline) / baseline) * 100


def compare_realtime_conditions(sequential: RealtimeConditionSummary, concurrent: RealtimeConditionSummary) -> RealtimeComparison:
    """H1/H2: concurrent execution CAN reduce wall-clock time, but higher
    concurrency does not necessarily reduce PER-REQUEST latency -- both
    numbers are reported side by side, never collapsed into one verdict."""
    throughput_speedup = (
        concurrent.successful_requests_per_second / sequential.successful_requests_per_second
        if sequential.successful_requests_per_second and concurrent.successful_requests_per_second
        else None
    )
    batch_time_reduction = (
        ((sequential.batch_wall_clock_ms - concurrent.batch_wall_clock_ms) / sequential.batch_wall_clock_ms) * 100
        if sequential.batch_wall_clock_ms and concurrent.batch_wall_clock_ms
        else None
    )
    success_rate_delta = (
        concurrent.success_rate - sequential.success_rate
        if sequential.success_rate is not None and concurrent.success_rate is not None
        else None
    )
    return RealtimeComparison(
        throughput_speedup_ratio=throughput_speedup,
        batch_time_reduction_percent=batch_time_reduction,
        median_service_latency_delta_percent=_percent_delta(
            sequential.request_service_latency_ms.median, concurrent.request_service_latency_ms.median
        ),
        success_rate_delta=success_rate_delta,
    )


@dataclass
class QualitySummary:
    condition: str
    total_evaluated: int
    correct_count: int
    accuracy: Optional[float]


def summarize_quality_realtime(condition: str, measurements: Sequence[RealtimeMeasurement]) -> QualitySummary:
    successes = [m for m in measurements if m.request_status == STATUS_SUCCESS]
    correct = sum(1 for m in successes if m.answer_correct is True)
    return QualitySummary(
        condition=condition, total_evaluated=len(successes), correct_count=correct,
        accuracy=(correct / len(successes)) if successes else None,
    )


def summarize_quality_batch(measurements: Sequence[BatchRequestMeasurement]) -> QualitySummary:
    completed = [m for m in measurements if m.reconciliation_status == RECON_COMPLETED]
    correct = sum(1 for m in completed if m.answer_correct is True)
    return QualitySummary(
        condition="batch", total_evaluated=len(completed), correct_count=correct,
        accuracy=(correct / len(completed)) if completed else None,
    )
