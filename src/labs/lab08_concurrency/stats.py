"""Descriptive statistics, per-concurrency-level summaries, and
baseline comparisons for Lab08.

`DescriptiveStats`/`compute_descriptive_stats` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats.

`concurrency_utilization_ratio` is a genuine TIME-WEIGHTED average of the
in-flight request count over the batch's wall-clock span (reconstructed
from the REQUEST_STARTED/REQUEST_COMPLETED timeline events), divided by
the configured `max_concurrent` -- not merely
`observed_peak_in_flight_requests / max_concurrent`, which would only
capture the single busiest instant, not how fully the configured capacity
was used on average.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
)
from labs.lab08_concurrency.evaluation import STATUS_CORRECT
from labs.lab08_concurrency.models import (
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_STARTED,
    STATUS_CANCELLED,
    STATUS_CONNECTION_ERROR,
    STATUS_PROVIDER_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_UNKNOWN_ERROR,
    ConcurrencyMeasurement,
    RequestTimelineEvent,
)


@dataclass
class ConditionSummary:
    max_concurrent: int
    execution_mode: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    rate_limited_requests: int
    timed_out_requests: int
    connection_error_requests: int
    provider_error_requests: int
    cancelled_requests: int
    unknown_error_requests: int
    success_rate: Optional[float]

    observed_peak_in_flight_requests: int
    concurrency_utilization_ratio: Optional[float]

    batch_wall_clock_ms: Optional[float]
    successful_requests_per_second: Optional[float]
    completed_requests_per_second: Optional[float]

    queue_wait_ms: DescriptiveStats
    request_service_latency_ms: DescriptiveStats
    task_end_to_end_latency_ms: DescriptiveStats

    total_estimated_cost_usd: Optional[float]
    mean_cost_per_request_usd: Optional[float]
    cost_per_successful_request_usd: Optional[float]

    correct_count: int


def time_weighted_average_in_flight(events: Sequence[RequestTimelineEvent], batch_wall_clock_ms: float) -> Optional[float]:
    if not events or not batch_wall_clock_ms:
        return None
    deltas = []
    for e in events:
        if e.event_type == EVENT_REQUEST_STARTED:
            deltas.append((e.relative_ms, 1))
        elif e.event_type == EVENT_REQUEST_COMPLETED:
            deltas.append((e.relative_ms, -1))
    if not deltas:
        return None
    deltas.sort(key=lambda pair: pair[0])

    area = 0.0
    current = 0
    last_t = 0.0
    for t, delta in deltas:
        area += current * (t - last_t)
        current += delta
        last_t = t
    area += current * (batch_wall_clock_ms - last_t)
    return area / batch_wall_clock_ms


def summarize_condition(
    max_concurrent: int,
    measurements: Sequence[ConcurrencyMeasurement],
    events: Sequence[RequestTimelineEvent],
    batch_wall_clock_ms: Optional[float],
    observed_peak_in_flight: int,
) -> ConditionSummary:
    total = len(measurements)
    successes = [m for m in measurements if m.request_status == STATUS_SUCCESS]
    status_counts = {status: sum(1 for m in measurements if m.request_status == status) for status in (
        STATUS_RATE_LIMITED, STATUS_TIMEOUT, STATUS_CONNECTION_ERROR, STATUS_PROVIDER_ERROR, STATUS_CANCELLED, STATUS_UNKNOWN_ERROR,
    )}

    costs = [m.estimated_cost_usd for m in measurements if m.estimated_cost_usd is not None]
    successful_costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]

    duration_s = (batch_wall_clock_ms / 1000) if batch_wall_clock_ms else None
    successful_rps = (len(successes) / duration_s) if duration_s else None
    completed_rps = (total / duration_s) if duration_s else None

    weighted_avg = time_weighted_average_in_flight(events, batch_wall_clock_ms) if batch_wall_clock_ms else None
    utilization = (weighted_avg / max_concurrent) if weighted_avg is not None and max_concurrent else None

    execution_mode = measurements[0].execution_mode if measurements else ("async_sequential" if max_concurrent == 1 else "async_concurrent")

    return ConditionSummary(
        max_concurrent=max_concurrent,
        execution_mode=execution_mode,
        total_requests=total,
        successful_requests=len(successes),
        failed_requests=total - len(successes),
        rate_limited_requests=status_counts[STATUS_RATE_LIMITED],
        timed_out_requests=status_counts[STATUS_TIMEOUT],
        connection_error_requests=status_counts[STATUS_CONNECTION_ERROR],
        provider_error_requests=status_counts[STATUS_PROVIDER_ERROR],
        cancelled_requests=status_counts[STATUS_CANCELLED],
        unknown_error_requests=status_counts[STATUS_UNKNOWN_ERROR],
        success_rate=(len(successes) / total) if total else None,
        observed_peak_in_flight_requests=observed_peak_in_flight,
        concurrency_utilization_ratio=utilization,
        batch_wall_clock_ms=batch_wall_clock_ms,
        successful_requests_per_second=successful_rps,
        completed_requests_per_second=completed_rps,
        queue_wait_ms=compute_descriptive_stats([m.queue_wait_ms for m in measurements]),
        request_service_latency_ms=compute_descriptive_stats([m.request_service_latency_ms for m in measurements]),
        task_end_to_end_latency_ms=compute_descriptive_stats([m.task_end_to_end_latency_ms for m in measurements]),
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_cost_per_request_usd=(sum(costs) / len(costs)) if costs else None,
        cost_per_successful_request_usd=(sum(successful_costs) / len(successes)) if successes and successful_costs else None,
        correct_count=sum(1 for m in successes if m.evaluation_status == STATUS_CORRECT),
    )


@dataclass
class ConcurrencyComparison:
    baseline_concurrency: int
    compared_concurrency: int
    throughput_speedup_ratio: Optional[float]
    batch_time_reduction_percent: Optional[float]
    median_service_latency_delta_percent: Optional[float]
    median_task_end_to_end_latency_delta_percent: Optional[float]
    success_rate_delta: Optional[float]
    cost_delta_percent: Optional[float]


def _percent_delta(baseline: Optional[float], compared: Optional[float]) -> Optional[float]:
    if baseline is None or compared is None or baseline == 0:
        return None
    return ((compared - baseline) / baseline) * 100


def compare_to_baseline(baseline: ConditionSummary, compared: ConditionSummary) -> ConcurrencyComparison:
    throughput_speedup = (
        compared.successful_requests_per_second / baseline.successful_requests_per_second
        if baseline.successful_requests_per_second and compared.successful_requests_per_second
        else None
    )
    batch_time_reduction = (
        ((baseline.batch_wall_clock_ms - compared.batch_wall_clock_ms) / baseline.batch_wall_clock_ms) * 100
        if baseline.batch_wall_clock_ms and compared.batch_wall_clock_ms
        else None
    )
    success_rate_delta = (
        compared.success_rate - baseline.success_rate
        if baseline.success_rate is not None and compared.success_rate is not None
        else None
    )
    return ConcurrencyComparison(
        baseline_concurrency=baseline.max_concurrent,
        compared_concurrency=compared.max_concurrent,
        throughput_speedup_ratio=throughput_speedup,
        batch_time_reduction_percent=batch_time_reduction,
        median_service_latency_delta_percent=_percent_delta(
            baseline.request_service_latency_ms.median, compared.request_service_latency_ms.median
        ),
        median_task_end_to_end_latency_delta_percent=_percent_delta(
            baseline.task_end_to_end_latency_ms.median, compared.task_end_to_end_latency_ms.median
        ),
        success_rate_delta=success_rate_delta,
        cost_delta_percent=_percent_delta(baseline.total_estimated_cost_usd, compared.total_estimated_cost_usd),
    )


def summarize_all_conditions(
    plan_results: Sequence[tuple[int, Sequence[ConcurrencyMeasurement], Sequence[RequestTimelineEvent], Optional[float], int]],
) -> list[ConditionSummary]:
    """`plan_results` holds (max_concurrent, measurements, events,
    batch_wall_clock_ms, observed_peak_in_flight) per executed batch --
    when repetitions > 1, batches for the SAME max_concurrent are merged
    here (their measurements/events concatenated, wall-clock/peak
    aggregated as a mean/max respectively) so one ConditionSummary per
    concurrency level is always returned, regardless of repetitions."""
    grouped: dict[int, list[tuple]] = defaultdict(list)
    for level, measurements, events, wall_clock_ms, peak in plan_results:
        grouped[level].append((measurements, events, wall_clock_ms, peak))

    summaries = []
    for level, batches in grouped.items():
        all_measurements = [m for measurements, _, _, _ in batches for m in measurements]
        all_events = [e for _, events, _, _ in batches for e in events]
        wall_clocks = [w for _, _, w, _ in batches if w is not None]
        mean_wall_clock = (sum(wall_clocks) / len(wall_clocks)) if wall_clocks else None
        peak = max((p for _, _, _, p in batches), default=0)
        summaries.append(summarize_condition(level, all_measurements, all_events, mean_wall_clock, peak))
    return summaries
