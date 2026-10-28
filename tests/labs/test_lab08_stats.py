from __future__ import annotations

from labs.lab08_concurrency.evaluation import STATUS_CORRECT, STATUS_INCORRECT
from labs.lab08_concurrency.models import (
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_STARTED,
    STATUS_CONNECTION_ERROR,
    STATUS_SUCCESS,
    ConcurrencyMeasurement,
    RequestTimelineEvent,
)
from labs.lab08_concurrency.stats import (
    compare_to_baseline,
    summarize_all_conditions,
    summarize_condition,
    time_weighted_average_in_flight,
)


def _measurement(task_id, max_concurrent, status=STATUS_SUCCESS, queue_wait=5.0, service=100.0, cost=0.001, correct=True, execution_mode="async_concurrent"):
    return ConcurrencyMeasurement(
        experiment_run_id="run-1",
        execution_index=1,
        repetition_index=1,
        task_id=task_id,
        max_concurrent=max_concurrent,
        execution_mode=execution_mode,
        timestamp="2026-01-01T00:00:00+00:00",
        model="m",
        reasoning_effort="low",
        max_completion_tokens=100,
        request_status=status,
        queue_wait_ms=queue_wait if status == STATUS_SUCCESS else queue_wait,
        request_service_latency_ms=service if status == STATUS_SUCCESS else None,
        task_end_to_end_latency_ms=(queue_wait + service) if status == STATUS_SUCCESS else queue_wait,
        estimated_cost_usd=cost if status == STATUS_SUCCESS else None,
        evaluation_status=STATUS_CORRECT if correct else STATUS_INCORRECT,
    )


def test_time_weighted_average_in_flight_for_fully_overlapping_pair():
    # Two requests start at t=0 and both run for the full 100ms span: average in-flight = 2.
    events = [
        RequestTimelineEvent(1, 1, 2, "t1", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(2, 1, 2, "t2", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(1, 1, 2, "t1", EVENT_REQUEST_COMPLETED, 100.0),
        RequestTimelineEvent(2, 1, 2, "t2", EVENT_REQUEST_COMPLETED, 100.0),
    ]
    assert time_weighted_average_in_flight(events, 100.0) == 2.0


def test_time_weighted_average_in_flight_for_sequential_pair():
    # Two back-to-back non-overlapping requests, each half the batch: average in-flight = 1.
    events = [
        RequestTimelineEvent(1, 1, 1, "t1", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(1, 1, 1, "t1", EVENT_REQUEST_COMPLETED, 50.0),
        RequestTimelineEvent(2, 1, 1, "t2", EVENT_REQUEST_STARTED, 50.0),
        RequestTimelineEvent(2, 1, 1, "t2", EVENT_REQUEST_COMPLETED, 100.0),
    ]
    assert time_weighted_average_in_flight(events, 100.0) == 1.0


def test_time_weighted_average_in_flight_empty_events_returns_none():
    assert time_weighted_average_in_flight([], 100.0) is None
    assert time_weighted_average_in_flight([RequestTimelineEvent(1, 1, 1, "t1", EVENT_REQUEST_STARTED, 0.0)], 0.0) is None


def test_summarize_condition_counts_success_and_failure():
    measurements = [
        _measurement("t1", 2, status=STATUS_SUCCESS),
        _measurement("t2", 2, status=STATUS_CONNECTION_ERROR),
    ]
    events = [
        RequestTimelineEvent(1, 1, 2, "t1", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(1, 1, 2, "t1", EVENT_REQUEST_COMPLETED, 100.0),
    ]
    summary = summarize_condition(2, measurements, events, batch_wall_clock_ms=100.0, observed_peak_in_flight=2)
    assert summary.total_requests == 2
    assert summary.successful_requests == 1
    assert summary.failed_requests == 1
    assert summary.connection_error_requests == 1
    assert summary.success_rate == 0.5


def test_summarize_condition_utilization_divides_by_configured_concurrency():
    measurements = [_measurement("t1", 4), _measurement("t2", 4)]
    events = [
        RequestTimelineEvent(1, 1, 4, "t1", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(2, 1, 4, "t2", EVENT_REQUEST_STARTED, 0.0),
        RequestTimelineEvent(1, 1, 4, "t1", EVENT_REQUEST_COMPLETED, 100.0),
        RequestTimelineEvent(2, 1, 4, "t2", EVENT_REQUEST_COMPLETED, 100.0),
    ]
    summary = summarize_condition(4, measurements, events, batch_wall_clock_ms=100.0, observed_peak_in_flight=2)
    # time-weighted average in-flight is 2, configured concurrency is 4 -> utilization 0.5.
    assert summary.concurrency_utilization_ratio == 0.5
    # peak must never be assumed equal to configured concurrency.
    assert summary.observed_peak_in_flight_requests == 2
    assert summary.observed_peak_in_flight_requests != summary.max_concurrent


def test_summarize_condition_cost_per_successful_request_excludes_failures():
    measurements = [
        _measurement("t1", 1, status=STATUS_SUCCESS, cost=0.002),
        _measurement("t2", 1, status=STATUS_CONNECTION_ERROR),
    ]
    summary = summarize_condition(1, measurements, [], batch_wall_clock_ms=50.0, observed_peak_in_flight=1)
    assert summary.cost_per_successful_request_usd == 0.002
    assert summary.total_estimated_cost_usd == 0.002


def test_summarize_condition_zero_successful_requests_never_divides_by_zero():
    measurements = [_measurement("t1", 1, status=STATUS_CONNECTION_ERROR)]
    summary = summarize_condition(1, measurements, [], batch_wall_clock_ms=10.0, observed_peak_in_flight=1)
    assert summary.cost_per_successful_request_usd is None
    assert summary.total_estimated_cost_usd is None
    assert summary.success_rate == 0.0


def test_summarize_condition_correct_count_only_counts_successes():
    measurements = [
        _measurement("t1", 1, status=STATUS_SUCCESS, correct=True),
        _measurement("t2", 1, status=STATUS_SUCCESS, correct=False),
    ]
    summary = summarize_condition(1, measurements, [], batch_wall_clock_ms=10.0, observed_peak_in_flight=1)
    assert summary.correct_count == 1


def test_compare_to_baseline_throughput_speedup_and_latency_delta():
    baseline = summarize_condition(1, [_measurement("t1", 1, queue_wait=0.0, service=200.0)], [], batch_wall_clock_ms=200.0, observed_peak_in_flight=1)
    compared = summarize_condition(3, [_measurement("t1", 3, queue_wait=0.0, service=100.0)], [], batch_wall_clock_ms=100.0, observed_peak_in_flight=3)
    comparison = compare_to_baseline(baseline, compared)
    assert comparison.baseline_concurrency == 1
    assert comparison.compared_concurrency == 3
    assert comparison.batch_time_reduction_percent == 50.0
    # concurrent service latency is lower here, so the delta is negative.
    assert comparison.median_service_latency_delta_percent < 0


def test_compare_to_baseline_handles_missing_wall_clock_without_crashing():
    baseline = summarize_condition(1, [], [], batch_wall_clock_ms=None, observed_peak_in_flight=0)
    compared = summarize_condition(3, [], [], batch_wall_clock_ms=None, observed_peak_in_flight=0)
    comparison = compare_to_baseline(baseline, compared)
    assert comparison.batch_time_reduction_percent is None
    assert comparison.throughput_speedup_ratio is None


def test_summarize_all_conditions_merges_repetitions_for_same_level():
    plan_results = [
        (1, [_measurement("t1", 1)], [], 100.0, 1),
        (1, [_measurement("t2", 1)], [], 120.0, 1),
        (5, [_measurement("t3", 5)], [], 40.0, 5),
    ]
    summaries = summarize_all_conditions(plan_results)
    by_level = {s.max_concurrent: s for s in summaries}
    assert by_level[1].total_requests == 2
    assert by_level[1].batch_wall_clock_ms == 110.0  # mean of 100 and 120
    assert by_level[5].total_requests == 1


def test_summarize_all_conditions_returns_one_summary_per_distinct_level():
    plan_results = [
        (1, [_measurement("t1", 1)], [], 50.0, 1),
        (3, [_measurement("t2", 3)], [], 30.0, 2),
        (5, [_measurement("t3", 5)], [], 20.0, 4),
    ]
    summaries = summarize_all_conditions(plan_results)
    assert {s.max_concurrent for s in summaries} == {1, 3, 5}
