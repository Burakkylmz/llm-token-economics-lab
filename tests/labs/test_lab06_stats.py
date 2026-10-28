from __future__ import annotations

from labs.lab06_streaming.models import CONDITION_NON_STREAMING, CONDITION_STREAMING, StreamMeasurement
from labs.lab06_streaming.stats import (
    BOTH_COMPLETE,
    BOTH_INCOMPLETE,
    INSUFFICIENT_DATA,
    NON_STREAMING_MORE_COMPLETE,
    STREAMING_MORE_COMPLETE,
    compute_pair_comparisons,
    summarize_all_conditions,
    summarize_condition,
)


def _streaming(pair_id=1, succeeded=True, first_visible=500.0, total=800.0, status="COMPLETE", complete=True, cost=0.001):
    return StreamMeasurement(
        experiment_run_id="run-1", execution_index=1, pair_id=pair_id, condition=CONDITION_STREAMING,
        timestamp="t", model="m", reasoning_effort="low", max_completion_tokens=100, request_hash="h",
        succeeded=succeeded, attempts_used=1, first_stream_event_ms=200.0 if succeeded else None,
        first_visible_content_ms=first_visible if succeeded else None,
        total_stream_latency_ms=total if succeeded else None,
        evaluation_status=status, answer_complete=complete if succeeded else None,
        estimated_cost_usd=cost if succeeded else None,
    )


def _non_streaming(pair_id=1, succeeded=True, total=1000.0, status="COMPLETE", complete=True, cost=0.001):
    return StreamMeasurement(
        experiment_run_id="run-1", execution_index=2, pair_id=pair_id, condition=CONDITION_NON_STREAMING,
        timestamp="t", model="m", reasoning_effort="low", max_completion_tokens=100, request_hash="h",
        succeeded=succeeded, attempts_used=1, non_streaming_total_latency_ms=total if succeeded else None,
        evaluation_status=status, answer_complete=complete if succeeded else None,
        estimated_cost_usd=cost if succeeded else None,
    )


def test_summarize_condition_counts_statuses():
    measurements = [_streaming(status="COMPLETE", complete=True), _streaming(status="INCOMPLETE", complete=False)]
    summary = summarize_condition("streaming", measurements)
    assert summary.completed_count == 1
    assert summary.incomplete_count == 1
    assert summary.successful_calls == 2


def test_summarize_condition_distinguishes_total_latency_field_by_condition():
    s_summary = summarize_condition("streaming", [_streaming(total=800.0)])
    n_summary = summarize_condition("non_streaming", [_non_streaming(total=1000.0)])
    assert s_summary.total_latency_ms.mean == 800.0
    assert n_summary.total_latency_ms.mean == 1000.0


def test_failed_call_counted_in_total_but_not_successful():
    measurements = [_streaming(succeeded=True), _streaming(succeeded=False, status="API_ERROR", complete=None)]
    summary = summarize_condition("streaming", measurements)
    assert summary.total_calls == 2
    assert summary.successful_calls == 1
    assert summary.api_error_count == 1


def test_summarize_all_conditions_groups_correctly():
    measurements = [_streaming(), _non_streaming()]
    summaries = summarize_all_conditions(["streaming", "non_streaming"], measurements)
    assert {s.condition for s in summaries} == {"streaming", "non_streaming"}


def test_pair_comparison_gain_is_non_streaming_total_minus_streaming_first_visible():
    measurements = [_streaming(pair_id=1, first_visible=500.0), _non_streaming(pair_id=1, total=1000.0)]
    comparisons = compute_pair_comparisons(measurements)
    assert len(comparisons) == 1
    assert comparisons[0].perceived_latency_gain_ms == 500.0


def test_pair_comparison_percent_denominator_is_non_streaming_total():
    measurements = [_streaming(pair_id=1, first_visible=250.0), _non_streaming(pair_id=1, total=1000.0)]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].perceived_latency_gain_percent == 75.0


def test_negative_perceived_latency_gain_is_preserved_not_clamped():
    measurements = [_streaming(pair_id=1, first_visible=1200.0), _non_streaming(pair_id=1, total=1000.0)]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].perceived_latency_gain_ms == -200.0


def test_total_latency_difference_is_non_streaming_minus_streaming():
    measurements = [_streaming(pair_id=1, total=800.0), _non_streaming(pair_id=1, total=1000.0)]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].total_latency_difference_ms == 200.0


def test_pair_missing_one_condition_is_skipped_not_fabricated():
    measurements = [_streaming(pair_id=1)]  # no non_streaming measurement for pair 1
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons == []


def test_both_complete_quality_comparison():
    measurements = [_streaming(pair_id=1, complete=True), _non_streaming(pair_id=1, complete=True)]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].quality_comparison == BOTH_COMPLETE


def test_streaming_more_complete_quality_comparison():
    measurements = [_streaming(pair_id=1, complete=True), _non_streaming(pair_id=1, complete=False, status="INCOMPLETE")]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].quality_comparison == STREAMING_MORE_COMPLETE


def test_non_streaming_more_complete_quality_comparison():
    measurements = [_streaming(pair_id=1, complete=False, status="INCOMPLETE"), _non_streaming(pair_id=1, complete=True)]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].quality_comparison == NON_STREAMING_MORE_COMPLETE


def test_both_incomplete_quality_comparison():
    measurements = [
        _streaming(pair_id=1, complete=False, status="INCOMPLETE"),
        _non_streaming(pair_id=1, complete=False, status="INCOMPLETE"),
    ]
    comparisons = compute_pair_comparisons(measurements)
    assert comparisons[0].quality_comparison == BOTH_INCOMPLETE


def test_insufficient_data_when_completeness_is_none():
    measurements = [
        _streaming(pair_id=1, succeeded=False, status="API_ERROR", complete=None),
        _non_streaming(pair_id=1, complete=True),
    ]
    comparisons = compute_pair_comparisons(measurements)
    # streaming failed -> no streaming measurement in by_pair at all -> pair skipped entirely
    assert comparisons == []


def test_multiple_pairs_sorted_by_pair_id():
    measurements = [
        _streaming(pair_id=2), _non_streaming(pair_id=2),
        _streaming(pair_id=1), _non_streaming(pair_id=1),
    ]
    comparisons = compute_pair_comparisons(measurements)
    assert [c.pair_id for c in comparisons] == [1, 2]
