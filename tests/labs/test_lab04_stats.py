from __future__ import annotations

from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement
from labs.lab04_output_control.stats import (
    BOTH_STRATEGIES_ACCEPTED,
    CHEAPER_STRATEGY_ALSO_ACCEPTED,
    INSUFFICIENT_DATA,
    LOWER_COST_BUT_QUALITY_FAILED,
    QUALITY_FAILED_BOTH,
    compare_invocation_strategies,
    summarize_all_output_limit_conditions,
    summarize_invocation_strategy,
    summarize_output_limit_condition,
)


def _ol(condition_id="restrictive", succeeded=True, completion_tokens=40, output_truncated=False, status="accepted", accepted=True, cost=0.001):
    return OutputLimitMeasurement(
        experiment_run_id="run-1", condition_id=condition_id, execution_index=1, repetition_index=1,
        timestamp="t", model="openai/gpt-oss-20b", max_completion_tokens=64, reasoning_effort="low",
        succeeded=succeeded, attempts_used=1, completion_tokens=completion_tokens, reasoning_tokens=10,
        non_reasoning_completion_tokens=completion_tokens - 10 if completion_tokens else None,
        output_truncated=output_truncated, estimated_cost_usd=cost if succeeded else None,
        answer_evaluation_status=status, overall_answer_accepted=accepted,
        field_completeness_ratio=1.0 if accepted else 0.5, field_correctness_ratio=1.0 if accepted else 0.5,
    )


def test_truncated_calls_counted_only_when_output_truncated_is_true():
    measurements = [_ol(output_truncated=True), _ol(output_truncated=False), _ol(output_truncated=None)]
    summary = summarize_output_limit_condition("restrictive", 64, 3, measurements)
    assert summary.truncated_calls == 1
    assert summary.successful_calls == 3


def test_accepted_incomplete_missing_counts_partition_status():
    measurements = [
        _ol(status="accepted", accepted=True),
        _ol(status="incomplete", accepted=False),
        _ol(status="missing_response", accepted=False),
    ]
    summary = summarize_output_limit_condition("restrictive", 64, 3, measurements)
    assert summary.accepted_count == 1
    assert summary.incomplete_count == 1
    assert summary.missing_response_count == 1


def test_failed_call_is_counted_as_missing_response_even_without_an_explicit_status_override():
    failed = OutputLimitMeasurement(
        experiment_run_id="run-1", condition_id="restrictive", execution_index=1, repetition_index=1,
        timestamp="t", model="m", max_completion_tokens=64, reasoning_effort="low",
        succeeded=False, attempts_used=2, error="boom",
    )
    summary = summarize_output_limit_condition("restrictive", 64, 1, [failed])
    assert summary.failed_calls == 1
    assert summary.missing_response_count == 1


def test_summarize_all_output_limit_conditions_groups_by_condition_id():
    measurements = [_ol("restrictive"), _ol("restrictive"), _ol("moderate")]
    summaries = summarize_all_output_limit_conditions(
        [("restrictive", 64), ("moderate", 256)], {"restrictive": 2, "moderate": 1}, measurements
    )
    by_id = {s.condition_id: s for s in summaries}
    assert by_id["restrictive"].successful_calls == 2
    assert by_id["moderate"].successful_calls == 1


def _inv(succeeded=True, accepted=True, status="accepted", cost=0.001, latency=100.0):
    return InvocationMeasurement(
        experiment_run_id="run-1", strategy="single", execution_index=1, question_index=None,
        timestamp="t", model="m", max_completion_tokens=2048, reasoning_effort="low",
        succeeded=succeeded, attempts_used=1, prompt_tokens=50, completion_tokens=20, reasoning_tokens=5,
        estimated_cost_usd=cost if succeeded else None, client_elapsed_ms=latency,
        answer_evaluation_status=status, overall_answer_accepted=accepted,
    )


def test_all_answers_accepted_requires_every_measurement_accepted():
    summary = summarize_invocation_strategy("single", [_inv(accepted=True), _inv(accepted=True)], 200.0)
    assert summary.all_answers_accepted is True

    summary_partial = summarize_invocation_strategy("single", [_inv(accepted=True), _inv(accepted=False, status="incomplete")], 200.0)
    assert summary_partial.all_answers_accepted is False
    assert summary_partial.invalid_answers == 1


def test_missing_answers_counts_failed_calls():
    summary = summarize_invocation_strategy(
        "multiple", [_inv(succeeded=False, accepted=False, status="missing_response")], None
    )
    assert summary.missing_answers == 1
    assert summary.all_answers_accepted is False


def test_both_strategies_accepted_when_both_fully_accept():
    single = summarize_invocation_strategy("single", [_inv(cost=0.001)], 100.0)
    multiple = summarize_invocation_strategy("multiple", [_inv(cost=0.0004), _inv(cost=0.0004), _inv(cost=0.0004)], 300.0)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.quality_comparison == BOTH_STRATEGIES_ACCEPTED


def test_quality_failed_both_when_neither_strategy_fully_accepts():
    single = summarize_invocation_strategy("single", [_inv(accepted=False, status="incomplete")], 100.0)
    multiple = summarize_invocation_strategy("multiple", [_inv(accepted=False, status="incomplete")], 300.0)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.quality_comparison == QUALITY_FAILED_BOTH


def test_lower_cost_but_quality_failed_when_the_cheaper_strategy_is_the_one_that_failed():
    # single is cheaper (0.001 < 0.003) but fails quality -- multiple (more
    # expensive) is the one that's fully accepted.
    single = summarize_invocation_strategy("single", [_inv(cost=0.001, accepted=False, status="incomplete")], 100.0)
    multiple = summarize_invocation_strategy("multiple", [_inv(cost=0.001, accepted=True), _inv(cost=0.001, accepted=True), _inv(cost=0.001, accepted=True)], 300.0)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.quality_comparison == LOWER_COST_BUT_QUALITY_FAILED


def test_cheaper_strategy_also_accepted_when_the_accepted_one_is_also_cheaper():
    single = summarize_invocation_strategy("single", [_inv(cost=0.001, accepted=True)], 100.0)
    multiple = summarize_invocation_strategy(
        "multiple", [_inv(cost=0.002, accepted=False, status="incomplete")], 300.0
    )
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.quality_comparison == CHEAPER_STRATEGY_ALSO_ACCEPTED


def test_insufficient_data_when_cost_is_unavailable_and_quality_split():
    single = summarize_invocation_strategy("single", [_inv(succeeded=False, accepted=False, status="missing_response")], 100.0)
    multiple = summarize_invocation_strategy("multiple", [_inv(accepted=True)], 300.0)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.quality_comparison == INSUFFICIENT_DATA


def test_cost_difference_is_multiple_minus_single():
    single = summarize_invocation_strategy("single", [_inv(cost=0.001)], 100.0)
    multiple = summarize_invocation_strategy("multiple", [_inv(cost=0.0005), _inv(cost=0.0005)], 300.0)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.absolute_cost_difference_usd is not None
    assert abs(comparison.absolute_cost_difference_usd - 0.0) < 1e-9


def test_relative_cost_difference_is_none_when_single_cost_is_zero():
    single = summarize_invocation_strategy("single", [], None)
    multiple = summarize_invocation_strategy("multiple", [_inv(cost=0.001)], None)
    comparison = compare_invocation_strategies(single, multiple)
    assert comparison.absolute_cost_difference_usd is None  # single has no cost data at all (n=0)
