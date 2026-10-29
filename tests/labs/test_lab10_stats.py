from __future__ import annotations

from labs.lab10_conversation_summarization.evaluation import STATUS_CORRECT, STATUS_INCORRECT, STATUS_INVALID_FORMAT
from labs.lab10_conversation_summarization.models import FollowupMeasurement, SummaryMeasurement
from labs.lab10_conversation_summarization.stats import (
    BREAK_EVEN_NOT_REACHED,
    BREAK_EVEN_REACHED,
    compute_break_even,
    compute_context_comparison,
    compute_workflow_cost,
    context_input_reduction_percent,
    context_input_reduction_tokens,
    cost_per_correct_answer,
    net_savings_percent,
    net_savings_usd,
    request_compression_ratio,
    summarize_followups,
)


def _followup(status=STATUS_CORRECT, request_status="SUCCESS", prompt_tokens=100, completion_tokens=10, reasoning_tokens=2, cost=0.001) -> FollowupMeasurement:
    return FollowupMeasurement(
        experiment_run_id="run-1", execution_index=1, dataset_id="customer_support_001", condition_id="full_history_baseline",
        question_id="q1", timestamp="t", model="m", reasoning_effort="low", request_status=request_status,
        followup_prompt_tokens=prompt_tokens, followup_completion_tokens=completion_tokens, followup_reasoning_tokens=reasoning_tokens,
        followup_estimated_cost_usd=cost, evaluation_status=status, answer_correct=(status == STATUS_CORRECT),
    )


def _summary(cost=0.01, prompt_tokens=1800) -> SummaryMeasurement:
    return SummaryMeasurement(
        experiment_run_id="run-1", execution_index=1, dataset_id="customer_support_001", condition_id="summarization_low",
        reasoning_effort="low", timestamp="t", model="m", request_status="SUCCESS",
        summary_generation_estimated_cost_usd=cost, summary_generation_prompt_tokens=prompt_tokens,
    )


# -- Pure ratio/percentage guards (Part 9.3/9.4/10.4/11.6) -----------------


def test_net_savings_positive_means_summarization_saved_money():
    assert net_savings_usd(1.00, 0.60) == 0.40


def test_net_savings_negative_means_summarization_cost_more():
    assert net_savings_usd(0.60, 1.00) == -0.40


def test_net_savings_none_when_either_cost_missing():
    assert net_savings_usd(None, 0.5) is None
    assert net_savings_usd(0.5, None) is None


def test_net_savings_percent_guards_zero_denominator():
    assert net_savings_percent(0.0, 0.5) is None


def test_net_savings_percent_computed_correctly():
    assert net_savings_percent(1.00, 0.75) == 25.0


def test_context_input_reduction_tokens_and_percent():
    assert context_input_reduction_tokens(4000, 500) == 3500
    assert context_input_reduction_percent(4000, 500) == 87.5


def test_context_input_reduction_percent_guards_zero_and_none():
    assert context_input_reduction_percent(0, 500) is None
    assert context_input_reduction_percent(None, 500) is None
    assert context_input_reduction_tokens(None, 500) is None


def test_request_compression_ratio():
    assert request_compression_ratio(4000, 500) == 8.0


def test_request_compression_ratio_guards_zero_and_none_denominator():
    assert request_compression_ratio(4000, 0) is None
    assert request_compression_ratio(4000, None) is None


def test_cost_per_correct_answer_guards_zero_correct_answers():
    assert cost_per_correct_answer(1.00, 0) is None


def test_cost_per_correct_answer_computed():
    assert cost_per_correct_answer(1.00, 4) == 0.25


def test_cost_per_correct_answer_guards_missing_cost():
    assert cost_per_correct_answer(None, 4) is None


# -- Followup aggregation ----------------------------------------------------


def test_summarize_followups_counts_statuses_separately():
    followups = [
        _followup(status=STATUS_CORRECT), _followup(status=STATUS_INCORRECT),
        _followup(status=STATUS_INVALID_FORMAT),
    ]
    stats = summarize_followups(followups)
    assert stats.evaluated_answers == 3
    assert stats.correct_answers == 1
    assert stats.invalid_format_count == 1
    assert stats.answer_accuracy == 1 / 3


def test_summarize_followups_excludes_failed_requests_from_evaluated():
    followups = [_followup(status=STATUS_CORRECT), _followup(request_status="UNKNOWN_ERROR", status="EVALUATION_UNAVAILABLE")]
    stats = summarize_followups(followups)
    assert stats.evaluated_answers == 1
    assert stats.correct_answers == 1


def test_summarize_followups_empty_list_returns_none_accuracy_not_zero():
    stats = summarize_followups([])
    assert stats.answer_accuracy is None
    assert stats.evaluated_answers == 0


# -- Workflow cost: summary-generation cost counted exactly once ------------


def test_compute_workflow_cost_counts_summary_generation_cost_once():
    followups = [_followup(cost=0.001), _followup(cost=0.001), _followup(cost=0.001), _followup(cost=0.001)]
    summary = _summary(cost=0.01)
    workflow = compute_workflow_cost("customer_support_001", "summarization_low", summary, followups)
    assert workflow.summary_generation_cost_usd == 0.01
    assert workflow.followup_cost_usd == 0.004
    assert workflow.total_workflow_cost_usd == 0.014


def test_compute_workflow_cost_baseline_has_no_summary_cost():
    followups = [_followup(cost=0.001)]
    workflow = compute_workflow_cost("customer_support_001", "full_history_baseline", None, followups)
    assert workflow.summary_generation_cost_usd is None
    assert workflow.total_workflow_cost_usd == 0.001


def test_compute_workflow_cost_missing_summary_cost_makes_total_none():
    summary = _summary(cost=None)
    followups = [_followup(cost=0.001)]
    workflow = compute_workflow_cost("customer_support_001", "summarization_low", summary, followups)
    assert workflow.total_workflow_cost_usd is None


# -- Context comparison ------------------------------------------------------


def test_compute_context_comparison_full_pipeline():
    row = compute_context_comparison("customer_support_001", "q1", 4000, 500, 600)
    assert row.low_context_reduction_tokens == 3500
    assert row.low_context_reduction_percent == 87.5
    assert row.low_compression_ratio == 8.0
    assert row.medium_context_reduction_tokens == 3400
    assert row.medium_compression_ratio == 4000 / 600


def test_compute_context_comparison_missing_baseline_tokens():
    row = compute_context_comparison("customer_support_001", "q1", None, 500, 600)
    assert row.low_context_reduction_percent is None
    assert row.low_compression_ratio is None


# -- Break-even (Part 10.6/19.6 -- synthetic sequences with known results) --


def test_break_even_reached_after_first_followup():
    result = compute_break_even(0.05, [0.10, 0.10, 0.10], [0.01, 0.01, 0.01])
    assert result.break_even_call_index == 1
    assert result.break_even_status == BREAK_EVEN_REACHED


def test_break_even_reached_after_third_followup():
    # Summary costs more up front; breaks even only once enough baseline
    # follow-ups have accumulated past the summary's one-time overhead.
    result = compute_break_even(0.22, [0.10, 0.10, 0.10, 0.10], [0.02, 0.02, 0.02, 0.02])
    assert result.break_even_call_index == 3
    assert result.break_even_status == BREAK_EVEN_REACHED
    # Not reached at the first two follow-ups.
    assert result.baseline_cumulative_costs[0] < result.summary_cumulative_costs[0]
    assert result.baseline_cumulative_costs[1] < result.summary_cumulative_costs[1]


def test_break_even_not_reached_when_summary_always_costs_more():
    result = compute_break_even(0.20, [0.01, 0.01], [0.01, 0.01])
    assert result.break_even_status == BREAK_EVEN_NOT_REACHED
    assert result.break_even_call_index is None


def test_break_even_equal_cumulative_costs_counts_as_reached():
    # <=  (never strictly less-than) -- equal cumulative costs is a reached break-even.
    result = compute_break_even(0.0, [0.05], [0.05])
    assert result.break_even_status == BREAK_EVEN_REACHED
    assert result.break_even_call_index == 1


def test_break_even_missing_summary_generation_cost_is_entirely_unknown():
    result = compute_break_even(None, [0.05, 0.05], [0.01, 0.01])
    assert result.summary_cumulative_costs == (None, None)
    assert result.break_even_status == BREAK_EVEN_NOT_REACHED


def test_break_even_missing_followup_cost_makes_remaining_cumulative_none():
    result = compute_break_even(0.0, [0.05, 0.05, 0.05], [0.01, None, 0.01])
    assert result.summary_cumulative_costs == (0.01, None, None)


def test_break_even_summary_more_expensive_than_all_observed_savings():
    # Summary overhead alone exceeds the entire baseline's total spend.
    result = compute_break_even(1.00, [0.01, 0.01, 0.01], [0.0, 0.0, 0.0])
    assert result.break_even_status == BREAK_EVEN_NOT_REACHED
    assert result.break_even_call_index is None


def test_break_even_rejects_mismatched_sequence_lengths():
    import pytest

    with pytest.raises(ValueError):
        compute_break_even(0.0, [0.01, 0.01], [0.01])
