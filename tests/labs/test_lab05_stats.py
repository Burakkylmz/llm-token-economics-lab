from __future__ import annotations

from labs.lab05_model_routing.models import RoutingMeasurement
from labs.lab05_model_routing.stats import (
    BOTH_FAILED,
    INSUFFICIENT_DATA,
    QUALITY_PRESERVED,
    ROUTING_QUALITY_IMPROVEMENT,
    ROUTING_QUALITY_REGRESSION,
    compare_strategy_cost,
    compute_task_comparisons,
    summarize_all_strategies,
    summarize_strategy,
)


def _m(
    strategy="always_small", task_id="task_001", succeeded=True, status="CORRECT", correct=True,
    model_tier="small", cost=0.001, latency=100.0, matched_rules=None,
):
    return RoutingMeasurement(
        experiment_run_id="run-1", execution_index=1, strategy=strategy, task_id=task_id,
        task_category="factual_retrieval", complexity_label="simple", timestamp="t",
        selected_model="m", model_tier=model_tier, routing_reason="r", matched_rules=matched_rules or [],
        succeeded=succeeded, attempts_used=1, estimated_cost_usd=cost if succeeded else None,
        total_latency_ms=latency, evaluation_status=status, answer_correct=correct if succeeded else None,
    )


def test_accuracy_computed_from_correct_status_only():
    measurements = [_m(status="CORRECT", correct=True), _m(status="INCORRECT", correct=False), _m(status="CORRECT", correct=True)]
    summary = summarize_strategy("always_small", measurements)
    assert summary.correct_tasks == 2
    assert summary.total_tasks == 3
    assert summary.accuracy == 2 / 3


def test_small_and_large_model_selection_counted_separately():
    measurements = [_m(model_tier="small"), _m(model_tier="small"), _m(model_tier="large")]
    summary = summarize_strategy("rule_based_routing", measurements)
    assert summary.small_model_calls == 2
    assert summary.large_model_calls == 1
    assert summary.small_model_selection_ratio == 2 / 3
    assert summary.large_model_selection_ratio == 1 / 3


def test_routing_rule_distribution_counts_each_matched_rule():
    measurements = [
        _m(matched_rules=["multi_step_reasoning_indicators"]),
        _m(matched_rules=["multi_step_reasoning_indicators"]),
        _m(matched_rules=["code_generation_indicator"]),
    ]
    summary = summarize_strategy("rule_based_routing", measurements)
    assert summary.routing_rule_distribution == {"multi_step_reasoning_indicators": 2, "code_generation_indicator": 1}


def test_failed_call_counted_in_total_but_not_correct():
    measurements = [_m(succeeded=True, status="CORRECT", correct=True), _m(succeeded=False, status="EVALUATION_UNAVAILABLE", correct=None)]
    summary = summarize_strategy("always_small", measurements)
    assert summary.total_tasks == 2
    assert summary.correct_tasks == 1


def test_summarize_all_strategies_groups_correctly():
    measurements = [_m(strategy="always_small"), _m(strategy="rule_based_routing")]
    summaries = summarize_all_strategies(["always_small", "rule_based_routing"], measurements)
    assert {s.strategy for s in summaries} == {"always_small", "rule_based_routing"}


def test_quality_preserved_when_both_correct():
    measurements = [
        _m(strategy="always_small", task_id="task_001", correct=True),
        _m(strategy="rule_based_routing", task_id="task_001", correct=True),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert comparisons[0].quality_outcome == QUALITY_PRESERVED


def test_routing_quality_regression_when_baseline_correct_routed_wrong():
    measurements = [
        _m(strategy="always_small", task_id="task_001", correct=True),
        _m(strategy="rule_based_routing", task_id="task_001", correct=False, status="INCORRECT"),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert comparisons[0].quality_outcome == ROUTING_QUALITY_REGRESSION


def test_routing_quality_improvement_when_baseline_wrong_routed_correct():
    measurements = [
        _m(strategy="always_small", task_id="task_001", correct=False, status="INCORRECT"),
        _m(strategy="rule_based_routing", task_id="task_001", correct=True),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert comparisons[0].quality_outcome == ROUTING_QUALITY_IMPROVEMENT


def test_both_failed_when_both_wrong():
    measurements = [
        _m(strategy="always_small", task_id="task_001", correct=False, status="INCORRECT"),
        _m(strategy="rule_based_routing", task_id="task_001", correct=False, status="INCORRECT"),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert comparisons[0].quality_outcome == BOTH_FAILED


def test_insufficient_data_when_either_side_correctness_is_none():
    measurements = [
        _m(strategy="always_small", task_id="task_001", succeeded=False, correct=None, status="EVALUATION_UNAVAILABLE"),
        _m(strategy="rule_based_routing", task_id="task_001", correct=True),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert comparisons[0].quality_outcome == INSUFFICIENT_DATA


def test_task_missing_one_side_is_skipped_not_fabricated():
    measurements = [_m(strategy="always_small", task_id="task_001")]  # no router measurement for this task
    comparisons = compute_task_comparisons(measurements)
    assert comparisons == []


def test_cost_delta_is_routed_minus_baseline():
    measurements = [
        _m(strategy="always_small", task_id="task_001", cost=0.001),
        _m(strategy="rule_based_routing", task_id="task_001", cost=0.0015),
    ]
    comparisons = compute_task_comparisons(measurements)
    assert abs(comparisons[0].cost_delta - 0.0005) < 1e-9


def test_compare_strategy_cost_reports_negative_savings_honestly():
    baseline = summarize_strategy("always_small", [_m(cost=0.001), _m(cost=0.001)])
    routing = summarize_strategy("rule_based_routing", [_m(cost=0.0015), _m(cost=0.0015)])
    comparison = compare_strategy_cost(baseline, routing)
    assert comparison.routing_cost_savings_percent < 0  # routing cost MORE -- never hidden/clamped


def test_compare_strategy_cost_reports_positive_savings_when_cheaper():
    baseline = summarize_strategy("always_small", [_m(cost=0.002)])
    routing = summarize_strategy("rule_based_routing", [_m(cost=0.001)])
    comparison = compare_strategy_cost(baseline, routing)
    assert comparison.routing_cost_savings_percent > 0


def test_compare_strategy_cost_none_when_cost_data_unavailable():
    baseline = summarize_strategy("always_small", [_m(succeeded=False, cost=None)])
    routing = summarize_strategy("rule_based_routing", [_m(cost=0.001)])
    comparison = compare_strategy_cost(baseline, routing)
    assert comparison.absolute_cost_delta_usd is None
