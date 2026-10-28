"""Descriptive statistics, per-strategy summaries, and task-level
baseline-vs-routed comparisons for Lab05.

`DescriptiveStats`/`compute_descriptive_stats` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats -- fully generic, already correct
and tested.

Quality regression detection (`_quality_outcome`) never looks at cost or
latency -- a cheaper/faster routed result is never allowed to overshadow a
quality regression, and the four/five outcome categories are reported
per-task, not just aggregated away.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
)
from labs.lab05_model_routing.evaluation import (
    STATUS_CORRECT,
    STATUS_EVALUATION_UNAVAILABLE,
    STATUS_INCOMPLETE,
    STATUS_INCORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_STRUCTURAL_VALIDATION_ONLY,
    STATUS_TRUNCATED,
)
from labs.lab05_model_routing.matrix import STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER
from labs.lab05_model_routing.models import RoutingMeasurement

ROUTING_QUALITY_REGRESSION = "ROUTING_QUALITY_REGRESSION"
ROUTING_QUALITY_IMPROVEMENT = "ROUTING_QUALITY_IMPROVEMENT"
QUALITY_PRESERVED = "QUALITY_PRESERVED"
BOTH_FAILED = "BOTH_FAILED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class StrategySummary:
    strategy: str
    total_tasks: int
    correct_tasks: int
    incorrect_tasks: int
    incomplete_tasks: int
    invalid_format_tasks: int
    truncated_tasks: int
    structural_validation_only_tasks: int
    # None of the total_tasks is 0 -- otherwise the ratio's denominator.
    accuracy: Optional[float]

    total_estimated_cost_usd: Optional[float]
    mean_cost_per_task_usd: Optional[float]
    latency_ms: DescriptiveStats
    total_workflow_wall_clock_ms: Optional[float]

    small_model_calls: int
    large_model_calls: int
    small_model_selection_ratio: Optional[float]
    large_model_selection_ratio: Optional[float]
    # Only meaningful for the router strategy -- empty dict for the two
    # fixed baselines, never a fabricated/misleading rule distribution.
    routing_rule_distribution: dict


def summarize_strategy(strategy: str, measurements: Sequence[RoutingMeasurement]) -> StrategySummary:
    successes = [m for m in measurements if m.succeeded]
    status_counts = {
        STATUS_CORRECT: 0, STATUS_INCORRECT: 0, STATUS_INCOMPLETE: 0, STATUS_INVALID_FORMAT: 0,
        STATUS_TRUNCATED: 0, STATUS_STRUCTURAL_VALIDATION_ONLY: 0, STATUS_EVALUATION_UNAVAILABLE: 0,
    }
    for m in measurements:
        status_counts[m.evaluation_status] = status_counts.get(m.evaluation_status, 0) + 1

    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    small_calls = sum(1 for m in successes if m.model_tier == "small")
    large_calls = sum(1 for m in successes if m.model_tier == "large")
    total_calls = small_calls + large_calls

    rule_distribution: dict[str, int] = {}
    for m in successes:
        for rule in m.matched_rules:
            rule_distribution[rule] = rule_distribution.get(rule, 0) + 1

    total_tasks = len(measurements)
    return StrategySummary(
        strategy=strategy,
        total_tasks=total_tasks,
        correct_tasks=status_counts[STATUS_CORRECT],
        incorrect_tasks=status_counts[STATUS_INCORRECT],
        incomplete_tasks=status_counts[STATUS_INCOMPLETE],
        invalid_format_tasks=status_counts[STATUS_INVALID_FORMAT],
        truncated_tasks=status_counts[STATUS_TRUNCATED],
        structural_validation_only_tasks=status_counts[STATUS_STRUCTURAL_VALIDATION_ONLY],
        accuracy=(status_counts[STATUS_CORRECT] / total_tasks) if total_tasks else None,
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_cost_per_task_usd=(sum(costs) / len(costs)) if costs else None,
        latency_ms=compute_descriptive_stats([m.total_latency_ms for m in successes]),
        total_workflow_wall_clock_ms=None,  # filled in by the CLI, which times the whole strategy loop
        small_model_calls=small_calls,
        large_model_calls=large_calls,
        small_model_selection_ratio=(small_calls / total_calls) if total_calls else None,
        large_model_selection_ratio=(large_calls / total_calls) if total_calls else None,
        routing_rule_distribution=rule_distribution,
    )


def summarize_all_strategies(
    strategies: Sequence[str], measurements: Sequence[RoutingMeasurement]
) -> list[StrategySummary]:
    by_strategy: dict[str, list[RoutingMeasurement]] = defaultdict(list)
    for m in measurements:
        by_strategy[m.strategy].append(m)
    return [summarize_strategy(s, by_strategy[s]) for s in strategies]


@dataclass
class TaskComparison:
    task_id: str
    baseline_strategy: str
    routed_strategy: str
    baseline_model: Optional[str]
    routed_model: Optional[str]
    baseline_correct: Optional[bool]
    routed_correct: Optional[bool]
    baseline_cost: Optional[float]
    routed_cost: Optional[float]
    baseline_latency: Optional[float]
    routed_latency: Optional[float]
    cost_delta: Optional[float]
    latency_delta: Optional[float]
    quality_outcome: str


def _quality_outcome(baseline_correct: Optional[bool], routed_correct: Optional[bool]) -> str:
    if baseline_correct is None or routed_correct is None:
        return INSUFFICIENT_DATA
    if baseline_correct and routed_correct:
        return QUALITY_PRESERVED
    if baseline_correct and not routed_correct:
        return ROUTING_QUALITY_REGRESSION
    if not baseline_correct and routed_correct:
        return ROUTING_QUALITY_IMPROVEMENT
    return BOTH_FAILED


def compute_task_comparisons(
    measurements: Sequence[RoutingMeasurement],
    baseline_strategy: str = STRATEGY_ALWAYS_SMALL,
    routed_strategy: str = STRATEGY_ROUTER,
) -> list[TaskComparison]:
    """Pairs every task_id that has exactly one measurement under each of
    `baseline_strategy`/`routed_strategy`. A task missing either side is
    simply skipped (never compared against a fabricated placeholder)."""
    by_task_strategy: dict[tuple[str, str], RoutingMeasurement] = {}
    for m in measurements:
        if m.strategy in (baseline_strategy, routed_strategy):
            by_task_strategy[(m.task_id, m.strategy)] = m

    task_ids = sorted({task_id for task_id, _ in by_task_strategy})
    comparisons = []
    for task_id in task_ids:
        baseline = by_task_strategy.get((task_id, baseline_strategy))
        routed = by_task_strategy.get((task_id, routed_strategy))
        if baseline is None or routed is None:
            continue

        cost_delta = (
            routed.estimated_cost_usd - baseline.estimated_cost_usd
            if routed.estimated_cost_usd is not None and baseline.estimated_cost_usd is not None
            else None
        )
        latency_delta = (
            routed.total_latency_ms - baseline.total_latency_ms
            if routed.total_latency_ms is not None and baseline.total_latency_ms is not None
            else None
        )
        comparisons.append(
            TaskComparison(
                task_id=task_id,
                baseline_strategy=baseline_strategy,
                routed_strategy=routed_strategy,
                baseline_model=baseline.selected_model if baseline.succeeded else None,
                routed_model=routed.selected_model if routed.succeeded else None,
                baseline_correct=baseline.answer_correct,
                routed_correct=routed.answer_correct,
                baseline_cost=baseline.estimated_cost_usd,
                routed_cost=routed.estimated_cost_usd,
                baseline_latency=baseline.total_latency_ms,
                routed_latency=routed.total_latency_ms,
                cost_delta=cost_delta,
                latency_delta=latency_delta,
                quality_outcome=_quality_outcome(baseline.answer_correct, routed.answer_correct),
            )
        )
    return comparisons


@dataclass
class CostComparison:
    baseline_total_cost_usd: Optional[float]
    routing_total_cost_usd: Optional[float]
    absolute_cost_delta_usd: Optional[float]
    relative_cost_delta_percent: Optional[float]
    # Positive = routing saved money; negative = routing cost MORE. Never
    # hidden or clamped to zero when negative (Part L's explicit rule).
    routing_cost_savings_percent: Optional[float]


def compare_strategy_cost(baseline: StrategySummary, routing: StrategySummary) -> CostComparison:
    baseline_cost = baseline.total_estimated_cost_usd
    routing_cost = routing.total_estimated_cost_usd
    if baseline_cost is None or routing_cost is None:
        return CostComparison(baseline_cost, routing_cost, None, None, None)

    absolute_delta = routing_cost - baseline_cost
    relative_delta = (absolute_delta / baseline_cost * 100) if baseline_cost > 0 else None
    savings_percent = -relative_delta if relative_delta is not None else None
    return CostComparison(baseline_cost, routing_cost, absolute_delta, relative_delta, savings_percent)
