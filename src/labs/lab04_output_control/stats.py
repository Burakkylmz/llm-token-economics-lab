"""Descriptive statistics and per-condition/per-strategy summaries for
Lab04.

`DescriptiveStats`/`compute_descriptive_stats` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats -- fully generic, already correct
and tested, same as labs.lab02_context_growth.stats and
labs.lab03_prompt_caching.stats.

Quality-constrained comparison (RQ7, Part H): a strategy is never judged
cheaper-and-better merely because its total cost is lower -- see
`compare_invocation_strategies` below, which checks
`all_answers_accepted` on BOTH strategies before naming a cost-based
status, and explicitly labels the case where the cheaper strategy's
answers were not all accepted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
)
from labs.lab04_output_control.evaluation import STATUS_ACCEPTED, STATUS_INCOMPLETE, STATUS_MISSING_RESPONSE
from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement

BOTH_STRATEGIES_ACCEPTED = "BOTH_STRATEGIES_ACCEPTED"
CHEAPER_STRATEGY_ALSO_ACCEPTED = "CHEAPER_STRATEGY_ALSO_ACCEPTED"
LOWER_COST_BUT_QUALITY_FAILED = "LOWER_COST_BUT_QUALITY_FAILED"
QUALITY_FAILED_BOTH = "QUALITY_FAILED_BOTH"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class OutputLimitConditionSummary:
    condition_id: str
    max_completion_tokens: int

    planned_calls: int
    successful_calls: int
    failed_calls: int
    truncated_calls: int
    truncation_rate: Optional[float]

    completion_tokens: DescriptiveStats
    reasoning_tokens: DescriptiveStats
    non_reasoning_completion_tokens: DescriptiveStats
    latency_ms: DescriptiveStats

    total_estimated_cost_usd: Optional[float]
    mean_estimated_cost_usd: Optional[float]

    accepted_count: int
    incomplete_count: int
    missing_response_count: int
    mean_field_completeness_ratio: Optional[float]
    mean_field_correctness_ratio: Optional[float]


def summarize_output_limit_condition(
    condition_id: str,
    max_completion_tokens: int,
    planned_calls: int,
    measurements: Sequence[OutputLimitMeasurement],
) -> OutputLimitConditionSummary:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.output_truncated is True]
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    completeness = [m.field_completeness_ratio for m in successes if m.field_completeness_ratio is not None]
    correctness = [m.field_correctness_ratio for m in successes if m.field_correctness_ratio is not None]

    return OutputLimitConditionSummary(
        condition_id=condition_id,
        max_completion_tokens=max_completion_tokens,
        planned_calls=planned_calls,
        successful_calls=len(successes),
        failed_calls=len(failures),
        truncated_calls=len(truncated),
        truncation_rate=(len(truncated) / len(successes)) if successes else None,
        completion_tokens=compute_descriptive_stats([m.completion_tokens for m in successes]),
        reasoning_tokens=compute_descriptive_stats([m.reasoning_tokens for m in successes]),
        non_reasoning_completion_tokens=compute_descriptive_stats(
            [m.non_reasoning_completion_tokens for m in successes]
        ),
        latency_ms=compute_descriptive_stats([m.total_latency_ms for m in successes]),
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_estimated_cost_usd=(sum(costs) / len(costs)) if costs else None,
        accepted_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_ACCEPTED),
        incomplete_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_INCOMPLETE),
        missing_response_count=sum(1 for m in measurements if m.answer_evaluation_status == STATUS_MISSING_RESPONSE),
        mean_field_completeness_ratio=(sum(completeness) / len(completeness)) if completeness else None,
        mean_field_correctness_ratio=(sum(correctness) / len(correctness)) if correctness else None,
    )


def summarize_all_output_limit_conditions(
    conditions: Sequence[tuple[str, int]],
    planned_calls_by_condition: dict[str, int],
    measurements: Sequence[OutputLimitMeasurement],
) -> list[OutputLimitConditionSummary]:
    by_condition: dict[str, list[OutputLimitMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition_id].append(m)

    return [
        summarize_output_limit_condition(
            condition_id, max_completion_tokens, planned_calls_by_condition.get(condition_id, 0), by_condition[condition_id]
        )
        for condition_id, max_completion_tokens in conditions
    ]


@dataclass
class InvocationStrategySummary:
    strategy: str
    api_call_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_estimated_cost_usd: Optional[float]
    workflow_wall_clock_ms: Optional[float]
    individual_request_latencies_ms: list[Optional[float]]
    correct_answers: int
    missing_answers: int
    invalid_answers: int
    all_answers_accepted: bool


def summarize_invocation_strategy(
    strategy: str, measurements: Sequence[InvocationMeasurement], workflow_wall_clock_ms: Optional[float]
) -> InvocationStrategySummary:
    successes = [m for m in measurements if m.succeeded]
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]

    return InvocationStrategySummary(
        strategy=strategy,
        api_call_count=len(measurements),
        total_prompt_tokens=sum(m.prompt_tokens or 0 for m in successes),
        total_completion_tokens=sum(m.completion_tokens or 0 for m in successes),
        total_reasoning_tokens=sum(m.reasoning_tokens or 0 for m in successes),
        total_estimated_cost_usd=sum(costs) if costs else None,
        workflow_wall_clock_ms=workflow_wall_clock_ms,
        individual_request_latencies_ms=[m.client_elapsed_ms for m in measurements],
        correct_answers=sum(1 for m in measurements if m.overall_answer_accepted),
        missing_answers=sum(1 for m in measurements if m.answer_evaluation_status == STATUS_MISSING_RESPONSE),
        invalid_answers=sum(1 for m in measurements if m.answer_evaluation_status == STATUS_INCOMPLETE),
        all_answers_accepted=bool(measurements) and all(m.overall_answer_accepted for m in measurements),
    )


@dataclass
class InvocationComparisonSummary:
    single: InvocationStrategySummary
    multiple: InvocationStrategySummary
    # multiple.total_estimated_cost_usd - single.total_estimated_cost_usd
    # (positive => multiple invocations cost more).
    absolute_cost_difference_usd: Optional[float]
    relative_cost_difference_percent: Optional[float]
    # multiple.workflow_wall_clock_ms - single.workflow_wall_clock_ms.
    workflow_latency_difference_ms: Optional[float]
    quality_comparison: str


def _quality_comparison(single_ok: bool, multiple_ok: bool, cost_diff: Optional[float]) -> str:
    if single_ok and multiple_ok:
        return BOTH_STRATEGIES_ACCEPTED
    if not single_ok and not multiple_ok:
        return QUALITY_FAILED_BOTH
    if cost_diff is None:
        return INSUFFICIENT_DATA
    single_is_cheaper = cost_diff > 0
    cheaper_strategy_accepted = single_ok if single_is_cheaper else multiple_ok
    return CHEAPER_STRATEGY_ALSO_ACCEPTED if cheaper_strategy_accepted else LOWER_COST_BUT_QUALITY_FAILED


def compare_invocation_strategies(
    single: InvocationStrategySummary, multiple: InvocationStrategySummary
) -> InvocationComparisonSummary:
    cost_diff = None
    relative_diff = None
    if single.total_estimated_cost_usd is not None and multiple.total_estimated_cost_usd is not None:
        cost_diff = multiple.total_estimated_cost_usd - single.total_estimated_cost_usd
        if single.total_estimated_cost_usd > 0:
            relative_diff = (cost_diff / single.total_estimated_cost_usd) * 100

    latency_diff = None
    if single.workflow_wall_clock_ms is not None and multiple.workflow_wall_clock_ms is not None:
        latency_diff = multiple.workflow_wall_clock_ms - single.workflow_wall_clock_ms

    return InvocationComparisonSummary(
        single=single,
        multiple=multiple,
        absolute_cost_difference_usd=cost_diff,
        relative_cost_difference_percent=relative_diff,
        workflow_latency_difference_ms=latency_diff,
        quality_comparison=_quality_comparison(single.all_answers_accepted, multiple.all_answers_accepted, cost_diff),
    )
