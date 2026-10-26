"""Descriptive statistics and per-condition summaries for Lab01's `effort`
mode -- the single-factor analogue of labs.lab01_token_anatomy.factorial_stats.

`DescriptiveStats`/`compute_descriptive_stats` and `compute_grouped_effect`/
`compute_level_difference` are intentionally REUSED from factorial_stats
rather than duplicated: they are fully generic (operate via getattr, no
temperature/top_p coupling at all) and already correct and tested. Only the
condition-summary shape is new here, because labs.lab01_token_anatomy.
factorial_stats.ConditionSummary carries temperature/top_p fields an
EffortCondition simply doesn't have.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.effort_models import EffortCondition, EffortMeasurement
from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
    compute_grouped_effect,
    compute_level_difference,
)


@dataclass
class EffortConditionSummary:
    condition_id: str
    prompt_type: str
    reasoning_effort: str

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

    mean_reasoning_ratio: Optional[float]
    token_weighted_reasoning_ratio: Optional[float]


def _reasoning_ratio(reasoning_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    if reasoning_tokens is None or completion_tokens is None or completion_tokens == 0:
        return None
    return reasoning_tokens / completion_tokens


def summarize_effort_condition(
    condition: EffortCondition,
    planned_calls: int,
    measurements: Sequence[EffortMeasurement],
) -> EffortConditionSummary:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.finish_reason == "length"]

    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    per_call_ratios = [
        r for r in (_reasoning_ratio(m.reasoning_tokens, m.completion_tokens) for m in successes) if r is not None
    ]
    reasoning_sum = sum(m.reasoning_tokens for m in successes if m.reasoning_tokens is not None)
    completion_sum_for_ratio = sum(
        m.completion_tokens for m in successes if m.reasoning_tokens is not None and m.completion_tokens is not None
    )

    return EffortConditionSummary(
        condition_id=condition.condition_id,
        prompt_type=condition.prompt_type,
        reasoning_effort=condition.reasoning_effort,
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
        mean_reasoning_ratio=(sum(per_call_ratios) / len(per_call_ratios)) if per_call_ratios else None,
        token_weighted_reasoning_ratio=(
            reasoning_sum / completion_sum_for_ratio if completion_sum_for_ratio else None
        ),
    )


def summarize_all_effort_conditions(
    conditions: Sequence[EffortCondition],
    planned_calls_by_condition: dict[str, int],
    measurements: Sequence[EffortMeasurement],
) -> list[EffortConditionSummary]:
    by_condition: dict[str, list[EffortMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition_id].append(m)

    return [
        summarize_effort_condition(
            condition, planned_calls_by_condition.get(condition.condition_id, 0), by_condition[condition.condition_id]
        )
        for condition in conditions
    ]
