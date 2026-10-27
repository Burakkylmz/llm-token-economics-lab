"""Descriptive statistics and per-condition summaries for Lab02.

`DescriptiveStats`/`compute_descriptive_stats`/`compute_grouped_effect`/
`compute_level_difference` are REUSED verbatim from
labs.lab01_token_anatomy.factorial_stats: they are fully generic (operate
via getattr on any object with a `.succeeded` flag and the named
attributes), already correct and tested, and require no Lab02-specific
behavior at all. Only the condition-summary shape (which adds the answer
correctness tally Lab01 has no equivalent of) is new here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
    compute_grouped_effect,
    compute_level_difference,
)
from labs.lab02_context_growth.evaluation import (
    STATUS_CORRECT,
    STATUS_FORMAT_INVALID,
    STATUS_INCORRECT,
    STATUS_MISSING_RESPONSE,
)
from labs.lab02_context_growth.models import ContextMeasurement


@dataclass
class ContextConditionSummary:
    condition_id: str

    planned_calls: int
    successful_calls: int
    failed_calls: int
    truncated_calls: int
    truncation_rate: float | None

    context_chars: int | None

    prompt_tokens: DescriptiveStats
    completion_tokens: DescriptiveStats
    reasoning_tokens: DescriptiveStats
    non_reasoning_completion_tokens: DescriptiveStats
    latency_ms: DescriptiveStats

    total_estimated_cost_usd: float | None
    mean_estimated_cost_usd: float | None

    correct_count: int
    incorrect_count: int
    format_invalid_count: int
    missing_response_count: int


def summarize_context_condition(
    condition_id: str,
    planned_calls: int,
    measurements: Sequence[ContextMeasurement],
) -> ContextConditionSummary:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.finish_reason == "length"]
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    context_chars_values = {m.context_chars for m in measurements if m.context_chars is not None}

    return ContextConditionSummary(
        condition_id=condition_id,
        planned_calls=planned_calls,
        successful_calls=len(successes),
        failed_calls=len(failures),
        truncated_calls=len(truncated),
        truncation_rate=(len(truncated) / len(successes)) if successes else None,
        context_chars=next(iter(context_chars_values)) if len(context_chars_values) == 1 else None,
        prompt_tokens=compute_descriptive_stats([m.prompt_tokens for m in successes]),
        completion_tokens=compute_descriptive_stats([m.completion_tokens for m in successes]),
        reasoning_tokens=compute_descriptive_stats([m.reasoning_tokens for m in successes]),
        non_reasoning_completion_tokens=compute_descriptive_stats(
            [m.non_reasoning_completion_tokens for m in successes]
        ),
        latency_ms=compute_descriptive_stats([m.total_latency_ms for m in successes]),
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_estimated_cost_usd=(sum(costs) / len(costs)) if costs else None,
        correct_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_CORRECT),
        incorrect_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_INCORRECT),
        format_invalid_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_FORMAT_INVALID),
        missing_response_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_MISSING_RESPONSE),
    )


def summarize_all_context_conditions(
    condition_ids: Sequence[str],
    planned_calls_by_condition: dict[str, int],
    measurements: Sequence[ContextMeasurement],
) -> list[ContextConditionSummary]:
    by_condition: dict[str, list[ContextMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition_id].append(m)

    return [
        summarize_context_condition(
            condition_id, planned_calls_by_condition.get(condition_id, 0), by_condition[condition_id]
        )
        for condition_id in condition_ids
    ]
