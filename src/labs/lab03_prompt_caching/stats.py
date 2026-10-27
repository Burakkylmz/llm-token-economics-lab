"""Descriptive statistics and per-condition summaries for Lab03.

`DescriptiveStats`/`compute_descriptive_stats`/`compute_grouped_effect`/
`compute_level_difference` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats -- fully generic, already correct
and tested, same as labs.lab02_context_growth.stats.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

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
from labs.lab03_prompt_caching.cache_metrics import compute_observed_request_cache_hit_rate, compute_token_cache_ratio
from labs.lab03_prompt_caching.cache_status import CACHE_STATUS_VALUES
from labs.lab03_prompt_caching.models import CacheMeasurement


@dataclass
class CacheConditionSummary:
    condition_id: str

    planned_calls: int
    successful_calls: int
    failed_calls: int
    truncated_calls: int
    truncation_rate: Optional[float]

    prompt_tokens: DescriptiveStats
    cached_tokens: DescriptiveStats
    uncached_prompt_tokens: DescriptiveStats
    completion_tokens: DescriptiveStats
    reasoning_tokens: DescriptiveStats
    non_reasoning_completion_tokens: DescriptiveStats
    latency_ms: DescriptiveStats

    # Metric A vs Metric B -- see labs.lab03_prompt_caching.cache_metrics.
    # Never the same number; both None when no successful call reported
    # cached_tokens at all.
    observed_request_cache_hit_rate: Optional[float]
    token_cache_ratio: Optional[float]
    reported_cache_usage_calls: int
    # Count of successful calls per canonical 4-way cache status (see
    # labs.lab03_prompt_caching.cache_status). Always sums to
    # successful_calls; CACHE_USAGE_UNAVAILABLE is tracked explicitly here
    # rather than being indistinguishable from CACHE_MISS.
    cache_status_counts: dict

    total_estimated_cost_usd: Optional[float]
    mean_estimated_cost_usd: Optional[float]
    total_estimated_cache_savings_usd: Optional[float]
    mean_estimated_cache_savings_usd: Optional[float]
    mean_estimated_cache_savings_percent: Optional[float]

    correct_count: int
    incorrect_count: int
    format_invalid_count: int
    missing_response_count: int


def summarize_cache_condition(
    condition_id: str,
    planned_calls: int,
    measurements: Sequence[CacheMeasurement],
) -> CacheConditionSummary:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    truncated = [m for m in successes if m.finish_reason == "length"]

    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    savings = [m.estimated_cache_savings_usd for m in successes if m.estimated_cache_savings_usd is not None]
    savings_percents = [
        m.estimated_cache_savings_percent for m in successes if m.estimated_cache_savings_percent is not None
    ]
    reported_cache_usage_calls = sum(1 for m in successes if m.cached_tokens is not None)
    cache_status_counts = {status: sum(1 for m in successes if m.cache_status == status) for status in CACHE_STATUS_VALUES}

    return CacheConditionSummary(
        condition_id=condition_id,
        planned_calls=planned_calls,
        successful_calls=len(successes),
        failed_calls=len(failures),
        truncated_calls=len(truncated),
        truncation_rate=(len(truncated) / len(successes)) if successes else None,
        prompt_tokens=compute_descriptive_stats([m.prompt_tokens for m in successes]),
        cached_tokens=compute_descriptive_stats([m.cached_tokens for m in successes]),
        uncached_prompt_tokens=compute_descriptive_stats([m.uncached_prompt_tokens for m in successes]),
        completion_tokens=compute_descriptive_stats([m.completion_tokens for m in successes]),
        reasoning_tokens=compute_descriptive_stats([m.reasoning_tokens for m in successes]),
        non_reasoning_completion_tokens=compute_descriptive_stats(
            [m.non_reasoning_completion_tokens for m in successes]
        ),
        latency_ms=compute_descriptive_stats([m.total_latency_ms for m in successes]),
        observed_request_cache_hit_rate=compute_observed_request_cache_hit_rate(successes),
        token_cache_ratio=compute_token_cache_ratio(successes),
        reported_cache_usage_calls=reported_cache_usage_calls,
        cache_status_counts=cache_status_counts,
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_estimated_cost_usd=(sum(costs) / len(costs)) if costs else None,
        total_estimated_cache_savings_usd=sum(savings) if savings else None,
        mean_estimated_cache_savings_usd=(sum(savings) / len(savings)) if savings else None,
        mean_estimated_cache_savings_percent=(sum(savings_percents) / len(savings_percents)) if savings_percents else None,
        correct_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_CORRECT),
        incorrect_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_INCORRECT),
        format_invalid_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_FORMAT_INVALID),
        missing_response_count=sum(1 for m in successes if m.answer_evaluation_status == STATUS_MISSING_RESPONSE),
    )


def summarize_all_cache_conditions(
    condition_ids: Sequence[str],
    planned_calls_by_condition: dict[str, int],
    measurements: Sequence[CacheMeasurement],
) -> list[CacheConditionSummary]:
    by_condition: dict[str, list[CacheMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition_id].append(m)

    return [
        summarize_cache_condition(condition_id, planned_calls_by_condition.get(condition_id, 0), by_condition[condition_id])
        for condition_id in condition_ids
    ]
