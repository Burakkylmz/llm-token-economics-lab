"""Descriptive statistics, per-condition summaries, and main/interaction
effect tables for Lab01's factorial measurements.

Percentile method, explicitly documented (Part D of this project's design
requires this to be stated, not left implicit): `p50`/`median` use
statistics.median (the middle value / average of the two middle values);
`p95` uses statistics.quantiles(..., n=100, method="inclusive")[94] -- the
"inclusive" method treats the data's own min/max as the 0th/100th
percentile, which is the more common convention in spreadsheet/pandas-style
reporting. Both require at least 2 observations; with fewer, they are None
rather than a fabricated single-point "statistic".

None in these aggregates always means "not enough data", never 0 -- a
metric missing from every call in a group (e.g. no condition reported
reasoning_tokens) must not silently look identical to "reasoning_tokens was
reliably measured at 0".
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from labs.lab01_token_anatomy.factorial_models import FactorialCondition, FactorialMeasurement


@dataclass
class DescriptiveStats:
    count: int
    count_missing: int
    mean: Optional[float] = None
    median: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    range: Optional[float] = None
    sample_standard_deviation: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None


def compute_descriptive_stats(values: Sequence[Optional[float]]) -> DescriptiveStats:
    available = [float(v) for v in values if v is not None]
    count_missing = len(values) - len(available)

    if not available:
        return DescriptiveStats(count=0, count_missing=count_missing)

    stats = DescriptiveStats(
        count=len(available),
        count_missing=count_missing,
        mean=statistics.mean(available),
        median=statistics.median(available),
        minimum=min(available),
        maximum=max(available),
        range=max(available) - min(available),
        p50=statistics.median(available),
    )
    if len(available) >= 2:
        stats.sample_standard_deviation = statistics.stdev(available)
        stats.p95 = statistics.quantiles(available, n=100, method="inclusive")[94]
    return stats


@dataclass
class ConditionSummary:
    condition_id: str
    prompt_type: str
    reasoning_effort: str
    temperature: float
    top_p: float

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

    # Mean of each call's OWN reasoning_tokens/completion_tokens ratio --
    # distinct from token_weighted_reasoning_ratio below (see module/field
    # docstrings in the task spec this implements: these are NOT the same
    # number and must not be collapsed into one).
    mean_reasoning_ratio: Optional[float]
    # sum(reasoning_tokens) / sum(completion_tokens) across all calls in this
    # condition where both are known -- weights calls with more tokens more
    # heavily, unlike mean_reasoning_ratio's simple per-call average.
    token_weighted_reasoning_ratio: Optional[float]


def _reasoning_ratio(reasoning_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    if reasoning_tokens is None or completion_tokens is None or completion_tokens == 0:
        return None
    return reasoning_tokens / completion_tokens


def summarize_condition(
    condition: FactorialCondition,
    planned_calls: int,
    measurements: Sequence[FactorialMeasurement],
) -> ConditionSummary:
    """measurements must be only the (successful or failed) measurements for this
    one condition_id -- callers are responsible for the grouping."""
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

    return ConditionSummary(
        condition_id=condition.condition_id,
        prompt_type=condition.prompt_type,
        reasoning_effort=condition.reasoning_effort,
        temperature=condition.temperature,
        top_p=condition.top_p,
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


def summarize_all_conditions(
    conditions: Sequence[FactorialCondition],
    planned_calls_by_condition: dict[str, int],
    measurements: Sequence[FactorialMeasurement],
) -> list[ConditionSummary]:
    by_condition: dict[str, list[FactorialMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition_id].append(m)

    return [
        summarize_condition(condition, planned_calls_by_condition.get(condition.condition_id, 0), by_condition[condition.condition_id])
        for condition in conditions
    ]


def compute_grouped_effect(
    measurements: Sequence[FactorialMeasurement],
    factors: Sequence[str],
    metric: str,
) -> dict[Any, DescriptiveStats]:
    """Group successful `measurements` by the given factor attribute(s) and
    compute DescriptiveStats of `metric` within each group.

    One factor -> a main effect table (averaged over every other factor's
    levels). Two or three factors -> an interaction/pivot table. The caller
    decides what's "controlled" by pre-filtering `measurements` (e.g. to one
    prompt_type and one top_p level) before calling this -- this function
    itself just groups and aggregates whatever it's given.
    """
    groups: dict[Any, list[Optional[float]]] = defaultdict(list)
    for m in measurements:
        if not m.succeeded:
            continue
        key = tuple(getattr(m, f) for f in factors) if len(factors) > 1 else getattr(m, factors[0])
        groups[key].append(getattr(m, metric))

    return {key: compute_descriptive_stats(values) for key, values in sorted(groups.items(), key=repr)}


def compute_level_difference(
    grouped_stats: dict[Any, DescriptiveStats], level_a: Any, level_b: Any
) -> Optional[float]:
    """mean(level_b) - mean(level_a), or None if either side's mean is unavailable
    (e.g. that level has zero successful measurements in this slice)."""
    stats_a = grouped_stats.get(level_a)
    stats_b = grouped_stats.get(level_b)
    if stats_a is None or stats_b is None or stats_a.mean is None or stats_b.mean is None:
        return None
    return stats_b.mean - stats_a.mean
