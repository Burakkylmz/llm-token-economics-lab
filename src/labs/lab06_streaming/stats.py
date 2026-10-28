"""Descriptive statistics, per-condition summaries, and paired
streaming-vs-non-streaming comparisons for Lab06.

`DescriptiveStats`/`compute_descriptive_stats` are reused verbatim from
labs.lab01_token_anatomy.factorial_stats.

Paired comparison percentages are always denominated against
`non_streaming_total_latency_ms` (the baseline reference point, stated
explicitly per Part J's requirement) -- and a negative gain is never
clamped to zero or hidden (Part J: "Negative perceived latency gain
gizlenmemeli").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab01_token_anatomy.factorial_stats import (  # noqa: F401 -- re-exported for callers
    DescriptiveStats,
    compute_descriptive_stats,
)
from labs.lab06_streaming.evaluation import (
    STATUS_API_ERROR,
    STATUS_COMPLETE,
    STATUS_EVALUATION_UNAVAILABLE,
    STATUS_INCOMPLETE,
    STATUS_NO_VISIBLE_CONTENT,
    STATUS_STREAM_INTERRUPTED,
    STATUS_TRUNCATED,
)
from labs.lab06_streaming.models import CONDITION_NON_STREAMING, CONDITION_STREAMING, USAGE_AVAILABLE, StreamMeasurement

BOTH_COMPLETE = "BOTH_COMPLETE"
STREAMING_MORE_COMPLETE = "STREAMING_MORE_COMPLETE"
NON_STREAMING_MORE_COMPLETE = "NON_STREAMING_MORE_COMPLETE"
BOTH_INCOMPLETE = "BOTH_INCOMPLETE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ConditionSummary:
    condition: str
    total_calls: int
    successful_calls: int

    first_stream_event_ms: DescriptiveStats
    first_visible_content_ms: DescriptiveStats
    # Unified across conditions: total_stream_latency_ms values for
    # streaming rows, non_streaming_total_latency_ms values for
    # non_streaming rows -- never both at once for the same row.
    total_latency_ms: DescriptiveStats

    completed_count: int
    incomplete_count: int
    truncated_count: int
    no_visible_content_count: int
    stream_interrupted_count: int
    api_error_count: int
    evaluation_unavailable_count: int

    usage_available_count: int
    usage_unavailable_count: int

    total_estimated_cost_usd: Optional[float]
    mean_estimated_cost_usd: Optional[float]


def _total_latency(m: StreamMeasurement) -> Optional[float]:
    return m.total_stream_latency_ms if m.condition == CONDITION_STREAMING else m.non_streaming_total_latency_ms


def summarize_condition(condition: str, measurements: Sequence[StreamMeasurement]) -> ConditionSummary:
    successes = [m for m in measurements if m.succeeded]
    status_counts = {
        STATUS_COMPLETE: 0, STATUS_INCOMPLETE: 0, STATUS_TRUNCATED: 0, STATUS_NO_VISIBLE_CONTENT: 0,
        STATUS_STREAM_INTERRUPTED: 0, STATUS_API_ERROR: 0, STATUS_EVALUATION_UNAVAILABLE: 0,
    }
    for m in measurements:
        status_counts[m.evaluation_status] = status_counts.get(m.evaluation_status, 0) + 1

    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    usage_available = sum(1 for m in successes if m.usage_status == USAGE_AVAILABLE)

    return ConditionSummary(
        condition=condition,
        total_calls=len(measurements),
        successful_calls=len(successes),
        first_stream_event_ms=compute_descriptive_stats([m.first_stream_event_ms for m in successes]),
        first_visible_content_ms=compute_descriptive_stats([m.first_visible_content_ms for m in successes]),
        total_latency_ms=compute_descriptive_stats([_total_latency(m) for m in successes]),
        completed_count=status_counts[STATUS_COMPLETE],
        incomplete_count=status_counts[STATUS_INCOMPLETE],
        truncated_count=status_counts[STATUS_TRUNCATED],
        no_visible_content_count=status_counts[STATUS_NO_VISIBLE_CONTENT],
        stream_interrupted_count=status_counts[STATUS_STREAM_INTERRUPTED],
        api_error_count=status_counts[STATUS_API_ERROR],
        evaluation_unavailable_count=status_counts[STATUS_EVALUATION_UNAVAILABLE],
        usage_available_count=usage_available,
        usage_unavailable_count=len(successes) - usage_available,
        total_estimated_cost_usd=sum(costs) if costs else None,
        mean_estimated_cost_usd=(sum(costs) / len(costs)) if costs else None,
    )


def summarize_all_conditions(conditions: Sequence[str], measurements: Sequence[StreamMeasurement]) -> list[ConditionSummary]:
    by_condition: dict[str, list[StreamMeasurement]] = defaultdict(list)
    for m in measurements:
        by_condition[m.condition].append(m)
    return [summarize_condition(c, by_condition[c]) for c in conditions]


@dataclass
class PairComparison:
    pair_id: int
    streaming_first_visible_content_ms: Optional[float]
    streaming_total_latency_ms: Optional[float]
    non_streaming_total_latency_ms: Optional[float]
    # non_streaming_total_latency_ms - streaming_first_visible_content_ms.
    # Positive = streaming showed visible content earlier in THIS pair.
    perceived_latency_gain_ms: Optional[float]
    # Denominator is explicitly non_streaming_total_latency_ms (the
    # baseline reference point).
    perceived_latency_gain_percent: Optional[float]
    # non_streaming_total_latency_ms - streaming_total_latency_ms.
    total_latency_difference_ms: Optional[float]
    quality_comparison: str


def _quality_comparison(streaming_complete: Optional[bool], non_streaming_complete: Optional[bool]) -> str:
    if streaming_complete is None or non_streaming_complete is None:
        return INSUFFICIENT_DATA
    if streaming_complete and non_streaming_complete:
        return BOTH_COMPLETE
    if streaming_complete and not non_streaming_complete:
        return STREAMING_MORE_COMPLETE
    if not streaming_complete and non_streaming_complete:
        return NON_STREAMING_MORE_COMPLETE
    return BOTH_INCOMPLETE


def compute_pair_comparisons(measurements: Sequence[StreamMeasurement]) -> list[PairComparison]:
    """Pairs every pair_id that has exactly one successful streaming AND
    one successful non_streaming measurement. A pair missing either side
    (e.g. a single-condition run, or a budget-stopped run) is simply
    skipped -- never compared against a fabricated placeholder."""
    by_pair: dict[int, dict[str, StreamMeasurement]] = defaultdict(dict)
    for m in measurements:
        if m.succeeded and m.condition in (CONDITION_STREAMING, CONDITION_NON_STREAMING):
            by_pair[m.pair_id][m.condition] = m

    comparisons = []
    for pair_id in sorted(by_pair):
        streaming = by_pair[pair_id].get(CONDITION_STREAMING)
        non_streaming = by_pair[pair_id].get(CONDITION_NON_STREAMING)
        if streaming is None or non_streaming is None:
            continue

        gain_ms = (
            non_streaming.non_streaming_total_latency_ms - streaming.first_visible_content_ms
            if non_streaming.non_streaming_total_latency_ms is not None and streaming.first_visible_content_ms is not None
            else None
        )
        gain_percent = (
            (gain_ms / non_streaming.non_streaming_total_latency_ms) * 100
            if gain_ms is not None and non_streaming.non_streaming_total_latency_ms
            else None
        )
        total_diff_ms = (
            non_streaming.non_streaming_total_latency_ms - streaming.total_stream_latency_ms
            if non_streaming.non_streaming_total_latency_ms is not None and streaming.total_stream_latency_ms is not None
            else None
        )

        comparisons.append(
            PairComparison(
                pair_id=pair_id,
                streaming_first_visible_content_ms=streaming.first_visible_content_ms,
                streaming_total_latency_ms=streaming.total_stream_latency_ms,
                non_streaming_total_latency_ms=non_streaming.non_streaming_total_latency_ms,
                perceived_latency_gain_ms=gain_ms,
                perceived_latency_gain_percent=gain_percent,
                total_latency_difference_ms=total_diff_ms,
                quality_comparison=_quality_comparison(streaming.answer_complete, non_streaming.answer_complete),
            )
        )
    return comparisons
