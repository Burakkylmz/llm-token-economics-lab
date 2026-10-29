"""Token economics, cost accounting, context-compression measurement, and
break-even analysis for Lab10 (Parts 9, 10, 11.2, 11.6).

Every ratio/percentage guards against a missing or zero denominator by
returning `None` -- never a fabricated number, never a ZeroDivisionError
(Parts 9.3, 10.4, 11.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab10_conversation_summarization.evaluation import (
    STATUS_CORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_MISSING_RESPONSE,
    STATUS_TRUNCATED,
)
from labs.lab10_conversation_summarization.models import FollowupMeasurement, SummaryMeasurement

BREAK_EVEN_REACHED = "REACHED"
BREAK_EVEN_NOT_REACHED = "NOT_REACHED"


def net_savings_usd(baseline_cost: Optional[float], strategy_cost: Optional[float]) -> Optional[float]:
    if baseline_cost is None or strategy_cost is None:
        return None
    return baseline_cost - strategy_cost


def net_savings_percent(baseline_cost: Optional[float], strategy_cost: Optional[float]) -> Optional[float]:
    savings = net_savings_usd(baseline_cost, strategy_cost)
    if savings is None or baseline_cost in (None, 0):
        return None
    return 100 * savings / baseline_cost


def context_input_reduction_tokens(full_tokens: Optional[int], summary_tokens: Optional[int]) -> Optional[int]:
    if full_tokens is None or summary_tokens is None:
        return None
    return full_tokens - summary_tokens


def context_input_reduction_percent(full_tokens: Optional[int], summary_tokens: Optional[int]) -> Optional[float]:
    reduction = context_input_reduction_tokens(full_tokens, summary_tokens)
    if reduction is None or full_tokens in (None, 0):
        return None
    return 100 * reduction / full_tokens


def request_compression_ratio(full_tokens: Optional[int], summary_tokens: Optional[int]) -> Optional[float]:
    if full_tokens is None or summary_tokens in (None, 0):
        return None
    return full_tokens / summary_tokens


def cost_per_correct_answer(total_workflow_cost_usd: Optional[float], correct_answers: int) -> Optional[float]:
    if total_workflow_cost_usd is None or correct_answers <= 0:
        return None
    return total_workflow_cost_usd / correct_answers


@dataclass(frozen=True)
class FollowupConditionStats:
    evaluated_answers: int
    correct_answers: int
    invalid_format_count: int
    missing_answer_count: int
    truncated_count: int
    answer_accuracy: Optional[float]
    total_prompt_tokens: Optional[int]
    total_completion_tokens: Optional[int]
    total_reasoning_tokens: Optional[int]
    total_cost_usd: Optional[float]


def summarize_followups(followups: Sequence[FollowupMeasurement]) -> FollowupConditionStats:
    evaluated = [m for m in followups if m.request_status == "SUCCESS"]
    correct = sum(1 for m in evaluated if m.evaluation_status == STATUS_CORRECT)
    invalid_format = sum(1 for m in evaluated if m.evaluation_status == STATUS_INVALID_FORMAT)
    missing = sum(1 for m in evaluated if m.evaluation_status == STATUS_MISSING_RESPONSE)
    truncated = sum(1 for m in evaluated if m.evaluation_status == STATUS_TRUNCATED)

    prompt_tokens = [m.followup_prompt_tokens for m in evaluated if m.followup_prompt_tokens is not None]
    completion_tokens = [m.followup_completion_tokens for m in evaluated if m.followup_completion_tokens is not None]
    reasoning_tokens = [m.followup_reasoning_tokens for m in evaluated if m.followup_reasoning_tokens is not None]
    costs = [m.followup_estimated_cost_usd for m in evaluated if m.followup_estimated_cost_usd is not None]

    return FollowupConditionStats(
        evaluated_answers=len(evaluated),
        correct_answers=correct,
        invalid_format_count=invalid_format,
        missing_answer_count=missing,
        truncated_count=truncated,
        answer_accuracy=(correct / len(evaluated)) if evaluated else None,
        total_prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
        total_completion_tokens=sum(completion_tokens) if completion_tokens else None,
        total_reasoning_tokens=sum(reasoning_tokens) if reasoning_tokens else None,
        total_cost_usd=sum(costs) if costs else None,
    )


@dataclass(frozen=True)
class ConditionWorkflowCost:
    dataset_id: str
    condition_id: str
    summary_generation_cost_usd: Optional[float]
    followup_cost_usd: Optional[float]
    total_workflow_cost_usd: Optional[float]


def compute_workflow_cost(
    dataset_id: str, condition_id: str, summary_measurement: Optional[SummaryMeasurement], followups: Sequence[FollowupMeasurement],
) -> ConditionWorkflowCost:
    """The summary-generation cost is included EXACTLY ONCE here, never once
    per follow-up question (Part 10.2's explicit requirement)."""
    summary_cost = summary_measurement.summary_generation_estimated_cost_usd if summary_measurement is not None else None
    followup_stats = summarize_followups(followups)
    followup_cost = followup_stats.total_cost_usd

    if summary_measurement is None:
        total = followup_cost
    elif summary_cost is None or followup_cost is None:
        total = None
    else:
        total = summary_cost + followup_cost

    return ConditionWorkflowCost(
        dataset_id=dataset_id, condition_id=condition_id, summary_generation_cost_usd=summary_cost,
        followup_cost_usd=followup_cost, total_workflow_cost_usd=total,
    )


@dataclass(frozen=True)
class ContextComparisonRow:
    dataset_id: str
    question_id: str
    full_request_prompt_tokens: Optional[int]
    low_request_prompt_tokens: Optional[int]
    medium_request_prompt_tokens: Optional[int]
    low_context_reduction_tokens: Optional[int]
    low_context_reduction_percent: Optional[float]
    low_compression_ratio: Optional[float]
    medium_context_reduction_tokens: Optional[int]
    medium_context_reduction_percent: Optional[float]
    medium_compression_ratio: Optional[float]


def compute_context_comparison(
    dataset_id: str, question_id: str,
    baseline_tokens: Optional[int], low_tokens: Optional[int], medium_tokens: Optional[int],
) -> ContextComparisonRow:
    return ContextComparisonRow(
        dataset_id=dataset_id, question_id=question_id,
        full_request_prompt_tokens=baseline_tokens, low_request_prompt_tokens=low_tokens,
        medium_request_prompt_tokens=medium_tokens,
        low_context_reduction_tokens=context_input_reduction_tokens(baseline_tokens, low_tokens),
        low_context_reduction_percent=context_input_reduction_percent(baseline_tokens, low_tokens),
        low_compression_ratio=request_compression_ratio(baseline_tokens, low_tokens),
        medium_context_reduction_tokens=context_input_reduction_tokens(baseline_tokens, medium_tokens),
        medium_context_reduction_percent=context_input_reduction_percent(baseline_tokens, medium_tokens),
        medium_compression_ratio=request_compression_ratio(baseline_tokens, medium_tokens),
    )


@dataclass(frozen=True)
class BreakEvenResult:
    baseline_cumulative_costs: tuple[Optional[float], ...]
    summary_cumulative_costs: tuple[Optional[float], ...]
    break_even_call_index: Optional[int]
    break_even_status: str


def compute_break_even(
    summary_generation_cost_usd: Optional[float],
    baseline_followup_costs: Sequence[Optional[float]],
    summary_followup_costs: Sequence[Optional[float]],
) -> BreakEvenResult:
    """`break_even_call_index` is the smallest N >= 1 for which the
    summarization strategy's OBSERVED cumulative cost (summary generation
    cost once, plus the first N summarized follow-up costs) is <= the
    baseline's observed cumulative cost over the same first N follow-ups.
    Never extrapolated beyond the measured questions (Part 10.6) -- once
    any required cost in the running sum is missing, every cumulative
    value from that point on is also None (a missing term makes everything
    after it unknown, not zero)."""
    if len(baseline_followup_costs) != len(summary_followup_costs):
        raise ValueError("baseline_followup_costs and summary_followup_costs must have equal length")

    baseline_cumulative: list[Optional[float]] = []
    running: Optional[float] = 0.0
    for cost in baseline_followup_costs:
        running = None if (running is None or cost is None) else running + cost
        baseline_cumulative.append(running)

    summary_cumulative: list[Optional[float]] = []
    running = summary_generation_cost_usd
    for cost in summary_followup_costs:
        running = None if (running is None or cost is None) else running + cost
        summary_cumulative.append(running)

    break_even_index: Optional[int] = None
    for i, (b, s) in enumerate(zip(baseline_cumulative, summary_cumulative), start=1):
        if b is not None and s is not None and s <= b:
            break_even_index = i
            break

    status = BREAK_EVEN_REACHED if break_even_index is not None else BREAK_EVEN_NOT_REACHED
    return BreakEvenResult(
        baseline_cumulative_costs=tuple(baseline_cumulative), summary_cumulative_costs=tuple(summary_cumulative),
        break_even_call_index=break_even_index, break_even_status=status,
    )
