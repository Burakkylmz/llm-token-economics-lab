"""Executes (or dry-runs) Lab04's output-limit and invocation plans against
the real Groq API.

Reuses, rather than reimplements, this project's existing shared/Lab01
infrastructure (same pattern as labs.lab02_context_growth.runner and
labs.lab03_prompt_caching.runner):
  - lab_core.groq_client.call_sync for the actual API call, timing, and
    usage/finish_reason/system_fingerprint normalization.
  - labs.lab01_token_anatomy.factorial_runner.{Budget,
    call_with_operational_retry, conservative_per_call_cost_ceiling} for
    operational retry policy and cost/call budget enforcement.
  - labs.lab01_token_anatomy.cost.calculate_cost for billing.
  - labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens
    for the same "None on inconsistent/missing data, never fabricated" rule.
  - labs.lab04_output_control.evaluation for the deterministic field
    extraction/correctness/completeness evaluation.

Workflow wall-clock latency (RQ6) is measured HERE, around the whole
sequential loop of 3 calls for the multiple-invocation strategy, using
`time.perf_counter()` -- deliberately not the sum of the three individual
`client_elapsed_ms` values, since that sum would silently absorb any loop
overhead between calls. Both numbers are kept, never conflated (Part G.10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing, RetryConfig
from labs.lab01_token_anatomy.cost import calculate_cost
from lab_core.budget import Budget, conservative_per_call_cost_ceiling  # noqa: F401 -- re-exported
from lab_core.operational_retry import call_with_operational_retry  # noqa: F401 -- re-exported
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab04_output_control.evaluation import (
    EXPECTED_RETRY_FIELD_VALUES,
    REQUIRED_RETRY_FIELDS,
    evaluate_fields,
    evaluate_single_answer,
)
from labs.lab04_output_control.experiments import (
    EXPECTED_INVOCATION_FIELD_VALUES,
    REQUIRED_INVOCATION_FIELDS,
    build_invocation_multiple_messages,
    build_invocation_single_messages,
    build_output_limit_messages,
    expected_invocation_answer_for,
)
from labs.lab04_output_control.models import (
    InvocationMeasurement,
    OutputLimitMeasurement,
    PlannedInvocationCall,
    PlannedOutputLimitCall,
)


def _is_truncated(finish_reason: Optional[str]) -> Optional[bool]:
    """True iff the provider itself reported "length"; None when
    finish_reason is unavailable -- never inferred from token counts alone
    (Part D's non-negotiable rule)."""
    if finish_reason is None:
        return None
    return finish_reason == "length"


def _execute_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    reasoning_effort: str,
    messages: list[dict],
):
    create_kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
    }
    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)
    if call_result is None:
        return None, attempts_used, error, None

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    return call_result, attempts_used, None, non_reasoning


def execute_planned_output_limit_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    reasoning_effort: str,
    experiment_run_id: str,
    planned_call: PlannedOutputLimitCall,
) -> OutputLimitMeasurement:
    messages = build_output_limit_messages()
    base = dict(
        experiment_run_id=experiment_run_id,
        condition_id=planned_call.condition_id,
        execution_index=planned_call.execution_index,
        repetition_index=planned_call.repetition_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        max_completion_tokens=planned_call.max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )

    call_result, attempts_used, error, non_reasoning = _execute_call(
        client, retry_config, model, pricing, planned_call.max_completion_tokens, reasoning_effort, messages
    )
    if call_result is None:
        return OutputLimitMeasurement(**base, succeeded=False, attempts_used=attempts_used, error=error)

    tokens = call_result.tokens_and_cost
    evaluation = evaluate_fields(call_result.message_content, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None

    return OutputLimitMeasurement(
        **base,
        succeeded=True,
        attempts_used=attempts_used,
        error=None,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=tokens.finish_reason,
        output_truncated=_is_truncated(tokens.finish_reason),
        system_fingerprint=call_result.system_fingerprint,
        final_answer_char_count=final_answer_char_count,
        total_latency_ms=call_result.performance.client_elapsed_ms,
        provider_reported_processing_ms=call_result.performance.provider_reported_processing_ms,
        estimated_cost_usd=tokens.estimated_cost_usd,
        required_field_count=evaluation.required_field_count,
        present_field_count=evaluation.present_field_count,
        correct_field_count=evaluation.correct_field_count,
        incorrect_field_count=evaluation.incorrect_field_count,
        missing_field_count=evaluation.missing_field_count,
        duplicate_field_count=evaluation.duplicate_field_count,
        extra_field_count=evaluation.extra_field_count,
        field_completeness_ratio=evaluation.field_completeness_ratio,
        field_correctness_ratio=evaluation.field_correctness_ratio,
        answer_evaluation_status=evaluation.status,
        overall_answer_accepted=evaluation.overall_answer_accepted,
    )


def _invocation_measurement_from_call(
    experiment_run_id: str,
    strategy: str,
    planned_call: PlannedInvocationCall,
    model: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    call_result,
    attempts_used: int,
    error: Optional[str],
    non_reasoning: Optional[int],
) -> InvocationMeasurement:
    base = dict(
        experiment_run_id=experiment_run_id,
        strategy=strategy,
        execution_index=planned_call.execution_index,
        question_index=planned_call.question_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    if call_result is None:
        return InvocationMeasurement(**base, succeeded=False, attempts_used=attempts_used, error=error)

    tokens = call_result.tokens_and_cost
    if strategy == "single":
        evaluation = evaluate_fields(
            call_result.message_content, REQUIRED_INVOCATION_FIELDS, EXPECTED_INVOCATION_FIELD_VALUES
        )
    else:
        expected = expected_invocation_answer_for(planned_call.question_index)
        evaluation = evaluate_single_answer(call_result.message_content, expected)
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None

    return InvocationMeasurement(
        **base,
        succeeded=True,
        attempts_used=attempts_used,
        error=None,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=tokens.finish_reason,
        output_truncated=_is_truncated(tokens.finish_reason),
        final_answer_char_count=final_answer_char_count,
        client_elapsed_ms=call_result.performance.client_elapsed_ms,
        estimated_cost_usd=tokens.estimated_cost_usd,
        required_field_count=evaluation.required_field_count,
        present_field_count=evaluation.present_field_count,
        correct_field_count=evaluation.correct_field_count,
        incorrect_field_count=evaluation.incorrect_field_count,
        missing_field_count=evaluation.missing_field_count,
        duplicate_field_count=evaluation.duplicate_field_count,
        extra_field_count=evaluation.extra_field_count,
        field_completeness_ratio=evaluation.field_completeness_ratio,
        field_correctness_ratio=evaluation.field_correctness_ratio,
        answer_evaluation_status=evaluation.status,
        overall_answer_accepted=evaluation.overall_answer_accepted,
    )


@dataclass
class InvocationStrategyResult:
    measurements: list[InvocationMeasurement]
    # None if the budget stopped execution before any call in this strategy
    # completed (workflow wall-clock is only meaningful once at least the
    # first call has started).
    workflow_wall_clock_ms: Optional[float]


def run_invocation_strategy(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    reasoning_effort: str,
    experiment_run_id: str,
    planned_calls: list[PlannedInvocationCall],
    budget: "Budget",
) -> InvocationStrategyResult:
    """Runs every call in `planned_calls` (all "single", or all "multiple",
    in sequential order -- see lab04.invocation_execution_mode), timing the
    whole strategy's wall-clock span once, around the loop."""
    if not planned_calls:
        return InvocationStrategyResult(measurements=[], workflow_wall_clock_ms=None)

    strategy = planned_calls[0].strategy
    measurements: list[InvocationMeasurement] = []
    start = perf_counter()
    for planned_call in planned_calls:
        reason = budget.can_make_another_call()
        if reason:
            break
        messages = (
            build_invocation_single_messages()
            if planned_call.question_index is None
            else build_invocation_multiple_messages(planned_call.question_index)
        )
        call_result, attempts_used, error, non_reasoning = _execute_call(
            client, retry_config, model, pricing, max_completion_tokens, reasoning_effort, messages
        )
        measurement = _invocation_measurement_from_call(
            experiment_run_id, strategy, planned_call, model, max_completion_tokens, reasoning_effort,
            call_result, attempts_used, error, non_reasoning,
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        measurements.append(measurement)
    elapsed_ms = (perf_counter() - start) * 1000
    return InvocationStrategyResult(measurements=measurements, workflow_wall_clock_ms=elapsed_ms)
