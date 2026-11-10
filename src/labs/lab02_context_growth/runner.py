"""Executes (or dry-runs) Lab02's context-condition plan and multi-turn
growth conversation against the real Groq API.

Reuses, rather than reimplements, this project's existing shared/Lab01
infrastructure:
  - lab_core.groq_client.call_sync for the actual API call, timing, and
    usage/finish_reason/system_fingerprint normalization.
  - labs.lab01_token_anatomy.cost.calculate_cost for billing (same
    non-double-counting rule for reasoning_tokens as every other lab).
  - labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens
    for the same "None on inconsistent/missing data, never fabricated"
    rule.
  - labs.lab01_token_anatomy.factorial_runner.{Budget,
    call_with_operational_retry, conservative_per_call_cost_ceiling} for
    operational retry policy and cost/call budget enforcement -- these are
    fully generic (no temperature/top_p/prompt coupling), already correct
    and tested, and are exactly as applicable to Lab02's much smaller
    10-call default as they are to Lab01's factorial campaign.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing, RetryConfig
from labs.lab01_token_anatomy.cost import calculate_cost
from lab_core.budget import Budget, conservative_per_call_cost_ceiling  # noqa: F401 -- re-exported
from lab_core.operational_retry import call_with_operational_retry  # noqa: F401 -- re-exported
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab02_context_growth.evaluation import evaluate_integer_answer
from labs.lab02_context_growth.experiments import CONDITIONS, EXPECTED_ANSWER, build_context_messages
from labs.lab02_context_growth.models import ContextMeasurement, PlannedContextCall, TurnMeasurement

MULTI_TURN_SYSTEM_MESSAGE = "You are a concise assistant. Answer in one short sentence."


def execute_planned_context_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_call: PlannedContextCall,
) -> ContextMeasurement:
    context = CONDITIONS[planned_call.condition_id]
    create_kwargs = {
        "model": model,
        "messages": build_context_messages(context),
        "max_completion_tokens": max_completion_tokens,
    }

    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)

    base = dict(
        experiment_run_id=experiment_run_id,
        condition_id=planned_call.condition_id,
        execution_index=planned_call.execution_index,
        repetition_index=planned_call.repetition_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        max_completion_tokens=max_completion_tokens,
        attempts_used=attempts_used,
        context_chars=len(context),
        expected_answer=EXPECTED_ANSWER,
    )

    if call_result is None:
        return ContextMeasurement(**base, succeeded=False, error=error)

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)

    non_reasoning_completion_tokens = compute_non_reasoning_completion_tokens(
        tokens.completion_tokens, tokens.reasoning_tokens
    )
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None
    answer_status, answer_correct = evaluate_integer_answer(call_result.message_content, EXPECTED_ANSWER)

    return ContextMeasurement(
        **base,
        succeeded=True,
        error=None,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning_completion_tokens,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=tokens.finish_reason,
        system_fingerprint=call_result.system_fingerprint,
        final_answer_char_count=final_answer_char_count,
        total_latency_ms=call_result.performance.client_elapsed_ms,
        provider_reported_processing_ms=call_result.performance.provider_reported_processing_ms,
        estimated_cost_usd=tokens.estimated_cost_usd,
        answer_evaluation_status=answer_status,
        answer_correct=answer_correct,
    )


def execute_multi_turn_growth(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    experiment_run_id: str,
    turns: int,
    budget: Optional[Budget] = None,
) -> list[TurnMeasurement]:
    """Grows a single conversation turn by turn, measuring both the
    per-turn (incremental) and running (cumulative) input token/cost
    consumption at each turn.

    `budget`, if given, is checked before each turn (not just once up
    front) and execution stops -- returning whatever turns completed so
    far -- the moment it refuses another call; this never produces a
    partial/malformed turn, only fewer of them.
    """
    messages: list[dict] = [{"role": "system", "content": MULTI_TURN_SYSTEM_MESSAGE}]
    measurements: list[TurnMeasurement] = []
    cumulative_prompt_tokens = 0
    cumulative_total_tokens = 0
    cumulative_estimated_cost_usd = 0.0
    previous_prompt_tokens: Optional[int] = None

    for turn in range(1, turns + 1):
        if budget is not None:
            reason = budget.can_make_another_call()
            if reason is not None:
                break

        messages.append({"role": "user", "content": f"Tell me a short, distinct fact about topic #{turn}."})
        create_kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)

        base = dict(
            experiment_run_id=experiment_run_id,
            turn_index=turn,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            max_completion_tokens=max_completion_tokens,
            messages_count=len(messages),
            attempts_used=attempts_used,
        )

        if call_result is None:
            measurements.append(TurnMeasurement(**base, succeeded=False, error=error))
            if budget is not None:
                budget.record_call(None)
            # A failed turn cannot produce an assistant reply to continue the
            # conversation from -- stop growing it rather than silently
            # dropping this turn's (non-existent) user message either.
            messages.pop()
            break

        tokens = call_result.tokens_and_cost
        if pricing is not None:
            tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
        if budget is not None:
            budget.record_call(tokens.estimated_cost_usd)

        non_reasoning_completion_tokens = compute_non_reasoning_completion_tokens(
            tokens.completion_tokens, tokens.reasoning_tokens
        )

        incremental_prompt_tokens = (
            tokens.prompt_tokens - previous_prompt_tokens
            if tokens.prompt_tokens is not None and previous_prompt_tokens is not None
            else None
        )
        if tokens.prompt_tokens is not None:
            cumulative_prompt_tokens += tokens.prompt_tokens
            previous_prompt_tokens = tokens.prompt_tokens
        if tokens.total_tokens is not None:
            cumulative_total_tokens += tokens.total_tokens
        if tokens.estimated_cost_usd is not None:
            cumulative_estimated_cost_usd += tokens.estimated_cost_usd

        measurements.append(
            TurnMeasurement(
                **base,
                succeeded=True,
                error=None,
                prompt_tokens=tokens.prompt_tokens,
                completion_tokens=tokens.completion_tokens,
                reasoning_tokens=tokens.reasoning_tokens,
                non_reasoning_completion_tokens=non_reasoning_completion_tokens,
                total_tokens=tokens.total_tokens,
                cached_tokens=tokens.cached_tokens,
                finish_reason=tokens.finish_reason,
                total_latency_ms=call_result.performance.client_elapsed_ms,
                estimated_cost_usd=tokens.estimated_cost_usd,
                incremental_prompt_tokens=incremental_prompt_tokens,
                cumulative_prompt_tokens=cumulative_prompt_tokens if tokens.prompt_tokens is not None else None,
                cumulative_total_tokens=cumulative_total_tokens if tokens.total_tokens is not None else None,
                cumulative_estimated_cost_usd=(
                    cumulative_estimated_cost_usd if tokens.estimated_cost_usd is not None else None
                ),
            )
        )

        assistant_content = call_result.message_content
        messages.append({"role": "assistant", "content": assistant_content})

    return measurements
