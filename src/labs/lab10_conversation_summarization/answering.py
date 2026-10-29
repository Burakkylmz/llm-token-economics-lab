"""Downstream follow-up answering (Part 6) -- identical outer request shape
and identical generation settings (model/reasoning_effort/max_completion_tokens)
across all three conditions (Part 6.2/12.1). The ONLY variable passed in is
`context_payload`. The golden expected answer is NEVER included in the
request (Part 12.4)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing
from lab_core.groq_client import call_sync
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab10_conversation_summarization.context import build_followup_user_message
from labs.lab10_conversation_summarization.datasets import GoldenQuestion
from labs.lab10_conversation_summarization.evaluation import evaluate_followup_answer
from labs.lab10_conversation_summarization.models import FollowupMeasurement, TraceRecord
from labs.lab10_conversation_summarization.prompts import ANSWER_INSTRUCTION

STATUS_SUCCESS = "SUCCESS"
STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"


def answer_question(
    client: Groq,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    context_payload: str,
    question: GoldenQuestion,
    pricing: Optional[ModelPricing],
    experiment_run_id: str,
    execution_index: int,
    dataset_id: str,
    condition_id: str,
) -> tuple[FollowupMeasurement, TraceRecord]:
    messages = [
        {"role": "system", "content": ANSWER_INSTRUCTION},
        {"role": "user", "content": build_followup_user_message(context_payload, question.question, question.answer_type, question.allowed_enum_values)},
    ]
    base = dict(
        experiment_run_id=experiment_run_id, execution_index=execution_index, dataset_id=dataset_id,
        condition_id=condition_id, question_id=question.question_id, timestamp=datetime.now(timezone.utc).isoformat(),
        model=model, reasoning_effort=reasoning_effort,
    )
    call_id = str(uuid.uuid4())
    request_parameters = {"reasoning_effort": reasoning_effort, "max_completion_tokens": max_completion_tokens}

    try:
        result = call_sync(
            client, model=model, messages=messages, reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens, response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, never silently lost; other questions remain reportable.
        measurement = FollowupMeasurement(**base, request_status=STATUS_UNKNOWN_ERROR, error=f"{type(exc).__name__}: {exc}")
        trace = TraceRecord(
            experiment_run_id=experiment_run_id, dataset_id=dataset_id, condition_id=condition_id, call_id=call_id,
            call_purpose="followup_answer", question_id=question.question_id, request_model=model,
            request_parameters=request_parameters, request_messages=messages, response_visible_content=None,
            response_finish_reason=None, response_usage={}, response_latency_ms=None, evaluation_status=measurement.evaluation_status,
        )
        return measurement, trace

    tokens = result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    evaluation = evaluate_followup_answer(
        result.message_content, tokens.finish_reason, question.expected_answer, question.answer_type, question.allowed_enum_values,
    )

    measurement = FollowupMeasurement(
        **base,
        request_status=STATUS_SUCCESS,
        followup_prompt_tokens=tokens.prompt_tokens,
        followup_completion_tokens=tokens.completion_tokens,
        followup_reasoning_tokens=tokens.reasoning_tokens,
        followup_non_reasoning_completion_tokens=non_reasoning,
        followup_cached_tokens=tokens.cached_tokens,
        followup_total_tokens=tokens.total_tokens,
        followup_latency_ms=result.performance.client_elapsed_ms,
        followup_estimated_cost_usd=tokens.estimated_cost_usd,
        finish_reason=tokens.finish_reason,
        evaluation_status=evaluation.status,
        answer_correct=evaluation.answer_correct,
    )
    trace = TraceRecord(
        experiment_run_id=experiment_run_id, dataset_id=dataset_id, condition_id=condition_id, call_id=call_id,
        call_purpose="followup_answer", question_id=question.question_id, request_model=model,
        request_parameters=request_parameters, request_messages=messages,
        response_visible_content=result.message_content, response_finish_reason=tokens.finish_reason,
        response_usage={
            "prompt_tokens": tokens.prompt_tokens, "completion_tokens": tokens.completion_tokens,
            "reasoning_tokens": tokens.reasoning_tokens, "cached_tokens": tokens.cached_tokens,
            "total_tokens": tokens.total_tokens,
        }, response_latency_ms=result.performance.client_elapsed_ms, evaluation_status=evaluation.status,
    )
    return measurement, trace
