"""Structured summary generation (Part 5) -- exactly one call per
dataset/effort, never regenerated per follow-up question.

The summarization model receives ONLY the fixed instruction and the raw
conversation transcript -- never golden questions, expected answers, or
evaluation rules (Part 12.4's no-leakage requirement).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing
from lab_core.groq_client import call_sync
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab10_conversation_summarization.context import parse_summary_json
from labs.lab10_conversation_summarization.datasets import Conversation, render_transcript
from labs.lab10_conversation_summarization.models import SummaryMeasurement, TraceRecord
from labs.lab10_conversation_summarization.prompts import SUMMARIZATION_SYSTEM_PROMPT

STATUS_SUCCESS = "SUCCESS"
STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"


def generate_summary(
    client: Groq,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    conversation: Conversation,
    pricing: Optional[ModelPricing],
    experiment_run_id: str,
    execution_index: int,
    condition_id: str,
) -> tuple[SummaryMeasurement, Optional[dict], TraceRecord]:
    transcript = render_transcript(conversation)
    messages = [
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    base = dict(
        experiment_run_id=experiment_run_id, execution_index=execution_index, dataset_id=conversation.dataset_id,
        condition_id=condition_id, reasoning_effort=reasoning_effort, timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
    )
    call_id = str(uuid.uuid4())

    try:
        result = call_sync(
            client, model=model, messages=messages, reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens, response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 -- any provider/transport failure is recorded, never silently lost.
        measurement = SummaryMeasurement(**base, request_status=STATUS_UNKNOWN_ERROR, error=f"{type(exc).__name__}: {exc}")
        trace = TraceRecord(
            experiment_run_id=experiment_run_id, dataset_id=conversation.dataset_id, condition_id=condition_id,
            call_id=call_id, call_purpose="summary_generation", question_id=None, request_model=model,
            request_parameters={"reasoning_effort": reasoning_effort, "max_completion_tokens": max_completion_tokens},
            request_messages=messages, response_visible_content=None, response_finish_reason=None,
            response_usage={}, response_latency_ms=None, evaluation_status=None,
        )
        return measurement, None, trace

    tokens = result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    summary_dict, json_valid, schema_errors = parse_summary_json(result.message_content)

    measurement = SummaryMeasurement(
        **base,
        request_status=STATUS_SUCCESS,
        summary_generation_prompt_tokens=tokens.prompt_tokens,
        summary_generation_completion_tokens=tokens.completion_tokens,
        summary_generation_reasoning_tokens=tokens.reasoning_tokens,
        summary_generation_non_reasoning_completion_tokens=non_reasoning,
        summary_generation_cached_tokens=tokens.cached_tokens,
        summary_generation_total_tokens=tokens.total_tokens,
        summary_generation_latency_ms=result.performance.client_elapsed_ms,
        summary_generation_estimated_cost_usd=tokens.estimated_cost_usd,
        finish_reason=tokens.finish_reason,
        original_message_count=len(conversation.messages),
        original_conversation_character_count=len(transcript),
        generated_summary_character_count=len(result.message_content) if result.message_content else None,
        json_valid=json_valid,
        schema_errors=tuple(schema_errors),
    )
    trace = TraceRecord(
        experiment_run_id=experiment_run_id, dataset_id=conversation.dataset_id, condition_id=condition_id,
        call_id=call_id, call_purpose="summary_generation", question_id=None, request_model=model,
        request_parameters={"reasoning_effort": reasoning_effort, "max_completion_tokens": max_completion_tokens},
        request_messages=messages, response_visible_content=result.message_content,
        response_finish_reason=tokens.finish_reason, response_usage={
            "prompt_tokens": tokens.prompt_tokens, "completion_tokens": tokens.completion_tokens,
            "reasoning_tokens": tokens.reasoning_tokens, "cached_tokens": tokens.cached_tokens,
            "total_tokens": tokens.total_tokens,
        }, response_latency_ms=result.performance.client_elapsed_ms, evaluation_status=None,
    )
    return measurement, summary_dict, trace
