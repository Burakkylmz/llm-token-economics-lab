"""Executes (or dry-runs) Lab03's paired cache-condition plan against the
real Groq API.

Reuses, rather than reimplements, this project's existing shared/Lab01/
Lab02 infrastructure:
  - lab_core.groq_client.call_sync for the actual API call, timing, and
    usage/finish_reason/system_fingerprint normalization (this is also
    where cached_tokens itself is extracted, via
    prompt_tokens_details.cached_tokens -- see lab_core.groq_client.normalize_usage).
  - labs.lab01_token_anatomy.factorial_runner.{Budget,
    call_with_operational_retry, conservative_per_call_cost_ceiling} for
    operational retry policy and cost/call budget enforcement -- fully
    generic, already correct and tested.
  - labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens
    for the same "None on inconsistent/missing data, never fabricated" rule.
  - labs.lab02_context_growth.evaluation.evaluate_integer_answer for the
    same strict, non-substring-matching integer evaluator Lab02 uses --
    Lab03's paired questions are deliberately the same "single integer"
    answer shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing, RetryConfig
from lab_core.budget import Budget, conservative_per_call_cost_ceiling  # noqa: F401 -- re-exported
from lab_core.operational_retry import call_with_operational_retry  # noqa: F401 -- re-exported
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab02_context_growth.evaluation import evaluate_integer_answer
from labs.lab03_prompt_caching.cache_cost import compute_cache_cost_breakdown
from labs.lab03_prompt_caching.cache_metrics import compute_request_cache_hit, compute_request_cache_ratio
from labs.lab03_prompt_caching.cache_status import classify_cache_status
from labs.lab03_prompt_caching.diagnostics import compute_request_hash, groq_sdk_version, usage_diagnostic_fields
from labs.lab03_prompt_caching.experiments import build_cache_messages, expected_answer_for
from labs.lab03_prompt_caching.models import CacheMeasurement, PlannedCacheCall

_SDK_VERSION = groq_sdk_version()


def _seconds_to_ms(value: Optional[float]) -> Optional[float]:
    return value * 1000 if value is not None else None


def execute_planned_cache_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_call: PlannedCacheCall,
) -> CacheMeasurement:
    messages = build_cache_messages(planned_call.condition_id, planned_call.pair_index, planned_call.question_index)
    expected_answer = expected_answer_for(planned_call.question_index)
    prompt_chars = sum(len(m["content"]) for m in messages)
    request_hash = compute_request_hash(model, messages, max_completion_tokens)

    create_kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)

    base = dict(
        experiment_run_id=experiment_run_id,
        condition_id=planned_call.condition_id,
        execution_index=planned_call.execution_index,
        pair_index=planned_call.pair_index,
        question_index=planned_call.question_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        max_completion_tokens=max_completion_tokens,
        attempts_used=attempts_used,
        prompt_chars=prompt_chars,
        expected_answer=expected_answer,
        request_hash=request_hash,
        sdk_version=_SDK_VERSION,
    )

    if call_result is None:
        return CacheMeasurement(**base, succeeded=False, error=error)

    tokens = call_result.tokens_and_cost
    # Diagnostics are extracted from the RAW provider usage object (never
    # from the already-normalized `tokens`), so "the field was never
    # reported" (prompt_tokens_details_present=False) can be told apart
    # from "it was reported as 0" even though both normalize to
    # cached_tokens=None/0 at the TokenCostMetrics layer for the former.
    # In practice tokens.cached_tokens is the correct, already-normalized
    # read of the very same raw object -- this call only adds presence
    # facts, it never re-derives cached_tokens itself.
    raw_usage = getattr(call_result.raw_response, "usage", None)
    usage_diag = usage_diagnostic_fields(raw_usage)

    cache_status = classify_cache_status(tokens.cached_tokens, tokens.prompt_tokens)
    cache_usage_inconsistent = cache_status == "INVALID_USAGE"
    uncached_prompt_tokens = (
        None if cache_usage_inconsistent else _safe_uncached(tokens.prompt_tokens, tokens.cached_tokens)
    )
    cost_breakdown = compute_cache_cost_breakdown(tokens, pricing)
    # calculate_cost's own result (via compute_cache_cost_breakdown) is the
    # single source of truth for estimated_cost_usd, applied unchanged --
    # never recomputed differently here.
    tokens.estimated_cost_usd = cost_breakdown.estimated_cost_usd

    non_reasoning_completion_tokens = compute_non_reasoning_completion_tokens(
        tokens.completion_tokens, tokens.reasoning_tokens
    )
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None
    answer_status, answer_correct = evaluate_integer_answer(call_result.message_content, expected_answer)

    return CacheMeasurement(
        **base,
        succeeded=True,
        error=None,
        prompt_tokens=tokens.prompt_tokens,
        cached_tokens=tokens.cached_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning_completion_tokens,
        total_tokens=tokens.total_tokens,
        cache_usage_inconsistent=cache_usage_inconsistent,
        finish_reason=tokens.finish_reason,
        system_fingerprint=call_result.system_fingerprint,
        final_answer_char_count=final_answer_char_count,
        total_latency_ms=call_result.performance.client_elapsed_ms,
        provider_reported_processing_ms=call_result.performance.provider_reported_processing_ms,
        provider_prompt_time_ms=_seconds_to_ms(getattr(raw_usage, "prompt_time", None)),
        provider_completion_time_ms=_seconds_to_ms(getattr(raw_usage, "completion_time", None)),
        provider_queue_time_ms=_seconds_to_ms(getattr(raw_usage, "queue_time", None)),
        estimated_cost_usd=cost_breakdown.estimated_cost_usd,
        estimated_input_cost_without_cache_usd=cost_breakdown.estimated_input_cost_without_cache_usd,
        estimated_input_cost_with_cache_usd=cost_breakdown.estimated_input_cost_with_cache_usd,
        estimated_cache_savings_usd=cost_breakdown.estimated_cache_savings_usd,
        estimated_cache_savings_percent=cost_breakdown.estimated_cache_savings_percent,
        request_cache_hit=compute_request_cache_hit(tokens.cached_tokens),
        request_cache_ratio=compute_request_cache_ratio(tokens.cached_tokens, tokens.prompt_tokens),
        cache_status=cache_status,
        prompt_tokens_details_present=usage_diag["prompt_tokens_details_present"],
        cached_tokens_field_present=usage_diag["cached_tokens_field_present"],
        raw_usage_field_names=usage_diag["raw_usage_field_names"],
        answer_evaluation_status=answer_status,
        answer_correct=answer_correct,
    )


def _safe_uncached(prompt_tokens: Optional[int], cached_tokens: Optional[int]) -> Optional[int]:
    if prompt_tokens is None or cached_tokens is None:
        return None
    uncached = prompt_tokens - cached_tokens
    return uncached if uncached >= 0 else None
