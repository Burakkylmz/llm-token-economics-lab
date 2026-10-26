"""Executes (or dry-runs) a Lab01 `effort` experiment plan against the real
Groq API.

`Budget`, the retry wrapper, the cost-ceiling estimators, and the quality
heuristic are REUSED verbatim from labs.lab01_token_anatomy.factorial_runner:
none of them ever referenced temperature/top_p (they only cared about
max_completion_tokens, pricing, and retry/backoff policy), so they are
exactly as correct here as they were there. Only `execute_planned_call` is
new, because it is the one place that must NOT send temperature/top_p to
the API at all (see build_messages_and_params_for_condition below) and
must build an EffortMeasurement rather than a FactorialMeasurement.
"""

from __future__ import annotations

from datetime import datetime, timezone

from groq import Groq

from lab_core.config import ModelPricing, RetryConfig
from lab_core.groq_client import call_sync
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.effort_models import EffortMeasurement, PlannedEffortCall
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.factorial_runner import (  # noqa: F401 -- re-exported for callers
    Budget,
    call_with_operational_retry,
    conservative_per_call_cost_ceiling,
    estimate_plan_cost_ceiling,
    evaluate_quality_heuristic,
)
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens


def build_create_kwargs_for_condition(
    model: str, prompt_type: str, reasoning_effort: str, max_completion_tokens: int, api_seed: int | None = None
) -> dict:
    """The exact request this mode sends. `temperature`/`top_p` are not set
    to a default here or anywhere else -- they are simply absent from this
    dict, so the Groq API falls back to its own provider defaults, which
    this project does not know and does not claim to know (see
    docs/instructor/README.md's "Temperature ve top-p" note).
    """
    create_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": EXPERIMENTS[prompt_type]}],
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
    }
    if api_seed is not None:
        create_kwargs["seed"] = api_seed
    return create_kwargs


def execute_planned_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: ModelPricing | None,
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_call: PlannedEffortCall,
) -> EffortMeasurement:
    condition = planned_call.condition
    create_kwargs = build_create_kwargs_for_condition(
        model, condition.prompt_type, condition.reasoning_effort, max_completion_tokens, planned_call.api_seed
    )
    assert "temperature" not in create_kwargs and "top_p" not in create_kwargs

    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)

    timestamp = datetime.now(timezone.utc).isoformat()
    base = dict(
        experiment_run_id=experiment_run_id,
        condition_id=condition.condition_id,
        execution_index=planned_call.execution_index,
        repetition_index=planned_call.repetition_index,
        timestamp=timestamp,
        prompt_type=condition.prompt_type,
        model=model,
        reasoning_effort=condition.reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        api_seed=planned_call.api_seed,
        attempts_used=attempts_used,
    )

    if call_result is None:
        return EffortMeasurement(**base, succeeded=False, error=error)

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)

    non_reasoning_completion_tokens = compute_non_reasoning_completion_tokens(
        tokens.completion_tokens, tokens.reasoning_tokens
    )
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None
    quality_method, quality_passed = evaluate_quality_heuristic(condition.prompt_type, call_result.message_content)

    return EffortMeasurement(
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
        estimated_cost_usd=tokens.estimated_cost_usd,
        quality_evaluation_method=quality_method,
        quality_heuristic_passed=quality_passed,
    )
