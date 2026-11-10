"""Executes (or dry-runs) a Lab01 factorial experiment plan against the real
Groq API, with operational retries, a cost/call budget guard, and
checkpoint-friendly persistence.

Reuses, rather than reimplements:
  - lab_core.groq_client.call_sync for the actual API call, timing, and
    usage/finish_reason/system_fingerprint normalization.
  - labs.lab01_token_anatomy.cost.calculate_cost for billing (built from a
    throwaway TokenCostMetrics so the exact same non-double-counting cost
    logic basic-mode Lab01 uses applies here too).
  - labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens
    for the same "None on inconsistent data, warn, never fabricate" rule.
  - lab_core.operational_retry for shared operational retries, distinct from
    Lab07's retry-strategy experiments.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing, RetryConfig
from lab_core.budget import Budget, conservative_per_call_cost_ceiling
from lab_core.operational_retry import RETRYABLE_EXCEPTIONS, call_with_operational_retry
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.factorial_models import FactorialMeasurement, PlannedCall
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens

def build_messages_for_condition(prompt_type: str) -> list[dict]:
    """Reuses Lab01's existing, unmodified short_prompt/long_prompt text --
    the factorial engine studies how generation parameters affect the
    response to a FIXED prompt, so the prompt itself must never change."""
    return [{"role": "user", "content": EXPERIMENTS[prompt_type]}]


def estimate_plan_cost_ceiling(plan_size: int, model_pricing: ModelPricing, max_completion_tokens: int) -> float:
    """Worst-case-prompt-text-adjusted estimate across the whole plan. Uses the
    longer of Lab01's two prompts as a conservative stand-in for "whatever
    this plan's calls use" -- good enough for a pre-flight budget check, not
    a per-condition exact figure (see per-call docstring above for why)."""
    longest_prompt = max(EXPERIMENTS.values(), key=len)
    per_call = conservative_per_call_cost_ceiling(model_pricing, max_completion_tokens, longest_prompt)
    return per_call * plan_size


def evaluate_quality_heuristic(prompt_type: str, message_content: Optional[str]) -> tuple[str, Optional[bool]]:
    """A deliberately weak, explicitly-labeled heuristic -- NOT a correctness
    guarantee (see this project's Part G scientific-validity constraints).

    short_prompt asks for the capital of France: a case-insensitive
    substring check for "paris" is used, labeled "heuristic_contains_paris".
    This WILL misclassify a response that mentions "Paris" while denying it
    (e.g. "It is not Paris, it's Lyon") as a false positive -- that is a
    known, accepted limitation of substring matching, not a bug to silently
    paper over, which is exactly why the method name says "heuristic" and
    why this is never called "is_correct" or "answer_correct" anywhere in
    this project.

    long_prompt has no objective rubric in this project, so it is always
    reported as "not_evaluated" -- quality is never claimed to be measured
    for it.
    """
    if prompt_type != "short_prompt":
        return "not_evaluated", None
    if message_content is None:
        return "heuristic_contains_paris", None
    return "heuristic_contains_paris", "paris" in message_content.lower()


def execute_planned_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_call: PlannedCall,
) -> FactorialMeasurement:
    """Runs one PlannedCall end-to-end: API call (with operational retry),
    cost calc, reasoning-token derivation, and the quality heuristic. Always
    returns exactly one FactorialMeasurement -- on failure, every optional
    measurement field stays None (never fabricated) and `succeeded=False`.
    """
    condition = planned_call.condition
    create_kwargs: dict = {
        "model": model,
        "messages": build_messages_for_condition(condition.prompt_type),
        "reasoning_effort": condition.reasoning_effort,
        "temperature": condition.temperature,
        "top_p": condition.top_p,
        "max_completion_tokens": max_completion_tokens,
    }
    if planned_call.api_seed is not None:
        create_kwargs["seed"] = planned_call.api_seed

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
        temperature=condition.temperature,
        top_p=condition.top_p,
        max_completion_tokens=max_completion_tokens,
        api_seed=planned_call.api_seed,
        attempts_used=attempts_used,
    )

    if call_result is None:
        return FactorialMeasurement(**base, succeeded=False, error=error)

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)

    non_reasoning_completion_tokens = compute_non_reasoning_completion_tokens(
        tokens.completion_tokens, tokens.reasoning_tokens
    )
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None
    quality_method, quality_passed = evaluate_quality_heuristic(condition.prompt_type, call_result.message_content)

    return FactorialMeasurement(
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
