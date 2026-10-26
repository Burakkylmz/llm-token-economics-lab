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
  - lab_core.retry.compute_backoff_delay_seconds for backoff math (the
    *policy* below -- which exceptions are retryable, how many attempts --
    is this module's own operational-resilience concern, not Lab07's
    retry-strategy *experiment*; see this module's call_with_operational_retry
    docstring).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from groq import APIConnectionError, GroqError, Groq, InternalServerError, RateLimitError

from lab_core.config import ModelPricing, RetryConfig
from lab_core.groq_client import SyncCallResult, call_sync
from lab_core.retry import compute_backoff_delay_seconds
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.factorial_models import FactorialMeasurement, PlannedCall
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens

RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError)

# A crude, well-known heuristic (~4 characters per English token), used ONLY
# to put a rough, pre-flight, approximate figure on prompt-token cost for
# budget estimation -- never written into any measurement record, and never
# presented as a real token count. The real prompt_tokens for every call
# comes from the API's own usage field, same as basic-mode Lab01.
_APPROX_CHARS_PER_TOKEN = 4.0
# Safety margin applied on top of the chars/4 heuristic above, since it can
# understate token count for dense or non-English text; this only affects
# the PRE-FLIGHT estimate used for the budget gate, not any real measurement.
_PROMPT_ESTIMATE_SAFETY_MARGIN = 1.5


def build_messages_for_condition(prompt_type: str) -> list[dict]:
    """Reuses Lab01's existing, unmodified short_prompt/long_prompt text --
    the factorial engine studies how generation parameters affect the
    response to a FIXED prompt, so the prompt itself must never change."""
    return [{"role": "user", "content": EXPERIMENTS[prompt_type]}]


def conservative_per_call_cost_ceiling(model_pricing: ModelPricing, max_completion_tokens: int, prompt_text: str) -> float:
    """An upper bound on ONE call's cost, with one caveat stated explicitly:
    the completion-token part is a TRUE ceiling (completion_tokens can never
    exceed max_completion_tokens), but the prompt-token part is only an
    approximation (chars/4 with a 1.5x margin) because real tokenization
    isn't known before the call is made. The combined number is therefore a
    conservative ESTIMATE, not a guaranteed exact upper bound -- unusual
    tokenization (e.g. very non-English text) could in principle exceed it.
    """
    approx_prompt_tokens = (len(prompt_text) / _APPROX_CHARS_PER_TOKEN) * _PROMPT_ESTIMATE_SAFETY_MARGIN
    prompt_cost = approx_prompt_tokens / 1_000_000 * model_pricing.input_per_1m_usd
    completion_cost = max_completion_tokens / 1_000_000 * model_pricing.output_per_1m_usd
    return prompt_cost + completion_cost


def estimate_plan_cost_ceiling(plan_size: int, model_pricing: ModelPricing, max_completion_tokens: int) -> float:
    """Worst-case-prompt-text-adjusted estimate across the whole plan. Uses the
    longer of Lab01's two prompts as a conservative stand-in for "whatever
    this plan's calls use" -- good enough for a pre-flight budget check, not
    a per-condition exact figure (see per-call docstring above for why)."""
    longest_prompt = max(EXPERIMENTS.values(), key=len)
    per_call = conservative_per_call_cost_ceiling(model_pricing, max_completion_tokens, longest_prompt)
    return per_call * plan_size


@dataclass
class Budget:
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]
    per_call_cost_ceiling: float

    calls_made: int = 0
    accumulated_estimated_cost_usd: float = 0.0

    def can_make_another_call(self) -> Optional[str]:
        """None if another call is allowed; otherwise a human-readable reason
        it must stop, checked using the PER-CALL CEILING so the next call is
        never allowed to exceed the cost cap even in its worst case."""
        if self.max_api_calls is not None and self.calls_made >= self.max_api_calls:
            return f"max_api_calls budget ({self.max_api_calls}) reached"
        if self.max_estimated_cost_usd is not None:
            if self.accumulated_estimated_cost_usd + self.per_call_cost_ceiling > self.max_estimated_cost_usd:
                return (
                    f"max_estimated_cost_usd budget (${self.max_estimated_cost_usd:.4f}) would be "
                    f"exceeded by the next call's worst-case cost ceiling (${self.per_call_cost_ceiling:.6f}) "
                    f"on top of ${self.accumulated_estimated_cost_usd:.6f} already spent"
                )
        return None

    def record_call(self, actual_estimated_cost_usd: Optional[float]) -> None:
        self.calls_made += 1
        if actual_estimated_cost_usd is not None:
            self.accumulated_estimated_cost_usd += actual_estimated_cost_usd
        # NOTE: a FAILED call is still counted toward calls_made (it used up
        # one of the request budget's slots), but contributes $0 to
        # accumulated_estimated_cost_usd since no usage was returned. If a
        # provider still bills for some failed requests, that real cost
        # would NOT show up here -- accumulated_estimated_cost_usd is a
        # lower bound in the presence of failures, not a guarantee of exact
        # spend. The max_api_calls cap and the per-call ceiling check above
        # are what actually bound worst-case exposure regardless.


def call_with_operational_retry(
    client: Groq,
    retry_config: RetryConfig,
    sleep_fn: Callable[[float], None] = time.sleep,
    **create_kwargs,
) -> tuple[Optional[SyncCallResult], int, Optional[str]]:
    """Operational resilience for a long factorial campaign -- NOT Lab07's
    retry-STRATEGY experiment (lab07_rate_limiting/retry_runner.py), which
    exists specifically to compare retry policies against each other as an
    experiment variable. This wrapper applies one fixed, sensible policy
    (this project's existing [retry] config) so a 429/5xx/network blip
    doesn't abort a multi-hundred-call run, using the same backoff math.

    Returns (result, attempts_used, error_message). Exactly one of
    `result`/`error_message` is non-None.

    A non-retryable provider error (bad request, auth, permission, ...) is
    NOT retried, but it also does NOT propagate out of this function: it is
    recorded as this call's single failed attempt and returned like any
    other failure. One rejected parameter combination (e.g. an unsupported
    reasoning_effort/temperature/top_p combination the API itself refuses)
    must not abort the rest of a multi-hundred-call factorial campaign --
    it shows up as one explicit, visible failed measurement instead (see
    labs.lab01_token_anatomy.factorial_models.FactorialMeasurement.error).
    Any exception that is NOT a GroqError (a bug in this project's own code,
    e.g. a wrong keyword argument) is deliberately left to propagate and
    crash loudly rather than being misreported as an "API failure".
    """
    last_error: Optional[str] = None
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            result = call_sync(client, **create_kwargs)
            return result, attempt, None
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retry_config.max_attempts:
                return None, attempt, last_error

            retry_after_header = None
            if isinstance(exc, RateLimitError):
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None)
                retry_after_header = headers.get("retry-after") if headers is not None else None

            delay = (
                float(retry_after_header)
                if retry_after_header is not None
                else compute_backoff_delay_seconds(
                    attempt, retry_config.base_delay_seconds, retry_config.max_delay_seconds, retry_config.jitter
                )
            )
            if delay > 0:
                sleep_fn(delay)
        except GroqError as exc:
            return None, attempt, f"{type(exc).__name__}: {exc}"

    return None, retry_config.max_attempts, last_error


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
