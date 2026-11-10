"""Executes (or dry-runs) Lab05's routing plan against the real Groq API.

Reuses, rather than reimplements, this project's existing shared/Lab01
infrastructure (same pattern as labs.lab02-04's runners):
  - lab_core.groq_client.call_sync for the actual API call, timing, and
    usage/finish_reason/system_fingerprint normalization.
  - labs.lab01_token_anatomy.factorial_runner.{Budget,
    call_with_operational_retry, conservative_per_call_cost_ceiling} for
    operational retry policy and cost/call budget enforcement.
  - labs.lab01_token_anatomy.cost.calculate_cost for the total cost figure
    (labs.lab05_model_routing.cost.cost_breakdown adds the input/output
    split on top, from the exact same formula).
  - labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens
    for the same "None on inconsistent/missing data, never fabricated" rule.
  - labs.lab05_model_routing.evaluation for deterministic per-task
    correctness/completeness evaluation.
  - labs.lab05_model_routing.routing for the (zero-LLM-call) routing
    decision itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from lab_core.config import AppConfig, RetryConfig
from labs.lab01_token_anatomy.cost import calculate_cost
from lab_core.budget import Budget, conservative_per_call_cost_ceiling  # noqa: F401 -- re-exported
from lab_core.operational_retry import call_with_operational_retry  # noqa: F401 -- re-exported
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab05_model_routing.cost import cost_breakdown
from labs.lab05_model_routing.evaluation import evaluate_task_response
from labs.lab05_model_routing.golden_dataset import GoldenTask
from labs.lab05_model_routing.matrix import STRATEGY_ALWAYS_LARGE, STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER
from labs.lab05_model_routing.models import PlannedRoutingCall, RoutingMeasurement
from labs.lab05_model_routing.routing import RoutingDecision, choose_model, fixed_decision


def decide_for_strategy(strategy: str, task: GoldenTask, small_model: str, large_model: str) -> RoutingDecision:
    """The ONLY place a strategy is translated into an actual model choice
    -- a router test can call this directly without spinning up a client."""
    if strategy == STRATEGY_ALWAYS_SMALL:
        return fixed_decision(small_model, "small", STRATEGY_ALWAYS_SMALL)
    if strategy == STRATEGY_ALWAYS_LARGE:
        return fixed_decision(large_model, "large", STRATEGY_ALWAYS_LARGE)
    if strategy == STRATEGY_ROUTER:
        return choose_model(task.prompt, small_model, large_model)
    raise ValueError(f"Unknown strategy '{strategy}'")


def execute_planned_routing_call(
    client: Groq,
    retry_config: RetryConfig,
    config: AppConfig,
    experiment_run_id: str,
    planned_call: PlannedRoutingCall,
    task: GoldenTask,
) -> RoutingMeasurement:
    small_model = config.lab05.small_model
    large_model = config.lab05.large_model
    reasoning_effort = config.lab05.reasoning_effort
    max_completion_tokens = config.lab05.max_completion_tokens

    decision = decide_for_strategy(planned_call.strategy, task, small_model, large_model)
    pricing = config.pricing.get(decision.selected_model)

    base = dict(
        experiment_run_id=experiment_run_id,
        execution_index=planned_call.execution_index,
        strategy=planned_call.strategy,
        task_id=task.task_id,
        task_category=task.category,
        complexity_label=task.complexity_label,
        timestamp=datetime.now(timezone.utc).isoformat(),
        selected_model=decision.selected_model,
        model_tier=decision.model_tier,
        routing_reason=decision.routing_reason,
        matched_rules=list(decision.matched_rules),
        policy_version=decision.policy_version,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )

    create_kwargs = {
        "model": decision.selected_model,
        "messages": [{"role": "user", "content": task.prompt}],
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
    }
    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)

    if call_result is None:
        return RoutingMeasurement(**base, succeeded=False, attempts_used=attempts_used, error=error)

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
        input_cost, output_cost = cost_breakdown(tokens, pricing)
    else:
        input_cost, output_cost = None, None

    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    final_answer_char_count = len(call_result.message_content) if call_result.message_content is not None else None

    evaluation = evaluate_task_response(
        task.evaluation_type, call_result.message_content, task.expected_answer, task.evaluation_config, tokens.finish_reason
    )

    return RoutingMeasurement(
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
        system_fingerprint=call_result.system_fingerprint,
        final_answer_char_count=final_answer_char_count,
        total_latency_ms=call_result.performance.client_elapsed_ms,
        provider_reported_processing_ms=call_result.performance.provider_reported_processing_ms,
        estimated_input_cost_usd=input_cost,
        estimated_output_cost_usd=output_cost,
        estimated_cost_usd=tokens.estimated_cost_usd,
        evaluation_status=evaluation.status,
        answer_correct=evaluation.answer_correct,
        answer_complete=evaluation.answer_complete,
        evaluation_detail=evaluation.detail,
    )
