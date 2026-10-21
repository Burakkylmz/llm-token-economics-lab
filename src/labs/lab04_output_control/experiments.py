"""Lab04 experiments: output token limits and invocation-count optimization.

Two independent experiments:
  1. output_limit_sweep -- same prompt, varying `max_tokens`, to show how
     completion_tokens/cost/latency change with the output cap.
  2. invocation_comparison -- answering several questions via one combined
     call vs. one call per question, to show the fixed per-call overhead
     (prompt repetition, request latency) of unnecessary extra invocations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from labs.lab01_token_anatomy.cost import calculate_cost

VERBOSE_PROMPT = "Explain, in as much detail as you find useful, how exponential backoff works."

QUESTIONS = [
    "What is the capital of Japan?",
    "What is the boiling point of water in Celsius at sea level?",
    "Who wrote the play 'Hamlet'?",
]


def _make_result(experiment_id: str, model: str, call, extra_params: dict, pricing) -> ExperimentResult:
    if pricing is not None:
        call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id=experiment_id,
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode="sync",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_parameters=extra_params,
        ),
        tokens_and_cost=call.tokens_and_cost,
        performance=call.performance,
    )


def run_output_limit_sweep(
    client: Groq,
    config: AppConfig,
    model: str,
    max_tokens_values: list[int],
) -> list[ExperimentResult]:
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for max_tokens in max_tokens_values:
        call = call_sync(
            client,
            model=model,
            messages=[{"role": "user", "content": VERBOSE_PROMPT}],
            max_tokens=max_tokens,
        )
        results.append(
            _make_result("lab04_output_limit_sweep", model, call, {"max_tokens": max_tokens}, pricing)
        )

    return results


def run_single_invocation(client: Groq, config: AppConfig, model: str) -> list[ExperimentResult]:
    """Answer all QUESTIONS in exactly one call."""
    pricing = config.pricing.get(model)
    combined_prompt = "Answer each question on its own line, in order:\n" + "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(QUESTIONS)
    )
    call = call_sync(client, model=model, messages=[{"role": "user", "content": combined_prompt}])
    return [_make_result("lab04_single_invocation", model, call, {"question_count": len(QUESTIONS)}, pricing)]


def run_multiple_invocations(client: Groq, config: AppConfig, model: str) -> list[ExperimentResult]:
    """Answer each question in QUESTIONS with its own separate call."""
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for i, question in enumerate(QUESTIONS):
        call = call_sync(client, model=model, messages=[{"role": "user", "content": question}])
        results.append(_make_result("lab04_multiple_invocations", model, call, {"question_index": i}, pricing))

    return results


def summarize_invocation_comparison(
    single_call_results: list[ExperimentResult],
    multiple_call_results: list[ExperimentResult],
) -> dict:
    """Aggregate totals that make the two approaches directly comparable."""

    def totals(results: list[ExperimentResult]) -> dict:
        prompt_tokens = sum(r.tokens_and_cost.prompt_tokens or 0 for r in results)
        completion_tokens = sum(r.tokens_and_cost.completion_tokens or 0 for r in results)
        costs = [r.tokens_and_cost.estimated_cost_usd for r in results if r.tokens_and_cost.estimated_cost_usd is not None]
        return {
            "api_call_count": len(results),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_estimated_cost_usd": sum(costs) if costs else None,
        }

    return {
        "single_invocation": totals(single_call_results),
        "multiple_invocations": totals(multiple_call_results),
    }
