"""Lab05 experiments: fixed-model baseline vs rule-based routing over a golden dataset."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab05_model_routing.golden_dataset import GOLDEN_DATASET, GoldenExample, score
from labs.lab05_model_routing.routing import choose_model


def _make_result(experiment_id: str, model: str, call, example: GoldenExample, passed: bool, pricing) -> ExperimentResult:
    if pricing is not None:
        call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id=experiment_id,
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode="sync",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_parameters={
                "task_id": example.task_id,
                "ground_truth_complexity": example.complexity,
                "passed": passed,
            },
        ),
        tokens_and_cost=call.tokens_and_cost,
        performance=call.performance,
    )


def run_fixed_model_baseline(
    client: Groq,
    config: AppConfig,
    model: str,
) -> list[ExperimentResult]:
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for example in GOLDEN_DATASET:
        call = call_sync(client, model=model, messages=[{"role": "user", "content": example.prompt}])
        passed = score(example, call.raw_response.choices[0].message.content)
        results.append(_make_result("lab05_fixed_model_baseline", model, call, example, passed, pricing))

    return results


def run_rule_based_routing(
    client: Groq,
    config: AppConfig,
) -> list[ExperimentResult]:
    small_model = config.lab05.small_model
    large_model = config.lab05.large_model
    results: list[ExperimentResult] = []

    for example in GOLDEN_DATASET:
        chosen_model = choose_model(example.prompt, small_model, large_model)
        pricing = config.pricing.get(chosen_model)
        call = call_sync(client, model=chosen_model, messages=[{"role": "user", "content": example.prompt}])
        passed = score(example, call.raw_response.choices[0].message.content)
        results.append(_make_result("lab05_rule_based_routing", chosen_model, call, example, passed, pricing))

    return results


def summarize_quality_cost_latency(results: list[ExperimentResult]) -> dict:
    passed_count = sum(1 for r in results if r.metadata.model_parameters.get("passed"))
    costs = [r.tokens_and_cost.estimated_cost_usd for r in results if r.tokens_and_cost.estimated_cost_usd is not None]
    latencies = [r.performance.client_elapsed_ms for r in results if r.performance.client_elapsed_ms is not None]
    return {
        "accuracy": passed_count / len(results) if results else None,
        "total_estimated_cost_usd": sum(costs) if costs else None,
        "avg_client_elapsed_ms": sum(latencies) / len(latencies) if latencies else None,
    }
