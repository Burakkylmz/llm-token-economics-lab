"""Lab09 experiments: synchronous vs asynchronous vs Groq Batch API processing.

Synchronous processing, Python asyncio/concurrent requests, and the
provider-managed Batch API are three distinct concepts:
  - sequential/concurrent are THIS PROCESS making many real-time requests,
    one after another or overlapped;
  - the Batch API is the PROVIDER queuing and processing a whole file of
    requests asynchronously, on its own schedule (minutes to 7 days), at a
    discounted price -- this process only submits the job and polls for
    status, it is not doing the work itself.

PROMPTS/run_sequential/run_concurrent are Lab09's OWN, self-contained
fixed workload -- previously imported from Lab08, but Lab08's own
experiments module was redesigned around a structured ConcurrencyTask
workload with expected answers for correctness evaluation, which is not
what Lab09's simple sequential-vs-concurrent-vs-batch cost/latency
comparison needs. Keeping this workload local to Lab09 avoids coupling
two labs that explore different questions.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from groq import AsyncGroq, Groq

from lab_core.config import AppConfig, ModelPricing
from lab_core.groq_client import call_async
from lab_core.metrics import ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from lab_core.timing import StopwatchTimer
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab09_batch_processing.batch import build_batch_requests, write_batch_jsonl

__all__ = ["PROMPTS", "run_sequential", "run_concurrent", "prepare_batch_input_file", "batch_output_to_results"]

PROMPTS = [f"What is {i} plus {i}? Answer with only the number." for i in range(1, 9)]


def _percentile(sorted_values: list[float], fraction: float) -> Optional[float]:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _summarize_batch(latencies_ms: list[float], batch_elapsed_ms: float) -> dict:
    sorted_latencies = sorted(latencies_ms)
    return {
        "call_count": len(latencies_ms),
        "batch_elapsed_ms": batch_elapsed_ms,
        "mean_latency_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
        "p50_latency_ms": _percentile(sorted_latencies, 0.50),
        "p95_latency_ms": _percentile(sorted_latencies, 0.95),
    }


def _make_result(experiment_id: str, execution_mode: str, model: str, call, prompt_index: int, pricing: Optional[ModelPricing]) -> ExperimentResult:
    if pricing is not None:
        call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id=experiment_id,
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode=execution_mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_parameters={"prompt_index": prompt_index},
        ),
        tokens_and_cost=call.tokens_and_cost,
        performance=call.performance,
    )


async def run_sequential(client: AsyncGroq, config: AppConfig, model: str) -> tuple[list[ExperimentResult], dict]:
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    with StopwatchTimer() as batch_timer:
        for i, prompt in enumerate(PROMPTS):
            call = await call_async(client, model=model, messages=[{"role": "user", "content": prompt}])
            results.append(_make_result("lab09_sequential", "async_sequential", model, call, i, pricing))

    latencies = [r.performance.client_elapsed_ms for r in results if r.performance.client_elapsed_ms is not None]
    summary = _summarize_batch(latencies, batch_timer.result.elapsed_ms if batch_timer.result else 0.0)
    return results, summary


async def run_concurrent(client: AsyncGroq, config: AppConfig, model: str, max_concurrent: int) -> tuple[list[ExperimentResult], dict]:
    """max_concurrent=len(PROMPTS) or higher behaves as "unbounded" concurrency."""
    pricing = config.pricing.get(model)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_call(i: int, prompt: str) -> ExperimentResult:
        async with semaphore:
            call = await call_async(client, model=model, messages=[{"role": "user", "content": prompt}])
        return _make_result("lab09_async", "async_concurrent", model, call, i, pricing)

    with StopwatchTimer() as batch_timer:
        results = await asyncio.gather(*(bounded_call(i, p) for i, p in enumerate(PROMPTS)))

    latencies = [r.performance.client_elapsed_ms for r in results if r.performance.client_elapsed_ms is not None]
    summary = _summarize_batch(latencies, batch_timer.result.elapsed_ms if batch_timer.result else 0.0)
    summary["max_concurrent_requests"] = max_concurrent
    return list(results), summary


def prepare_batch_input_file(model: str, output_path: Path) -> Path:
    requests = build_batch_requests(PROMPTS, model)
    write_batch_jsonl(requests, output_path)
    return output_path


def batch_output_to_results(parsed_lines: list[dict], model: str, pricing: Optional[ModelPricing]) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for line in parsed_lines:
        response = line.get("response") or {}
        body = response.get("body") or {}
        usage = body.get("usage") or {}
        error = line.get("error")

        tokens = TokenCostMetrics(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            api_call_count=1,
        )
        if pricing is not None:
            tokens.estimated_cost_usd = calculate_cost(tokens, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id="lab09_batch_api",
                    run_id=str(line.get("custom_id", uuid.uuid4())),
                    model_id=model,
                    execution_mode="batch",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error=error.get("message") if isinstance(error, dict) else None,
                ),
                tokens_and_cost=tokens,
                performance=PerformanceMetrics(),
            )
        )

    return results
