"""Lab08 experiments: sequential vs async/concurrent request execution.

All conditions send the exact same set of PROMPTS; only execution strategy
(sequential loop, unbounded asyncio.gather, or a semaphore-limited gather)
varies, so total wall-clock time and per-call latency are directly
comparable across conditions.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from groq import AsyncGroq

from lab_core.config import AppConfig
from lab_core.groq_client import call_async
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from lab_core.timing import StopwatchTimer
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab08_concurrency.concurrency import summarize_batch

PROMPTS = [f"What is {i} plus {i}? Answer with only the number." for i in range(1, 9)]


def _make_result(experiment_id: str, model: str, call, prompt_index: int, pricing) -> ExperimentResult:
    if pricing is not None:
        call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id=experiment_id,
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode="async" if "async" in experiment_id else "sync",
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
            results.append(_make_result("lab08_sequential", model, call, i, pricing))

    latencies = [r.performance.client_elapsed_ms for r in results if r.performance.client_elapsed_ms is not None]
    summary = summarize_batch(latencies, batch_timer.result.elapsed_ms if batch_timer.result else 0)
    return results, summary


async def run_concurrent(
    client: AsyncGroq,
    config: AppConfig,
    model: str,
    max_concurrent: int,
) -> tuple[list[ExperimentResult], dict]:
    """max_concurrent=len(PROMPTS) or higher behaves as "unbounded" concurrency."""
    pricing = config.pricing.get(model)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_call(i: int, prompt: str) -> ExperimentResult:
        async with semaphore:
            call = await call_async(client, model=model, messages=[{"role": "user", "content": prompt}])
        return _make_result("lab08_concurrent", model, call, i, pricing)

    with StopwatchTimer() as batch_timer:
        results = await asyncio.gather(*(bounded_call(i, p) for i, p in enumerate(PROMPTS)))

    latencies = [r.performance.client_elapsed_ms for r in results if r.performance.client_elapsed_ms is not None]
    summary = summarize_batch(latencies, batch_timer.result.elapsed_ms if batch_timer.result else 0)
    summary["max_concurrent_requests"] = max_concurrent
    return list(results), summary
