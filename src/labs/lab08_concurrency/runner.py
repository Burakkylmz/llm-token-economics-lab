"""The instrumented, semaphore-based async execution engine for Lab08.

A SINGLE engine (`run_concurrency_level`) drives every concurrency level,
including the "sequential" condition -- sequential is simply the
`max_concurrent=1` case of the exact same semaphore-admission model, so
`queue_wait_ms` is a real, measured value in both conditions (never
hardcoded to 0 for one of them), and `execution_mode` is derived from
`max_concurrent` (see labs.lab08_concurrency.models.execution_mode_for),
never set independently of it.

Reuses, rather than reimplements, this project's existing shared/Lab01
infrastructure: lab_core.groq_client.call_async (which already measures
`client_elapsed_ms` -- used directly as `request_service_latency_ms`,
since that measurement's own start/end boundary already excludes queue
wait: it begins only once this engine has invoked it, strictly after the
semaphore permit was acquired), labs.lab01_token_anatomy.cost.calculate_cost,
labs.lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens.

Peak in-flight concurrency is tracked with a plain module-level counter,
not a lock: asyncio tasks only yield control at `await` points, so the
increment (right after the semaphore permit is acquired, before the first
`await` of the actual request) and the decrement (right after the request
finishes) are each atomic with respect to other tasks.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from groq import APIConnectionError, APITimeoutError, AsyncGroq, GroqError, RateLimitError

from lab_core.config import ModelPricing
from lab_core.groq_client import call_async
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab08_concurrency.evaluation import evaluate_numeric_answer
from labs.lab08_concurrency.experiments import WORKLOAD_BY_ID
from labs.lab08_concurrency.models import (
    EVENT_PERMIT_ACQUIRED,
    EVENT_QUEUED,
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_STARTED,
    STATUS_CANCELLED,
    STATUS_CONNECTION_ERROR,
    STATUS_PROVIDER_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_UNKNOWN_ERROR,
    ConcurrencyMeasurement,
    PlannedConcurrencyCall,
    RequestTimelineEvent,
    execution_mode_for,
)


async def _execute_request(client: AsyncGroq, model: str, prompt: str, reasoning_effort: str, max_completion_tokens: int):
    """Returns (call_result_or_None, status, error_message_or_None). Never
    raises except for a genuine asyncio.CancelledError, which must keep
    propagating (cooperative cancellation is never silently absorbed at
    this layer)."""
    try:
        call_result = await call_async(
            client, model=model, messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort, max_completion_tokens=max_completion_tokens,
        )
        return call_result, STATUS_SUCCESS, None
    except RateLimitError as exc:
        return None, STATUS_RATE_LIMITED, f"{type(exc).__name__}: {exc}"
    except APITimeoutError as exc:
        return None, STATUS_TIMEOUT, f"{type(exc).__name__}: {exc}"
    except APIConnectionError as exc:
        return None, STATUS_CONNECTION_ERROR, f"{type(exc).__name__}: {exc}"
    except GroqError as exc:
        return None, STATUS_PROVIDER_ERROR, f"{type(exc).__name__}: {exc}"
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any other, unanticipated failure
        return None, STATUS_UNKNOWN_ERROR, f"{type(exc).__name__}: {exc}"


def _build_measurement(
    base: dict, call_result, status: str, error: Optional[str], pricing: Optional[ModelPricing],
    expected_answer: str, queue_wait_ms: float, request_service_latency_ms: Optional[float], task_end_to_end_latency_ms: float,
) -> ConcurrencyMeasurement:
    if call_result is None:
        return ConcurrencyMeasurement(
            **base, request_status=status, error=error, queue_wait_ms=queue_wait_ms,
            task_end_to_end_latency_ms=task_end_to_end_latency_ms,
        )

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    evaluation = evaluate_numeric_answer(call_result.message_content, expected_answer, tokens.finish_reason)

    return ConcurrencyMeasurement(
        **base,
        request_status=STATUS_SUCCESS,
        error=None,
        queue_wait_ms=queue_wait_ms,
        request_service_latency_ms=request_service_latency_ms,
        task_end_to_end_latency_ms=task_end_to_end_latency_ms,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=tokens.finish_reason,
        output_truncated=(tokens.finish_reason == "length") if tokens.finish_reason is not None else None,
        estimated_cost_usd=tokens.estimated_cost_usd,
        evaluation_status=evaluation.status,
        answer_correct=evaluation.answer_correct,
        answer_complete=evaluation.answer_complete,
    )


async def run_concurrency_level(
    client: AsyncGroq,
    model: str,
    pricing: Optional[ModelPricing],
    reasoning_effort: str,
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_calls: list[PlannedConcurrencyCall],
) -> tuple[list[ConcurrencyMeasurement], list[RequestTimelineEvent], float, int]:
    """Runs every call in `planned_calls` (all sharing the same
    max_concurrent/repetition_index) under one semaphore. Returns
    (measurements, timeline_events, batch_wall_clock_ms, observed_peak_in_flight)."""
    if not planned_calls:
        return [], [], 0.0, 0

    max_concurrent = planned_calls[0].max_concurrent
    repetition_index = planned_calls[0].repetition_index
    execution_mode = execution_mode_for(max_concurrent)
    semaphore = asyncio.Semaphore(max_concurrent)
    events: list[RequestTimelineEvent] = []
    in_flight = 0
    peak_in_flight = 0
    start_ns = time.perf_counter_ns()

    def _rel_ms(ns: int) -> float:
        return (ns - start_ns) / 1_000_000

    def _record(execution_index: int, task_id: str, event_type: str, ns: int) -> None:
        events.append(
            RequestTimelineEvent(
                execution_index=execution_index, repetition_index=repetition_index, max_concurrent=max_concurrent,
                task_id=task_id, event_type=event_type, relative_ms=_rel_ms(ns),
            )
        )

    async def _run_one(planned: PlannedConcurrencyCall) -> ConcurrencyMeasurement:
        nonlocal in_flight, peak_in_flight
        task = WORKLOAD_BY_ID[planned.task_id]
        queued_ns = time.perf_counter_ns()
        _record(planned.execution_index, planned.task_id, EVENT_QUEUED, queued_ns)

        base = dict(
            experiment_run_id=experiment_run_id, execution_index=planned.execution_index,
            repetition_index=planned.repetition_index, task_id=planned.task_id, max_concurrent=max_concurrent,
            execution_mode=execution_mode, timestamp=datetime.now(timezone.utc).isoformat(), model=model,
            reasoning_effort=reasoning_effort, max_completion_tokens=max_completion_tokens,
        )

        async with semaphore:
            permit_ns = time.perf_counter_ns()
            _record(planned.execution_index, planned.task_id, EVENT_PERMIT_ACQUIRED, permit_ns)
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            request_start_ns = time.perf_counter_ns()
            _record(planned.execution_index, planned.task_id, EVENT_REQUEST_STARTED, request_start_ns)

            try:
                call_result, status, error = await _execute_request(
                    client, model, task.prompt, reasoning_effort, max_completion_tokens
                )
            except asyncio.CancelledError:
                call_result, status, error = None, STATUS_CANCELLED, "Task was cancelled"
            finally:
                in_flight -= 1

            request_end_ns = time.perf_counter_ns()
            _record(planned.execution_index, planned.task_id, EVENT_REQUEST_COMPLETED, request_end_ns)

        queue_wait_ms = (permit_ns - queued_ns) / 1_000_000
        task_end_to_end_latency_ms = (request_end_ns - queued_ns) / 1_000_000
        request_service_latency_ms = (
            call_result.performance.client_elapsed_ms if call_result is not None else None
        )

        return _build_measurement(
            base, call_result, status, error, pricing, task.expected_answer,
            queue_wait_ms, request_service_latency_ms, task_end_to_end_latency_ms,
        )

    measurements = await asyncio.gather(*(_run_one(p) for p in planned_calls))
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    return list(measurements), events, elapsed_ms, peak_in_flight
