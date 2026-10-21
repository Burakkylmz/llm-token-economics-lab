"""Lab06 experiments: streaming vs non-streaming, TTFT and perceived latency.

Two separate calls are used deliberately:
  - a non-streaming call, to get authoritative token usage (needed for
    output_tokens_per_second) from the final response's `usage` field;
  - a streaming call, purely to measure time-to-first-token (the first
    network chunk) and time-to-first-visible-content (the first chunk
    whose delta actually carries visible text) -- these are two different
    timestamps and must not be collapsed into one.

Usage is intentionally NOT extracted from the streaming response: Groq's
documented support for a usage field on the final stream chunk was not
confirmed during this project's research, so this code does not assume it.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync, call_stream
from lab_core.metrics import ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from labs.lab01_token_anatomy.cost import calculate_cost

PROMPT = "Write a short, four-sentence story about a robot who learns to paint."


def run_non_streaming(client: Groq, config: AppConfig, model: str) -> ExperimentResult:
    pricing = config.pricing.get(model)
    call = call_sync(client, model=model, messages=[{"role": "user", "content": PROMPT}])
    if pricing is not None:
        call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

    performance = call.performance
    if call.tokens_and_cost.completion_tokens and performance.client_elapsed_ms:
        performance.output_tokens_per_second = call.tokens_and_cost.completion_tokens / (
            performance.client_elapsed_ms / 1000
        )

    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id="lab06_non_streaming",
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode="sync",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        tokens_and_cost=call.tokens_and_cost,
        performance=performance,
    )


def run_streaming(client: Groq, config: AppConfig, model: str) -> ExperimentResult:
    handle = call_stream(client, model=model, messages=[{"role": "user", "content": PROMPT}])

    ttft_ms: float | None = None
    first_visible_content_ms: float | None = None
    chunk_count = 0
    last_chunk_ns = handle.start_perf_counter_ns

    for chunk in handle.stream:
        now_ns = time.perf_counter_ns()
        chunk_count += 1
        last_chunk_ns = now_ns
        if ttft_ms is None:
            ttft_ms = (now_ns - handle.start_perf_counter_ns) / 1_000_000

        content = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
        if content and first_visible_content_ms is None:
            first_visible_content_ms = (now_ns - handle.start_perf_counter_ns) / 1_000_000

    total_elapsed_ms = (last_chunk_ns - handle.start_perf_counter_ns) / 1_000_000

    performance = PerformanceMetrics(
        client_elapsed_ms=total_elapsed_ms,
        ttft_ms=ttft_ms,
        first_visible_content_ms=first_visible_content_ms,
    )

    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id="lab06_streaming",
            run_id=str(uuid.uuid4()),
            model_id=model,
            execution_mode="stream",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_parameters={"chunk_count": chunk_count},
        ),
        # Not fabricated: usage is not reliably available on a stream, so every
        # token/cost field stays None rather than being guessed from chunk counts.
        tokens_and_cost=TokenCostMetrics(api_call_count=1),
        performance=performance,
    )
