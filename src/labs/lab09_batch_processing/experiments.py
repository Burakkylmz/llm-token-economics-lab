"""Lab09 experiments: synchronous vs asynchronous vs Groq Batch API processing.

Synchronous processing, Python asyncio/concurrent requests, and the
provider-managed Batch API are three distinct concepts:
  - sequential/concurrent (reused from Lab08) are THIS PROCESS making many
    real-time requests, one after another or overlapped;
  - the Batch API is the PROVIDER queuing and processing a whole file of
    requests asynchronously, on its own schedule (minutes to 7 days), at a
    discounted price -- this process only submits the job and polls for
    status, it is not doing the work itself.
The same PROMPTS are reused from Lab08 so all three approaches are directly
comparable on cost and total processing time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from groq import Groq

from lab_core.config import AppConfig, ModelPricing
from lab_core.metrics import ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab08_concurrency.experiments import PROMPTS  # shared workload, for apples-to-apples comparison
from labs.lab09_batch_processing.batch import build_batch_requests, write_batch_jsonl

__all__ = ["PROMPTS", "prepare_batch_input_file", "batch_output_to_results"]


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
