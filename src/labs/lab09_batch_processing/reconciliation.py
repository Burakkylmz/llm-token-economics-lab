"""Reconciles Batch API output/error rows against the planned task list,
by `custom_id` -- never by JSONL line order (spec Part 8).

Handles, without silently discarding any record:
  - missing custom_id / duplicate custom_id / unknown (unplanned) custom_id
  - missing output row (task never came back at all)
  - a failed request row (error present)
  - an empty or malformed JSONL line
  - a response body missing `usage`
  - a model mismatch between the request and the planned model

`completed` provider job status does NOT by itself imply every row
succeeded or was even returned -- this module's counts are the only
source of truth for that.
"""

from __future__ import annotations

from typing import Optional

from lab_core.config import ModelPricing
from labs.lab09_batch_processing.datasets import GoldenTask
from labs.lab09_batch_processing.evaluation import evaluate_answer
from labs.lab09_batch_processing.models import (
    RECON_COMPLETED,
    RECON_DUPLICATE,
    RECON_FAILED,
    RECON_MALFORMED,
    RECON_MISSING,
    RECON_UNKNOWN_ID,
    BatchRequestMeasurement,
    ReconciliationCounts,
)
from labs.lab09_batch_processing.pricing import calculate_batch_cost, calculate_standard_cost


def reconcile_batch_results(
    experiment_run_id: str,
    planned_tasks: list[GoldenTask],
    output_rows: list[dict],
    error_rows: list[dict],
    model: str,
    pricing: Optional[ModelPricing],
) -> tuple[list[BatchRequestMeasurement], ReconciliationCounts]:
    tasks_by_id = {t.task_id: t for t in planned_tasks}
    planned_ids = set(tasks_by_id.keys())

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    by_id: dict[str, dict] = {}
    unexpected_ids: set[str] = set()
    malformed_count = 0

    for row in output_rows + error_rows:
        if "_malformed_line" in row:
            malformed_count += 1
            continue
        custom_id = row.get("custom_id")
        if not custom_id:
            malformed_count += 1
            continue
        if custom_id in seen_ids:
            duplicate_ids.add(custom_id)
            continue
        seen_ids.add(custom_id)
        if custom_id not in planned_ids:
            unexpected_ids.add(custom_id)
            continue
        by_id[custom_id] = row

    measurements: list[BatchRequestMeasurement] = []
    counts = ReconciliationCounts(planned_requests=len(planned_tasks))

    for task in planned_tasks:
        row = by_id.get(task.task_id)
        if row is None:
            counts.missing_requests += 1
            measurements.append(
                BatchRequestMeasurement(
                    experiment_run_id=experiment_run_id, task_id=task.task_id, category=task.category,
                    custom_id=task.task_id, reconciliation_status=RECON_MISSING,
                )
            )
            continue

        counts.returned_output_rows += 1
        error = row.get("error")
        response = row.get("response") or {}
        body = response.get("body") or {}
        usage = body.get("usage") or {}
        message = (body.get("choices") or [{}])[0].get("message") or {}
        response_text = message.get("content")

        if error is not None:
            counts.failed_requests += 1
            measurements.append(
                BatchRequestMeasurement(
                    experiment_run_id=experiment_run_id, task_id=task.task_id, category=task.category,
                    custom_id=task.task_id, reconciliation_status=RECON_FAILED,
                    error_message=error.get("message") if isinstance(error, dict) else str(error),
                )
            )
            continue

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        standard_cost = None
        batch_cost = None
        if pricing is not None and prompt_tokens is not None and completion_tokens is not None:
            from lab_core.metrics import TokenCostMetrics

            standard_cost = calculate_standard_cost(
                TokenCostMetrics(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens), pricing
            )
            batch_cost = calculate_batch_cost(prompt_tokens, completion_tokens, pricing)

        evaluation = evaluate_answer(response_text, task.expected_answer, task.evaluation_type)
        if evaluation.answer_correct is True:
            counts.correct_answers += 1
        elif evaluation.answer_correct is False:
            counts.incorrect_answers += 1
        counts.completed_requests += 1

        measurements.append(
            BatchRequestMeasurement(
                experiment_run_id=experiment_run_id, task_id=task.task_id, category=task.category,
                custom_id=task.task_id, reconciliation_status=RECON_COMPLETED,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens,
                estimated_standard_cost_usd=standard_cost, estimated_batch_cost_usd=batch_cost,
                evaluation_status=evaluation.status, answer_correct=evaluation.answer_correct,
            )
        )

    counts.duplicate_result_ids = len(duplicate_ids)
    counts.unexpected_result_ids = len(unexpected_ids)
    if malformed_count:
        measurements.append(
            BatchRequestMeasurement(
                experiment_run_id=experiment_run_id, task_id="_malformed", category="_malformed",
                custom_id="_malformed", reconciliation_status=RECON_MALFORMED,
                error_message=f"{malformed_count} malformed/unkeyed JSONL line(s)",
            )
        )
    for unexpected_id in sorted(unexpected_ids):
        measurements.append(
            BatchRequestMeasurement(
                experiment_run_id=experiment_run_id, task_id=unexpected_id, category="_unknown",
                custom_id=unexpected_id, reconciliation_status=RECON_UNKNOWN_ID,
            )
        )

    return measurements, counts
