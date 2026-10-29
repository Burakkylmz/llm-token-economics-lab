"""Builds Lab09's real-time execution plan: sequential (max_concurrent=1)
first, then concurrent (max_concurrent=config value), over the SAME
ordered task list -- spec Part 5's "real-time count is the sum of
sequential and concurrent calls, with one repetition."
"""

from __future__ import annotations

from labs.lab09_batch_processing.models import CONDITION_CONCURRENT, CONDITION_SEQUENTIAL, PlannedRealtimeCall


def build_realtime_plan(task_ids: tuple[str, ...], max_concurrent: int) -> list[PlannedRealtimeCall]:
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be a positive integer")

    plan: list[PlannedRealtimeCall] = []
    index = 1
    for task_index, task_id in enumerate(task_ids):
        plan.append(
            PlannedRealtimeCall(
                execution_index=index, condition=CONDITION_SEQUENTIAL,
                max_concurrent=1, task_id=task_id, task_index=task_index,
            )
        )
        index += 1
    for task_index, task_id in enumerate(task_ids):
        plan.append(
            PlannedRealtimeCall(
                execution_index=index, condition=CONDITION_CONCURRENT,
                max_concurrent=max_concurrent, task_id=task_id, task_index=task_index,
            )
        )
        index += 1
    return plan
