from __future__ import annotations

import pytest

from labs.lab09_batch_processing.matrix import build_realtime_plan
from labs.lab09_batch_processing.models import CONDITION_CONCURRENT, CONDITION_SEQUENTIAL


def test_build_realtime_plan_sequential_runs_before_concurrent():
    task_ids = ("task_001", "task_002", "task_003")
    plan = build_realtime_plan(task_ids, max_concurrent=5)

    assert len(plan) == 6
    assert [c.condition for c in plan[:3]] == [CONDITION_SEQUENTIAL] * 3
    assert [c.condition for c in plan[3:]] == [CONDITION_CONCURRENT] * 3


def test_build_realtime_plan_uses_same_task_order_in_both_conditions():
    task_ids = ("task_001", "task_002", "task_003")
    plan = build_realtime_plan(task_ids, max_concurrent=5)
    sequential_ids = [c.task_id for c in plan if c.condition == CONDITION_SEQUENTIAL]
    concurrent_ids = [c.task_id for c in plan if c.condition == CONDITION_CONCURRENT]
    assert sequential_ids == list(task_ids)
    assert concurrent_ids == list(task_ids)


def test_build_realtime_plan_assigns_correct_max_concurrent():
    plan = build_realtime_plan(("task_001",), max_concurrent=7)
    sequential_call = next(c for c in plan if c.condition == CONDITION_SEQUENTIAL)
    concurrent_call = next(c for c in plan if c.condition == CONDITION_CONCURRENT)
    assert sequential_call.max_concurrent == 1
    assert concurrent_call.max_concurrent == 7


def test_build_realtime_plan_execution_indices_are_sequential_and_unique():
    plan = build_realtime_plan(("task_001", "task_002"), max_concurrent=3)
    indices = [c.execution_index for c in plan]
    assert indices == list(range(1, len(plan) + 1))


def test_build_realtime_plan_rejects_non_positive_concurrency():
    with pytest.raises(ValueError):
        build_realtime_plan(("task_001",), max_concurrent=0)
