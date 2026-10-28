from __future__ import annotations

import pytest

from labs.lab08_concurrency.matrix import (
    EXTENDED_LEVELS,
    PILOT_LEVELS,
    build_plan,
    levels_for_profile,
    task_ids_for_profile,
)


def test_pilot_levels_are_one_and_three():
    assert levels_for_profile("pilot", default_levels=(1, 5)) == PILOT_LEVELS == (1, 3)


def test_extended_levels_are_fixed():
    assert levels_for_profile("extended", default_levels=(1, 5)) == EXTENDED_LEVELS == (1, 2, 3, 5)


def test_default_profile_uses_the_configured_levels():
    assert levels_for_profile("default", default_levels=(1, 5)) == (1, 5)
    assert levels_for_profile("default", default_levels=(1, 2, 4)) == (1, 2, 4)


def test_pilot_task_ids_are_four():
    assert len(task_ids_for_profile("pilot")) == 4


def test_default_and_extended_task_ids_are_eight():
    assert len(task_ids_for_profile("default")) == 8
    assert len(task_ids_for_profile("extended")) == 8


def test_pilot_plans_eight_calls():
    task_ids = task_ids_for_profile("pilot")
    plan = build_plan(task_ids, PILOT_LEVELS)
    assert len(plan) == 8


def test_default_plans_sixteen_calls():
    task_ids = task_ids_for_profile("default")
    plan = build_plan(task_ids, (1, 5))
    assert len(plan) == 16


def test_extended_plans_thirty_two_calls():
    task_ids = task_ids_for_profile("extended")
    plan = build_plan(task_ids, EXTENDED_LEVELS)
    assert len(plan) == 32


def test_every_level_gets_every_task():
    task_ids = ("t1", "t2")
    plan = build_plan(task_ids, (1, 5))
    for level in (1, 5):
        tasks_seen = {c.task_id for c in plan if c.max_concurrent == level}
        assert tasks_seen == set(task_ids)


def test_default_order_alternates_level_order_per_repetition():
    task_ids = ("t1",)
    plan = build_plan(task_ids, (1, 5), repetitions=2)
    rep1_levels = [c.max_concurrent for c in plan if c.repetition_index == 1]
    rep2_levels = [c.max_concurrent for c in plan if c.repetition_index == 2]
    assert rep1_levels == [1, 5]
    assert rep2_levels == [5, 1]


def test_seeded_randomized_order_is_reproducible():
    task_ids = ("t1", "t2")
    plan_a = build_plan(task_ids, (1, 2, 3, 5), repetitions=3, randomize=True, seed=7)
    plan_b = build_plan(task_ids, (1, 2, 3, 5), repetitions=3, randomize=True, seed=7)
    assert [(c.repetition_index, c.max_concurrent, c.task_id) for c in plan_a] == [
        (c.repetition_index, c.max_concurrent, c.task_id) for c in plan_b
    ]


def test_execution_index_is_sequential_from_start():
    plan = build_plan(("t1",), (1, 5), start_execution_index=5)
    assert [c.execution_index for c in plan] == [5, 6]


def test_zero_repetitions_rejected():
    with pytest.raises(ValueError):
        build_plan(("t1",), (1,), repetitions=0)
