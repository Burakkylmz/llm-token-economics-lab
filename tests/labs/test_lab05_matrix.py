from __future__ import annotations

from labs.lab05_model_routing.matrix import (
    STRATEGY_ALWAYS_LARGE,
    STRATEGY_ALWAYS_SMALL,
    STRATEGY_ROUTER,
    build_plan_for_profile,
    build_routing_plan,
    strategies_for_profile,
    task_ids_for,
)


def test_pilot_profile_plans_eight_calls():
    plan = build_plan_for_profile("pilot")
    assert len(plan) == 8


def test_default_profile_plans_twelve_calls():
    plan = build_plan_for_profile("default")
    assert len(plan) == 12


def test_extended_profile_plans_eighteen_calls():
    plan = build_plan_for_profile("extended")
    assert len(plan) == 18


def test_default_and_extended_use_the_same_six_tasks():
    assert task_ids_for("default") == task_ids_for("extended")
    assert len(task_ids_for("default")) == 6


def test_pilot_uses_four_tasks():
    assert len(task_ids_for("pilot")) == 4


def test_strategies_for_pilot_and_default_are_two():
    assert strategies_for_profile("pilot") == (STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER)
    assert strategies_for_profile("default") == (STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER)


def test_strategies_for_extended_adds_always_large():
    strategies = strategies_for_profile("extended")
    assert STRATEGY_ALWAYS_LARGE in strategies
    assert len(strategies) == 3


def test_every_task_gets_every_strategy_exactly_once():
    task_ids = ["t1", "t2", "t3"]
    strategies = [STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER]
    plan = build_routing_plan(task_ids, strategies)
    for task_id in task_ids:
        strategies_seen = {c.strategy for c in plan if c.task_id == task_id}
        assert strategies_seen == set(strategies)


def test_execution_order_alternates_strategy_order_per_task():
    task_ids = ["t1", "t2"]
    strategies = [STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER]
    plan = build_routing_plan(task_ids, strategies)
    task1_calls = [c.strategy for c in plan if c.task_id == "t1"]
    task2_calls = [c.strategy for c in plan if c.task_id == "t2"]
    assert task1_calls == [STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER]
    assert task2_calls == [STRATEGY_ROUTER, STRATEGY_ALWAYS_SMALL]


def test_execution_index_is_sequential_from_start():
    plan = build_routing_plan(["t1", "t2"], [STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER], start_execution_index=5)
    assert [c.execution_index for c in plan] == [5, 6, 7, 8]


def test_plan_is_reproducible_across_calls():
    plan_a = build_plan_for_profile("default")
    plan_b = build_plan_for_profile("default")
    assert [(c.execution_index, c.strategy, c.task_id) for c in plan_a] == [
        (c.execution_index, c.strategy, c.task_id) for c in plan_b
    ]
