from __future__ import annotations

import pytest

from labs.lab06_streaming.matrix import (
    CONDITION_ORDER,
    build_single_condition_plan,
    build_streaming_plan,
    pairs_for_profile,
)
from labs.lab06_streaming.models import CONDITION_NON_STREAMING, CONDITION_STREAMING


def test_pilot_profile_is_two_pairs():
    assert pairs_for_profile("pilot", default_pairs=5) == 2


def test_extended_profile_is_eight_pairs():
    assert pairs_for_profile("extended", default_pairs=5) == 8


def test_default_profile_uses_the_configured_pair_count():
    assert pairs_for_profile("default", default_pairs=5) == 5
    assert pairs_for_profile("default", default_pairs=3) == 3


def test_pilot_plans_four_calls():
    plan = build_streaming_plan(pairs_for_profile("pilot", 5))
    assert len(plan) == 4


def test_default_plans_ten_calls():
    plan = build_streaming_plan(pairs_for_profile("default", 5))
    assert len(plan) == 10


def test_extended_plans_sixteen_calls():
    plan = build_streaming_plan(pairs_for_profile("extended", 5))
    assert len(plan) == 16


def test_every_pair_has_exactly_one_streaming_and_one_non_streaming_call():
    plan = build_streaming_plan(5)
    by_pair: dict[int, set[str]] = {}
    for call in plan:
        by_pair.setdefault(call.pair_id, set()).add(call.condition)
    for conditions in by_pair.values():
        assert conditions == set(CONDITION_ORDER)


def test_default_order_alternates_which_condition_goes_first():
    plan = build_streaming_plan(3)
    pair1 = [c.condition for c in plan if c.pair_id == 1]
    pair2 = [c.condition for c in plan if c.pair_id == 2]
    pair3 = [c.condition for c in plan if c.pair_id == 3]
    assert pair1 == [CONDITION_NON_STREAMING, CONDITION_STREAMING]
    assert pair2 == [CONDITION_STREAMING, CONDITION_NON_STREAMING]
    assert pair3 == [CONDITION_NON_STREAMING, CONDITION_STREAMING]


def test_seeded_randomized_order_is_reproducible():
    plan_a = build_streaming_plan(5, randomize=True, seed=7)
    plan_b = build_streaming_plan(5, randomize=True, seed=7)
    assert [(c.pair_id, c.condition) for c in plan_a] == [(c.pair_id, c.condition) for c in plan_b]


def test_execution_index_is_sequential_from_start():
    plan = build_streaming_plan(2, start_execution_index=5)
    assert [c.execution_index for c in plan] == [5, 6, 7, 8]


def test_zero_pairs_rejected():
    with pytest.raises(ValueError):
        build_streaming_plan(0)


def test_single_condition_plan_has_one_call_per_pair_slot():
    plan = build_single_condition_plan(CONDITION_STREAMING, 4)
    assert len(plan) == 4
    assert all(c.condition == CONDITION_STREAMING for c in plan)
    assert [c.pair_id for c in plan] == [1, 2, 3, 4]
