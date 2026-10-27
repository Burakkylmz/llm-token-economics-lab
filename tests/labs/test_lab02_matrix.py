from __future__ import annotations

import pytest

from labs.lab02_context_growth.experiments import CONTEXT_CONDITION_ORDER
from labs.lab02_context_growth.matrix import ContextMatrixError, build_context_plan


def test_default_matrix_is_3_conditions_times_2_repetitions_equals_6():
    plan = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2)
    assert len(plan) == 6


def test_all_mode_plus_multi_turn_equals_documented_10_call_default():
    plan = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2)
    multi_turn_turns = 4
    assert len(plan) + multi_turn_turns == 10


def test_single_condition_plan_is_exactly_repetitions_long():
    plan = build_context_plan(["irrelevant_context"], repetitions=3)
    assert len(plan) == 3
    assert all(c.condition_id == "irrelevant_context" for c in plan)


def test_empty_condition_list_yields_empty_plan():
    assert build_context_plan([], repetitions=2) == []


def test_unknown_condition_id_is_rejected():
    with pytest.raises(ContextMatrixError):
        build_context_plan(["not_a_real_condition"], repetitions=1)


def test_zero_repetitions_is_rejected():
    with pytest.raises(ContextMatrixError):
        build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=0)


def test_non_randomized_plan_preserves_condition_order():
    plan = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2, randomize_context_order=False)
    condition_sequence = [c.condition_id for c in plan]
    assert condition_sequence == [
        "minimal_context", "minimal_context",
        "expanded_context", "expanded_context",
        "irrelevant_context", "irrelevant_context",
    ]


def test_randomized_plan_is_deterministic_given_the_same_seed():
    plan_a = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2, randomize_context_order=True, randomization_seed=42)
    plan_b = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2, randomize_context_order=True, randomization_seed=42)
    assert [c.condition_id for c in plan_a] == [c.condition_id for c in plan_b]


def test_randomized_plan_with_different_seeds_can_differ():
    plan_a = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=5, randomize_context_order=True, randomization_seed=1)
    plan_b = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=5, randomize_context_order=True, randomization_seed=2)
    assert [c.condition_id for c in plan_a] != [c.condition_id for c in plan_b]


def test_randomization_never_changes_the_multiset_of_planned_calls():
    plan = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=2, randomize_context_order=True, randomization_seed=99)
    counts = {}
    for call in plan:
        counts[call.condition_id] = counts.get(call.condition_id, 0) + 1
    assert counts == {"minimal_context": 2, "expanded_context": 2, "irrelevant_context": 2}


def test_execution_index_is_1_based_and_sequential():
    plan = build_context_plan(list(CONTEXT_CONDITION_ORDER), repetitions=1)
    assert [c.execution_index for c in plan] == [1, 2, 3]
