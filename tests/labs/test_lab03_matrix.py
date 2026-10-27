from __future__ import annotations

import pytest

from labs.lab03_prompt_caching.experiments import CONDITION_CACHE_UNFRIENDLY, CONDITION_ORDER, CONDITION_REPEATED_PREFIX
from labs.lab03_prompt_caching.matrix import CacheMatrixError, build_cache_plan


def test_default_matrix_is_six_calls():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=3)
    assert len(plan) == 6


def test_minimal_pilot_is_four_calls():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=2)
    assert len(plan) == 4


def test_extended_profile_is_eight_calls():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=4)
    assert len(plan) == 8


def test_single_condition_plan_has_no_hidden_calls():
    plan = build_cache_plan([CONDITION_REPEATED_PREFIX], repetitions=3)
    assert len(plan) == 3
    assert all(c.condition_id == CONDITION_REPEATED_PREFIX for c in plan)


def test_unknown_condition_rejected():
    with pytest.raises(CacheMatrixError):
        build_cache_plan(["not_a_condition"], repetitions=3)


def test_zero_repetitions_rejected():
    with pytest.raises(CacheMatrixError):
        build_cache_plan(list(CONDITION_ORDER), repetitions=0)


def test_plan_is_paired_and_interleaved_by_default():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=3)
    condition_sequence = [c.condition_id for c in plan]
    # NOT all-of-one-condition-then-all-of-the-other (the old bug) --
    # each pair is adjacent: repeated_prefix, cache_unfriendly, repeated_prefix, ...
    assert condition_sequence == [
        CONDITION_REPEATED_PREFIX, CONDITION_CACHE_UNFRIENDLY,
        CONDITION_REPEATED_PREFIX, CONDITION_CACHE_UNFRIENDLY,
        CONDITION_REPEATED_PREFIX, CONDITION_CACHE_UNFRIENDLY,
    ]


def test_paired_calls_share_the_same_pair_index_and_question_index():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=3)
    for pair_index in (1, 2, 3):
        pair_calls = [c for c in plan if c.pair_index == pair_index]
        assert len(pair_calls) == 2
        assert pair_calls[0].question_index == pair_calls[1].question_index


def test_execution_index_is_1_based_and_sequential():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=3)
    assert [c.execution_index for c in plan] == list(range(1, 7))


def test_randomization_is_seed_reproducible():
    plan_a = build_cache_plan(list(CONDITION_ORDER), repetitions=4, randomize_condition_order_within_pair=True, randomization_seed=7)
    plan_b = build_cache_plan(list(CONDITION_ORDER), repetitions=4, randomize_condition_order_within_pair=True, randomization_seed=7)
    assert [c.condition_id for c in plan_a] == [c.condition_id for c in plan_b]


def test_randomization_never_moves_a_call_across_pairs():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=4, randomize_condition_order_within_pair=True, randomization_seed=1)
    for pair_index in (1, 2, 3, 4):
        pair_calls = {c.condition_id for c in plan if c.pair_index == pair_index}
        assert pair_calls == set(CONDITION_ORDER)


def test_randomization_never_changes_total_call_count():
    plan = build_cache_plan(list(CONDITION_ORDER), repetitions=3, randomize_condition_order_within_pair=True, randomization_seed=99)
    assert len(plan) == 6
