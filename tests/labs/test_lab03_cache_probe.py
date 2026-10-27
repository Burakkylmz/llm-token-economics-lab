from __future__ import annotations

import pytest

from labs.lab03_prompt_caching.cache_probe import (
    DEFAULT_PROBE_CALLS,
    build_probe_plan,
    verify_probe_plan_is_identical,
)
from labs.lab03_prompt_caching.experiments import CONDITION_REPEATED_PREFIX


def test_default_probe_plan_is_three_calls():
    plan = build_probe_plan()
    assert len(plan) == DEFAULT_PROBE_CALLS == 3


def test_probe_plan_overridable_call_count():
    plan = build_probe_plan(5)
    assert len(plan) == 5


def test_probe_plan_rejects_non_positive_count():
    with pytest.raises(ValueError):
        build_probe_plan(0)


def test_every_probe_call_uses_the_repeated_prefix_condition():
    plan = build_probe_plan(3)
    assert all(c.condition_id == CONDITION_REPEATED_PREFIX for c in plan)


def test_every_probe_call_pins_question_index_zero():
    plan = build_probe_plan(3)
    assert all(c.question_index == 0 for c in plan)


def test_probe_plan_execution_index_is_1_based_sequential():
    plan = build_probe_plan(3)
    assert [c.execution_index for c in plan] == [1, 2, 3]


def test_probe_requests_are_verified_byte_identical():
    plan = build_probe_plan(3)
    stability = verify_probe_plan_is_identical(plan, "openai/gpt-oss-20b", 2048)
    assert stability["stable"] is True
    assert stability["probe_calls"] == 3
    assert stability["request_hash"] is not None
    assert stability["stable_prefix_length"] > 0
    assert stability["full_prompt_length"] > 0


def test_probe_stability_check_is_offline_and_deterministic():
    plan = build_probe_plan(3)
    stability_1 = verify_probe_plan_is_identical(plan, "openai/gpt-oss-20b", 2048)
    stability_2 = verify_probe_plan_is_identical(plan, "openai/gpt-oss-20b", 2048)
    assert stability_1 == stability_2


def test_probe_plan_contains_no_nonce_or_timestamp_variation():
    # Every planned call must build to the EXACT same message payload --
    # verified by hashing each one individually, not just trusting the
    # aggregate "stable" flag.
    from labs.lab03_prompt_caching.diagnostics import compute_request_hash
    from labs.lab03_prompt_caching.experiments import build_cache_messages

    plan = build_probe_plan(3)
    hashes = {
        compute_request_hash("openai/gpt-oss-20b", build_cache_messages(c.condition_id, c.pair_index, c.question_index), 2048)
        for c in plan
    }
    assert len(hashes) == 1
