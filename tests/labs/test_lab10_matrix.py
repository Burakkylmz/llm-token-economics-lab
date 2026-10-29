from __future__ import annotations

import pytest

from labs.lab10_conversation_summarization.datasets import DATASET_IDS
from labs.lab10_conversation_summarization.matrix import (
    build_plan,
    calls_per_dataset,
    datasets_for_profile,
    question_count_for_profile,
    select_questions,
    total_planned_calls,
)
from labs.lab10_conversation_summarization.models import (
    CALL_PURPOSE_FOLLOWUP_ANSWER,
    CALL_PURPOSE_SUMMARY_GENERATION,
    CONDITION_FULL_HISTORY_BASELINE,
    CONDITION_SUMMARIZATION_LOW,
    CONDITION_SUMMARIZATION_MEDIUM,
)


def test_pilot_is_one_dataset_two_questions_eight_calls():
    assert datasets_for_profile("pilot") == ("customer_support_001",)
    assert question_count_for_profile("pilot") == 2
    assert total_planned_calls("pilot") == 8


def test_default_is_one_dataset_four_questions_fourteen_calls():
    assert datasets_for_profile("default") == ("customer_support_001",)
    assert question_count_for_profile("default") == 4
    assert total_planned_calls("default") == 14


def test_extended_is_two_datasets_four_questions_twenty_eight_calls():
    assert datasets_for_profile("extended") == DATASET_IDS
    assert question_count_for_profile("extended") == 4
    assert total_planned_calls("extended") == 28


def test_default_profile_dataset_is_configurable():
    assert datasets_for_profile("default", default_dataset_ids=("technical_incident_001",)) == ("technical_incident_001",)


def test_calls_per_dataset_formula():
    assert calls_per_dataset(2) == 8
    assert calls_per_dataset(4) == 14


def test_select_questions_returns_stable_first_n():
    all_four = select_questions("customer_support_001", 4)
    first_two = select_questions("customer_support_001", 2)
    assert [q.question_id for q in first_two] == [q.question_id for q in all_four[:2]]


def test_select_questions_rejects_count_larger_than_available():
    with pytest.raises(ValueError):
        select_questions("customer_support_001", 99)


def test_build_plan_pilot_has_eight_calls_two_summary_six_followup():
    plan = build_plan("pilot")
    assert len(plan) == 8
    summary_calls = [c for c in plan if c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION]
    followup_calls = [c for c in plan if c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
    assert len(summary_calls) == 2
    assert len(followup_calls) == 6


def test_build_plan_summary_generation_always_precedes_its_followups():
    plan = build_plan("default")
    low_summary_index = next(c.execution_index for c in plan if c.condition_id == CONDITION_SUMMARIZATION_LOW and c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION)
    medium_summary_index = next(c.execution_index for c in plan if c.condition_id == CONDITION_SUMMARIZATION_MEDIUM and c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION)
    low_followup_indices = [c.execution_index for c in plan if c.condition_id == CONDITION_SUMMARIZATION_LOW and c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
    medium_followup_indices = [c.execution_index for c in plan if c.condition_id == CONDITION_SUMMARIZATION_MEDIUM and c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
    assert all(low_summary_index < i for i in low_followup_indices)
    assert all(medium_summary_index < i for i in medium_followup_indices)


def test_build_plan_every_condition_gets_the_same_question_set():
    plan = build_plan("default")
    by_condition = {}
    for cid in (CONDITION_FULL_HISTORY_BASELINE, CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM):
        by_condition[cid] = {c.question_id for c in plan if c.condition_id == cid and c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER}
    assert by_condition[CONDITION_FULL_HISTORY_BASELINE] == by_condition[CONDITION_SUMMARIZATION_LOW] == by_condition[CONDITION_SUMMARIZATION_MEDIUM]
    assert len(by_condition[CONDITION_FULL_HISTORY_BASELINE]) == 4


def test_build_plan_extended_covers_both_datasets_with_no_hidden_calls():
    plan = build_plan("extended")
    assert len(plan) == 28
    dataset_ids_seen = {c.dataset_id for c in plan}
    assert dataset_ids_seen == set(DATASET_IDS)
    for dataset_id in DATASET_IDS:
        assert sum(1 for c in plan if c.dataset_id == dataset_id) == 14


def test_build_plan_execution_index_is_sequential_from_one():
    plan = build_plan("pilot")
    assert [c.execution_index for c in plan] == list(range(1, 9))


def test_build_plan_low_and_medium_summary_calls_have_correct_reasoning_effort():
    plan = build_plan("pilot")
    low_call = next(c for c in plan if c.condition_id == CONDITION_SUMMARIZATION_LOW and c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION)
    medium_call = next(c for c in plan if c.condition_id == CONDITION_SUMMARIZATION_MEDIUM and c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION)
    assert low_call.reasoning_effort == "low"
    assert medium_call.reasoning_effort == "medium"
