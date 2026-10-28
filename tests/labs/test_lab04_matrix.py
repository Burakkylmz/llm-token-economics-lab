from __future__ import annotations

from labs.lab04_output_control.experiments import QUESTIONS
from labs.lab04_output_control.matrix import build_invocation_plan, build_output_limit_plan


def test_default_output_limit_plan_has_six_calls():
    plan = build_output_limit_plan([64, 256, 1024], repetitions=2)
    assert len(plan) == 6


def test_output_limit_plan_sends_correct_max_completion_tokens_per_condition():
    plan = build_output_limit_plan([64, 256, 1024], repetitions=2)
    assert [c.max_completion_tokens for c in plan] == [64, 64, 256, 256, 1024, 1024]


def test_output_limit_plan_labels_conditions_by_recommended_profile():
    plan = build_output_limit_plan([64, 256, 1024], repetitions=1)
    assert [c.condition_id for c in plan] == ["restrictive", "moderate", "generous"]


def test_output_limit_plan_execution_index_is_sequential_from_start():
    plan = build_output_limit_plan([64, 256], repetitions=2, start_execution_index=5)
    assert [c.execution_index for c in plan] == [5, 6, 7, 8]


def test_output_limit_plan_repetition_index_resets_per_condition():
    plan = build_output_limit_plan([64, 256], repetitions=3)
    reps = [c.repetition_index for c in plan]
    assert reps == [1, 2, 3, 1, 2, 3]


def test_invocation_plan_has_exactly_one_single_and_three_multiple_calls():
    plan = build_invocation_plan()
    assert len(plan) == 1 + len(QUESTIONS)
    assert sum(1 for c in plan if c.strategy == "single") == 1
    assert sum(1 for c in plan if c.strategy == "multiple") == len(QUESTIONS)


def test_invocation_plan_single_call_has_no_question_index():
    plan = build_invocation_plan()
    single = [c for c in plan if c.strategy == "single"][0]
    assert single.question_index is None


def test_invocation_plan_multiple_calls_cover_every_question_index_once():
    plan = build_invocation_plan()
    multiple = [c for c in plan if c.strategy == "multiple"]
    assert sorted(c.question_index for c in multiple) == list(range(len(QUESTIONS)))


def test_invocation_plan_execution_index_continues_from_start():
    plan = build_invocation_plan(start_execution_index=7)
    assert [c.execution_index for c in plan] == [7, 8, 9, 10]


def test_all_mode_default_matrix_is_exactly_ten_calls():
    output_limit_plan = build_output_limit_plan([64, 256, 1024], repetitions=2)
    invocation_plan = build_invocation_plan(start_execution_index=len(output_limit_plan) + 1)
    assert len(output_limit_plan) + len(invocation_plan) == 10
