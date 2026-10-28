from __future__ import annotations

from labs.lab04_output_control.experiments import (
    EXPECTED_INVOCATION_FIELD_VALUES,
    OUTPUT_LIMIT_PROFILE_NAMES,
    QUESTIONS,
    REQUIRED_INVOCATION_FIELDS,
    build_invocation_multiple_messages,
    build_invocation_single_messages,
    build_output_limit_messages,
    condition_label_for,
    expected_invocation_answer_for,
)


def test_condition_label_uses_recommended_profile_names():
    assert condition_label_for(64) == "restrictive"
    assert condition_label_for(256) == "moderate"
    assert condition_label_for(1024) == "generous"


def test_condition_label_falls_back_to_raw_value_for_unknown_limits():
    assert condition_label_for(512) == "max_512"


def test_output_limit_messages_are_byte_identical_across_calls():
    a = build_output_limit_messages()
    b = build_output_limit_messages()
    assert a == b


def test_output_limit_messages_contain_the_retry_policy_document():
    messages = build_output_limit_messages()
    system_content = messages[0]["content"]
    assert "Maximum retry attempts: 4" in system_content
    assert "Retryable HTTP status for this exercise: 429" in system_content


def test_invocation_single_messages_ask_all_three_questions():
    messages = build_invocation_single_messages()
    content = messages[0]["content"]
    assert all(q in content for q in QUESTIONS)
    assert len(messages) == 1


def test_invocation_multiple_messages_ask_one_bare_question_per_index():
    for i, question in enumerate(QUESTIONS):
        messages = build_invocation_multiple_messages(i)
        assert messages == [{"role": "user", "content": question}]


def test_expected_invocation_answer_matches_field_values():
    for i, field in enumerate(REQUIRED_INVOCATION_FIELDS):
        assert expected_invocation_answer_for(i) == EXPECTED_INVOCATION_FIELD_VALUES[field]


def test_output_limit_profile_names_cover_the_recommended_defaults():
    assert OUTPUT_LIMIT_PROFILE_NAMES == {64: "restrictive", 256: "moderate", 1024: "generous"}
