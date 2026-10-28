from __future__ import annotations

from labs.lab04_output_control.evaluation import (
    EXPECTED_RETRY_FIELD_VALUES,
    REQUIRED_RETRY_FIELDS,
    STATUS_ACCEPTED,
    STATUS_INCOMPLETE,
    STATUS_MISSING_RESPONSE,
    evaluate_fields,
    evaluate_single_answer,
)

ALL_CORRECT = """\
max_retry_attempts: 4
max_total_attempts: 5
base_delay_seconds: 1
last_nominal_delay_seconds: 8
retryable_http_status: 429
"""


def test_all_five_fields_correct_is_accepted():
    result = evaluate_fields(ALL_CORRECT, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.status == STATUS_ACCEPTED
    assert result.overall_answer_accepted is True
    assert result.required_field_count == 5
    assert result.present_field_count == 5
    assert result.correct_field_count == 5
    assert result.incorrect_field_count == 0
    assert result.missing_field_count == 0
    assert result.duplicate_field_count == 0
    assert result.field_completeness_ratio == 1.0
    assert result.field_correctness_ratio == 1.0


def test_missing_required_field_is_rejected():
    text = "max_retry_attempts: 4\nmax_total_attempts: 5\nbase_delay_seconds: 1\n"
    result = evaluate_fields(text, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.overall_answer_accepted is False
    assert result.status == STATUS_INCOMPLETE
    assert result.present_field_count == 3
    assert result.missing_field_count == 2


def test_wrong_value_is_rejected_not_silently_accepted():
    text = ALL_CORRECT.replace("retryable_http_status: 429", "retryable_http_status: 500")
    result = evaluate_fields(text, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.overall_answer_accepted is False
    assert result.correct_field_count == 4
    assert result.incorrect_field_count == 1


def test_duplicate_field_is_rejected_even_if_one_copy_is_correct():
    text = ALL_CORRECT + "retryable_http_status: 429\n"
    result = evaluate_fields(text, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.duplicate_field_count == 1
    assert result.overall_answer_accepted is False
    # the duplicated field is never counted correct, even though both copies match
    assert result.correct_field_count == 4
    assert result.incorrect_field_count == 1


def test_extra_unexpected_field_is_reported_but_does_not_block_acceptance():
    text = ALL_CORRECT + "some_extra_field: whatever\n"
    result = evaluate_fields(text, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.extra_field_count == 1
    assert result.overall_answer_accepted is True


def test_invalid_format_with_no_colon_lines_is_incomplete_not_accepted():
    result = evaluate_fields("I cannot answer this question.", REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.status == STATUS_INCOMPLETE
    assert result.present_field_count == 0
    assert result.overall_answer_accepted is False


def test_none_message_is_missing_response_not_incomplete():
    result = evaluate_fields(None, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.status == STATUS_MISSING_RESPONSE
    assert result.missing_field_count == 5
    assert result.overall_answer_accepted is False


def test_empty_string_message_is_missing_response():
    result = evaluate_fields("   \n  ", REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.status == STATUS_MISSING_RESPONSE


def test_finish_reason_stop_does_not_bypass_correctness_validation():
    # evaluate_fields never looks at finish_reason at all -- stop vs length
    # is an orthogonal signal (see labs.lab04_output_control.runner), so a
    # wrong answer is rejected regardless of what finish_reason was.
    text = ALL_CORRECT.replace("max_retry_attempts: 4", "max_retry_attempts: 99")
    result = evaluate_fields(text, REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.overall_answer_accepted is False


def test_cheapest_failed_response_is_never_marked_optimal():
    # A response that is short (cheap) but wrong must never report
    # overall_answer_accepted=True merely because it used few tokens.
    result = evaluate_fields("4", REQUIRED_RETRY_FIELDS, EXPECTED_RETRY_FIELD_VALUES)
    assert result.overall_answer_accepted is False


def test_single_answer_accepts_exact_match():
    result = evaluate_single_answer("Tokyo", "Tokyo")
    assert result.overall_answer_accepted is True
    assert result.status == STATUS_ACCEPTED


def test_single_answer_accepts_a_full_sentence_containing_the_expected_words():
    result = evaluate_single_answer("The capital of Japan is Tokyo.", "Tokyo")
    assert result.overall_answer_accepted is True


def test_single_answer_requires_every_expected_word():
    result = evaluate_single_answer("Shakespeare", "William Shakespeare")
    assert result.overall_answer_accepted is False


def test_single_answer_rejects_wrong_value():
    result = evaluate_single_answer("Osaka", "Tokyo")
    assert result.overall_answer_accepted is False
    assert result.status == STATUS_INCOMPLETE


def test_single_answer_none_is_missing_response():
    result = evaluate_single_answer(None, "Tokyo")
    assert result.status == STATUS_MISSING_RESPONSE
    assert result.overall_answer_accepted is False
