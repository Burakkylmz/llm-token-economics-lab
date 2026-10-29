from __future__ import annotations

from labs.lab09_batch_processing.evaluation import (
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_MISSING_RESPONSE,
    STATUS_TRUNCATED,
    evaluate_answer,
)


def test_exact_match_correct_case_insensitive():
    result = evaluate_answer("  positive  ", "POSITIVE", "exact_match")
    assert result.status == STATUS_CORRECT
    assert result.answer_correct is True


def test_exact_match_incorrect():
    result = evaluate_answer("NEGATIVE", "POSITIVE", "exact_match")
    assert result.status == STATUS_INCORRECT
    assert result.answer_correct is False


def test_exact_match_rejects_near_miss_not_substring_matched():
    # "POSITIVE" is a substring of this, but the whole value is not equal --
    # a naive substring check would wrongly accept this.
    result = evaluate_answer("I think it is POSITIVE overall", "POSITIVE", "exact_match")
    assert result.status == STATUS_INCORRECT
    assert result.answer_correct is False


def test_numeric_match_correct_with_currency_and_commas():
    result = evaluate_answer("$1,250.00", "1250.00", "numeric_match")
    assert result.status == STATUS_CORRECT
    assert result.answer_correct is True


def test_numeric_match_incorrect():
    result = evaluate_answer("61", "60", "numeric_match")
    assert result.status == STATUS_INCORRECT
    assert result.answer_correct is False


def test_numeric_match_invalid_format():
    result = evaluate_answer("about sixty", "60", "numeric_match")
    assert result.status == STATUS_INVALID_FORMAT
    assert result.answer_correct is False


def test_missing_response_is_reported_not_incorrect():
    result = evaluate_answer(None, "60", "numeric_match")
    assert result.status == STATUS_MISSING_RESPONSE
    assert result.answer_correct is False

    result_empty = evaluate_answer("   ", "POSITIVE", "exact_match")
    assert result_empty.status == STATUS_MISSING_RESPONSE


def test_truncated_reported_when_finish_reason_is_length():
    result = evaluate_answer("abou", "60", "numeric_match", finish_reason="length")
    assert result.status == STATUS_TRUNCATED
    assert result.answer_correct is False


def test_invalid_format_when_finish_reason_is_not_length():
    result = evaluate_answer("abou", "60", "numeric_match", finish_reason="stop")
    assert result.status == STATUS_INVALID_FORMAT
