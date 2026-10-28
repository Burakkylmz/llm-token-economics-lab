from __future__ import annotations

from labs.lab08_concurrency.evaluation import (
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_MISSING_RESPONSE,
    STATUS_TRUNCATED,
    evaluate_numeric_answer,
)


def test_exact_correct_answer():
    r = evaluate_numeric_answer("8", "8", "stop")
    assert r.status == STATUS_CORRECT
    assert r.answer_correct is True
    assert r.answer_complete is True


def test_incorrect_answer():
    r = evaluate_numeric_answer("9", "8", "stop")
    assert r.status == STATUS_INCORRECT
    assert r.answer_correct is False


def test_substring_false_positive_rejected():
    # "18" contains "8" but is a different, wrong number.
    r = evaluate_numeric_answer("18", "8", "stop")
    assert r.status == STATUS_INCORRECT
    assert r.answer_correct is False


def test_prose_containing_the_right_digit_is_rejected_as_invalid_format():
    r = evaluate_numeric_answer("The answer is not 8.", "8", "stop")
    assert r.status == STATUS_INVALID_FORMAT
    assert r.answer_correct is False


def test_ambiguous_answer_is_rejected_as_invalid_format():
    r = evaluate_numeric_answer("8 or 9", "8", "stop")
    assert r.status == STATUS_INVALID_FORMAT
    assert r.answer_correct is False


def test_missing_response_is_none_not_incorrect():
    r = evaluate_numeric_answer(None, "8", "stop")
    assert r.status == STATUS_MISSING_RESPONSE
    assert r.answer_correct is False
    assert r.answer_complete is False


def test_empty_string_is_missing_response():
    r = evaluate_numeric_answer("   ", "8", "stop")
    assert r.status == STATUS_MISSING_RESPONSE


def test_invalid_format_with_length_finish_reason_is_truncated():
    r = evaluate_numeric_answer("The ans", "8", "length")
    assert r.status == STATUS_TRUNCATED


def test_invalid_format_without_length_finish_reason_stays_invalid_format():
    r = evaluate_numeric_answer("The ans", "8", "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_signed_and_decimal_numbers_accepted():
    assert evaluate_numeric_answer("+8", "8", "stop").answer_correct is True
    assert evaluate_numeric_answer("8.0", "8", "stop").answer_correct is True


def test_correct_answer_never_marked_truncated_even_with_length_finish_reason():
    r = evaluate_numeric_answer("8", "8", "length")
    assert r.status == STATUS_CORRECT
