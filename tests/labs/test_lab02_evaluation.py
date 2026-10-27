from __future__ import annotations

from labs.lab02_context_growth.evaluation import (
    STATUS_CORRECT,
    STATUS_FORMAT_INVALID,
    STATUS_INCORRECT,
    STATUS_MISSING_RESPONSE,
    evaluate_integer_answer,
)


def test_correct_integer_answer_is_accepted():
    status, correct = evaluate_integer_answer("4", "4")
    assert status == STATUS_CORRECT
    assert correct is True


def test_correct_answer_with_surrounding_whitespace_is_accepted():
    status, correct = evaluate_integer_answer("  4\n", "4")
    assert status == STATUS_CORRECT
    assert correct is True


def test_incorrect_integer_answer_is_rejected():
    status, correct = evaluate_integer_answer("5", "4")
    assert status == STATUS_INCORRECT
    assert correct is False


def test_substring_containing_the_answer_is_not_accepted():
    # "The answer is not 4." contains "4" but must NOT be scored correct --
    # substring matching is explicitly forbidden by this lab's design task.
    status, correct = evaluate_integer_answer("The answer is not 4.", "4")
    assert status == STATUS_FORMAT_INVALID
    assert correct is False


def test_answer_with_extra_text_is_format_invalid_not_correct():
    status, correct = evaluate_integer_answer("4 retries", "4")
    assert status == STATUS_FORMAT_INVALID
    assert correct is False


def test_non_numeric_answer_is_format_invalid():
    status, correct = evaluate_integer_answer("four", "4")
    assert status == STATUS_FORMAT_INVALID
    assert correct is False


def test_decimal_answer_is_format_invalid():
    status, correct = evaluate_integer_answer("4.0", "4")
    assert status == STATUS_FORMAT_INVALID
    assert correct is False


def test_missing_response_is_reported_distinctly_from_incorrect():
    status, correct = evaluate_integer_answer(None, "4")
    assert status == STATUS_MISSING_RESPONSE
    assert correct is None


def test_empty_response_is_missing_not_format_invalid():
    status, correct = evaluate_integer_answer("   ", "4")
    assert status == STATUS_MISSING_RESPONSE
    assert correct is None


def test_truncated_response_with_no_integer_is_format_invalid():
    status, correct = evaluate_integer_answer("The maximum number of retry", "4")
    assert status == STATUS_FORMAT_INVALID
    assert correct is False


def test_signed_integer_matching_expected_is_accepted():
    status, correct = evaluate_integer_answer("+4", "4")
    assert status == STATUS_CORRECT
    assert correct is True
