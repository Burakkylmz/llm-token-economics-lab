from __future__ import annotations

from labs.lab05_model_routing.evaluation import (
    STATUS_CORRECT,
    STATUS_EVALUATION_UNAVAILABLE,
    STATUS_INCOMPLETE,
    STATUS_INCORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_STRUCTURAL_VALIDATION_ONLY,
    STATUS_TRUNCATED,
    evaluate_task_response,
)


def test_exact_match_correct():
    r = evaluate_task_response("exact_match", "Rome", "Rome", {}, "stop")
    assert r.status == STATUS_CORRECT
    assert r.answer_correct is True


def test_exact_match_incorrect():
    r = evaluate_task_response("exact_match", "Paris", "Rome", {}, "stop")
    assert r.status == STATUS_INCORRECT
    assert r.answer_correct is False


def test_exact_match_missing_response():
    r = evaluate_task_response("exact_match", None, "Rome", {}, "stop")
    assert r.status == STATUS_INCOMPLETE
    assert r.answer_correct is False


def test_numeric_correct_value():
    r = evaluate_task_response("numeric", "19", "19", {"tolerance": 0}, "stop")
    assert r.status == STATUS_CORRECT


def test_numeric_wrong_value():
    r = evaluate_task_response("numeric", "20", "19", {"tolerance": 0}, "stop")
    assert r.status == STATUS_INCORRECT
    assert r.answer_correct is False


def test_numeric_substring_false_positive_is_rejected():
    # "19" should not be accepted just because it appears inside a larger,
    # unrelated number like "190".
    r = evaluate_task_response("numeric", "190", "19", {"tolerance": 0}, "stop")
    assert r.answer_correct is False


def test_numeric_no_number_in_response_is_invalid_format():
    r = evaluate_task_response("numeric", "I don't know", "19", {"tolerance": 0}, "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_numeric_within_tolerance_is_correct():
    r = evaluate_task_response("numeric", "19.04", "19", {"tolerance": 0.1}, "stop")
    assert r.status == STATUS_CORRECT


def test_numeric_missing_response():
    r = evaluate_task_response("numeric", "", "19", {"tolerance": 0}, "stop")
    assert r.status == STATUS_INCOMPLETE


def test_classification_correct_label():
    r = evaluate_task_response("classification", "negative", "negative", {"allowed_labels": ["positive", "negative", "neutral"]}, "stop")
    assert r.status == STATUS_CORRECT


def test_classification_wrong_label():
    r = evaluate_task_response("classification", "positive", "negative", {"allowed_labels": ["positive", "negative", "neutral"]}, "stop")
    assert r.status == STATUS_INCORRECT


def test_classification_substring_false_positive_rejected():
    # "spam" must not be matched inside "not_spam" via naive substring search.
    r = evaluate_task_response("classification", "not_spam", "spam", {"allowed_labels": ["spam", "not_spam"]}, "stop")
    assert r.status == STATUS_INCORRECT  # matched not_spam exactly, correctly rejected as wrong label
    assert r.answer_correct is False


def test_classification_no_allowed_label_found_is_invalid_format():
    r = evaluate_task_response("classification", "I cannot decide", "spam", {"allowed_labels": ["spam", "not_spam"]}, "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_multi_field_all_correct():
    content = "invoice_number: INV-1\ntotal_amount: 10.00\n"
    config = {"required_fields": ["invoice_number", "total_amount"], "expected_values": {"invoice_number": "INV-1", "total_amount": "10.00"}}
    r = evaluate_task_response("multi_field", content, "", config, "stop")
    assert r.status == STATUS_CORRECT
    assert r.answer_complete is True


def test_multi_field_missing_field_is_incomplete():
    content = "invoice_number: INV-1\n"
    config = {"required_fields": ["invoice_number", "total_amount"], "expected_values": {"invoice_number": "INV-1", "total_amount": "10.00"}}
    r = evaluate_task_response("multi_field", content, "", config, "stop")
    assert r.status == STATUS_INCOMPLETE
    assert r.answer_complete is False


def test_multi_field_duplicate_field_is_incorrect():
    content = "invoice_number: INV-1\ntotal_amount: 10.00\ntotal_amount: 10.00\n"
    config = {"required_fields": ["invoice_number", "total_amount"], "expected_values": {"invoice_number": "INV-1", "total_amount": "10.00"}}
    r = evaluate_task_response("multi_field", content, "", config, "stop")
    assert r.answer_correct is False


def test_json_schema_valid_and_correct():
    r = evaluate_task_response("json_schema", '{"name": "Alice", "age": "30"}', "", {"required_fields": ["name", "age"], "expected_values": {"name": "Alice", "age": "30"}}, "stop")
    assert r.status == STATUS_CORRECT


def test_json_schema_invalid_json_is_invalid_format():
    r = evaluate_task_response("json_schema", "not json at all", "", {"required_fields": ["name"], "expected_values": {"name": "Alice"}}, "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_json_schema_missing_field_is_incomplete():
    r = evaluate_task_response("json_schema", '{"name": "Alice"}', "", {"required_fields": ["name", "age"], "expected_values": {"name": "Alice", "age": "30"}}, "stop")
    assert r.status == STATUS_INCOMPLETE


def test_json_schema_non_object_json_is_invalid_format():
    r = evaluate_task_response("json_schema", "[1, 2, 3]", "", {"required_fields": ["name"], "expected_values": {}}, "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_structural_validation_only_never_reports_correct():
    code = "def is_palindrome(s):\n    return s == s[::-1]\n"
    r = evaluate_task_response("structural_validation_only", code, "", {"function_name": "is_palindrome", "min_params": 1}, "stop")
    assert r.status == STATUS_STRUCTURAL_VALIDATION_ONLY
    assert r.answer_correct is None  # never fabricated as True


def test_structural_validation_only_signature_without_implementation_is_not_functional_correctness():
    code = "def is_palindrome(s):\n    pass\n"
    r = evaluate_task_response("structural_validation_only", code, "", {"function_name": "is_palindrome", "min_params": 1}, "stop")
    assert r.status == STATUS_STRUCTURAL_VALIDATION_ONLY
    assert r.answer_correct is None
    assert r.answer_complete is True  # signature present, but this is NOT correctness


def test_structural_validation_only_missing_function_is_not_complete():
    code = "def some_other_function():\n    pass\n"
    r = evaluate_task_response("structural_validation_only", code, "", {"function_name": "is_palindrome", "min_params": 1}, "stop")
    assert r.answer_complete is False


def test_structural_validation_only_invalid_python_is_invalid_format():
    r = evaluate_task_response("structural_validation_only", "def broken(:::", "", {"function_name": "is_palindrome", "min_params": 1}, "stop")
    assert r.status == STATUS_INVALID_FORMAT


def test_structural_validation_only_never_executes_code():
    # A malicious/dangerous-looking body must never actually run.
    code = "def is_palindrome(s):\n    import os\n    os.system('echo should-never-run')\n    return True\n"
    r = evaluate_task_response("structural_validation_only", code, "", {"function_name": "is_palindrome", "min_params": 1}, "stop")
    assert r.status == STATUS_STRUCTURAL_VALIDATION_ONLY
    assert r.answer_correct is None


def test_unknown_evaluation_type_is_evaluation_unavailable_never_success():
    r = evaluate_task_response("not_a_real_type", "anything", "x", {}, "stop")
    assert r.status == STATUS_EVALUATION_UNAVAILABLE
    assert r.answer_correct is None


def test_finish_reason_length_overrides_incorrect_to_truncated():
    r = evaluate_task_response("exact_match", "Ro", "Rome", {}, "length")
    assert r.status == STATUS_TRUNCATED


def test_finish_reason_length_never_overrides_a_correct_answer():
    r = evaluate_task_response("exact_match", "Rome", "Rome", {}, "length")
    assert r.status == STATUS_CORRECT


def test_finish_reason_length_never_overrides_structural_validation_only():
    code = "def is_palindrome(s):\n    return True\n"
    r = evaluate_task_response("structural_validation_only", code, "", {"function_name": "is_palindrome", "min_params": 1}, "length")
    assert r.status == STATUS_STRUCTURAL_VALIDATION_ONLY
