"""Deterministic, per-evaluation-type correctness evaluation for Lab05.

Replaces the old substring-only checks (`"19" in text`, `"def
is_palindrome" in text`), which could accept a malformed or wrong response
merely for containing a lucky substring. Six evaluator types, chosen
per-task via `GoldenTask.evaluation_type`:

  - exact_match: normalized string equality.
  - numeric: first numeric value in the response vs. expected, with an
    explicit tolerance.
  - classification: response must match exactly one of the task's
    allowed_labels.
  - multi_field: reuses labs.lab04_output_control.evaluation.evaluate_fields
    verbatim (same "key: value" multi-field extraction/correctness rules)
    -- no new framework introduced for this, per this lab's own design
    constraint.
  - json_schema: response must parse as a JSON object; required fields are
    checked for presence and correctness independently.
  - structural_validation_only: Python code is NEVER exec()'d or eval()'d.
    An `ast.parse` + `ast.walk` check only confirms a function with the
    expected name/signature exists -- this is NEVER reported as
    `answer_correct=True` (always None), since a matching signature is not
    functional correctness.

Six evaluation states (`TaskEvaluationResult.status`): CORRECT, INCORRECT,
INCOMPLETE, INVALID_FORMAT, TRUNCATED, EVALUATION_UNAVAILABLE -- plus the
code-only STRUCTURAL_VALIDATION_ONLY state, which is deliberately never
folded into CORRECT/INCORRECT. TRUNCATED overrides any non-CORRECT base
status when the provider's own `finish_reason == "length"`, so truncation
is visible as the likely cause rather than just another "incomplete".
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Optional

from labs.lab04_output_control.evaluation import evaluate_fields

EVAL_EXACT_MATCH = "exact_match"
EVAL_NUMERIC = "numeric"
EVAL_CLASSIFICATION = "classification"
EVAL_MULTI_FIELD = "multi_field"
EVAL_JSON_SCHEMA = "json_schema"
EVAL_STRUCTURAL_VALIDATION_ONLY = "structural_validation_only"

EVALUATION_TYPES = (
    EVAL_EXACT_MATCH, EVAL_NUMERIC, EVAL_CLASSIFICATION, EVAL_MULTI_FIELD, EVAL_JSON_SCHEMA,
    EVAL_STRUCTURAL_VALIDATION_ONLY,
)

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_INVALID_FORMAT = "INVALID_FORMAT"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_EVALUATION_UNAVAILABLE = "EVALUATION_UNAVAILABLE"
STATUS_STRUCTURAL_VALIDATION_ONLY = "STRUCTURAL_VALIDATION_ONLY"


@dataclass
class TaskEvaluationResult:
    status: str
    answer_correct: Optional[bool]
    answer_complete: Optional[bool]
    detail: str = ""


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def _evaluate_exact_match(content: Optional[str], expected: str) -> TaskEvaluationResult:
    if content is None or not content.strip():
        return TaskEvaluationResult(STATUS_INCOMPLETE, False, False, "empty response")
    correct = _normalize_text(content) == _normalize_text(expected)
    return TaskEvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct, True)


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_first_number(text: str) -> Optional[float]:
    match = _NUMBER_PATTERN.search(text)
    return float(match.group()) if match else None


def _evaluate_numeric(content: Optional[str], expected: str, config: dict) -> TaskEvaluationResult:
    if content is None or not content.strip():
        return TaskEvaluationResult(STATUS_INCOMPLETE, False, False, "empty response")
    actual = _extract_first_number(content)
    if actual is None:
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, False, False, "no numeric value found in response")
    try:
        expected_value = float(expected)
    except ValueError:
        return TaskEvaluationResult(STATUS_EVALUATION_UNAVAILABLE, None, None, "task's expected_answer is not numeric")
    tolerance = float((config or {}).get("tolerance", 0.0))
    correct = abs(actual - expected_value) <= tolerance
    return TaskEvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct, True)


def _evaluate_classification(content: Optional[str], expected: str, config: dict) -> TaskEvaluationResult:
    allowed = [str(label) for label in (config or {}).get("allowed_labels", [])]
    if content is None or not content.strip():
        return TaskEvaluationResult(STATUS_INCOMPLETE, False, False, "empty response")
    normalized = _normalize_text(content)
    matches = [label for label in allowed if _normalize_text(label) == normalized]
    if not matches:
        # Fall back to a whole-word match inside a longer response, but
        # still require exactly one allowed label to appear -- never a
        # bare substring match (e.g. "spam" inside "not_spam").
        words = set(re.findall(r"[a-z0-9_]+", normalized))
        matches = [label for label in allowed if _normalize_text(label) in words]
    if not matches:
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, False, False, f"no allowed label found (allowed={allowed})")
    if len(matches) > 1:
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, False, False, "ambiguous: more than one allowed label matched")
    correct = _normalize_text(matches[0]) == _normalize_text(expected)
    return TaskEvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct, True)


def _evaluate_multi_field(content: Optional[str], config: dict) -> TaskEvaluationResult:
    required = tuple((config or {}).get("required_fields", ()))
    expected_values = (config or {}).get("expected_values", {})
    result = evaluate_fields(content, required, expected_values)
    if result.status == "missing_response":
        return TaskEvaluationResult(STATUS_INCOMPLETE, False, False, "missing response")
    complete = result.present_field_count == result.required_field_count
    correct = result.overall_answer_accepted
    if correct:
        status = STATUS_CORRECT
    elif not complete:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_INCORRECT
    return TaskEvaluationResult(status, correct, complete, f"{result.present_field_count}/{result.required_field_count} fields present")


def _evaluate_json_schema(content: Optional[str], config: dict) -> TaskEvaluationResult:
    required = list((config or {}).get("required_fields", []))
    expected_values = (config or {}).get("expected_values", {})
    if content is None or not content.strip():
        return TaskEvaluationResult(STATUS_INCOMPLETE, False, False, "empty response")
    try:
        parsed = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, False, False, "response is not valid JSON")
    if not isinstance(parsed, dict):
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, False, False, "JSON response is not an object")

    missing = [k for k in required if k not in parsed]
    present = [k for k in required if k in parsed]
    correct_fields = [k for k in present if str(parsed[k]).strip().lower() == str(expected_values.get(k, "")).strip().lower()]
    complete = not missing
    correct = complete and len(correct_fields) == len(required)
    status = STATUS_CORRECT if correct else (STATUS_INCOMPLETE if missing else STATUS_INCORRECT)
    return TaskEvaluationResult(status, correct, complete, f"missing={missing}")


def _evaluate_structural_validation_only(content: Optional[str], config: dict) -> TaskEvaluationResult:
    function_name = (config or {}).get("function_name")
    min_params = int((config or {}).get("min_params", 0))
    if content is None or not content.strip():
        return TaskEvaluationResult(STATUS_INCOMPLETE, None, False, "empty response")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return TaskEvaluationResult(STATUS_INVALID_FORMAT, None, False, "response is not syntactically valid Python")
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name]
    if not functions:
        return TaskEvaluationResult(STATUS_STRUCTURAL_VALIDATION_ONLY, None, False, f"function '{function_name}' not found")
    has_enough_params = any(len(f.args.args) >= min_params for f in functions)
    return TaskEvaluationResult(
        STATUS_STRUCTURAL_VALIDATION_ONLY, None, has_enough_params,
        "function signature present -- functional correctness NOT verified (no sandboxed execution)",
    )


def evaluate_task_response(
    evaluation_type: str,
    message_content: Optional[str],
    expected_answer: str,
    evaluation_config: dict,
    finish_reason: Optional[str],
) -> TaskEvaluationResult:
    if evaluation_type == EVAL_EXACT_MATCH:
        base = _evaluate_exact_match(message_content, expected_answer)
    elif evaluation_type == EVAL_NUMERIC:
        base = _evaluate_numeric(message_content, expected_answer, evaluation_config)
    elif evaluation_type == EVAL_CLASSIFICATION:
        base = _evaluate_classification(message_content, expected_answer, evaluation_config)
    elif evaluation_type == EVAL_MULTI_FIELD:
        base = _evaluate_multi_field(message_content, evaluation_config)
    elif evaluation_type == EVAL_JSON_SCHEMA:
        base = _evaluate_json_schema(message_content, evaluation_config)
    elif evaluation_type == EVAL_STRUCTURAL_VALIDATION_ONLY:
        base = _evaluate_structural_validation_only(message_content, evaluation_config)
    else:
        return TaskEvaluationResult(STATUS_EVALUATION_UNAVAILABLE, None, None, f"unknown evaluation_type '{evaluation_type}'")

    if base.status not in (STATUS_CORRECT, STATUS_STRUCTURAL_VALIDATION_ONLY) and finish_reason == "length":
        return TaskEvaluationResult(STATUS_TRUNCATED, base.answer_correct, base.answer_complete, base.detail + " (finish_reason=length)")
    return base
