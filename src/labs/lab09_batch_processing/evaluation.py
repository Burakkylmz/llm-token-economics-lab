"""Deterministic, whole-value evaluation for Lab09's golden workload.

Never substring matching. Two evaluation types:
  - exact_match: the ENTIRE response, trimmed and case-folded, must equal
    the expected answer exactly.
  - numeric_match: the entire response must parse as a single number
    (after stripping a leading currency symbol and thousands separators);
    compared to the expected numeric value within a small float tolerance.

Reused identically for real-time (sequential/concurrent) and Batch API
results, so "the same golden answers across all strategies" (spec H5) is
enforced by construction, not by convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from labs.lab09_batch_processing.datasets import (
    EVALUATION_TYPE_EXACT_MATCH,
    EVALUATION_TYPE_NUMERIC_MATCH,
)

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_INVALID_FORMAT = "INVALID_FORMAT"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_MISSING_RESPONSE = "MISSING_RESPONSE"
STATUS_EVALUATION_UNAVAILABLE = "EVALUATION_UNAVAILABLE"

_NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass
class EvaluationResult:
    status: str
    answer_correct: Optional[bool]


def _evaluate_exact_match(response_text: str, expected_answer: str) -> EvaluationResult:
    normalized_response = response_text.strip().casefold()
    normalized_expected = expected_answer.strip().casefold()
    correct = normalized_response == normalized_expected
    return EvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)


def _evaluate_numeric_match(response_text: str, expected_answer: str) -> EvaluationResult:
    normalized = response_text.strip().lstrip("$").replace(",", "")
    if not _NUMERIC_PATTERN.match(normalized):
        return EvaluationResult(STATUS_INVALID_FORMAT, False)

    try:
        actual = float(normalized)
        expected = float(expected_answer)
    except ValueError:
        return EvaluationResult(STATUS_INVALID_FORMAT, False)

    correct = abs(actual - expected) < 1e-6
    return EvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)


def evaluate_answer(
    response_text: Optional[str], expected_answer: str, evaluation_type: str, finish_reason: Optional[str] = None
) -> EvaluationResult:
    """Dispatches on `evaluation_type`. `finish_reason == "length"` on an
    otherwise-invalid response is reported as TRUNCATED rather than
    INVALID_FORMAT, since the model may simply have run out of tokens."""
    if response_text is None or not response_text.strip():
        return EvaluationResult(STATUS_MISSING_RESPONSE, False)

    if evaluation_type == EVALUATION_TYPE_EXACT_MATCH:
        result = _evaluate_exact_match(response_text, expected_answer)
    elif evaluation_type == EVALUATION_TYPE_NUMERIC_MATCH:
        result = _evaluate_numeric_match(response_text, expected_answer)
    else:
        return EvaluationResult(STATUS_EVALUATION_UNAVAILABLE, None)

    if result.status == STATUS_INVALID_FORMAT and finish_reason == "length":
        return EvaluationResult(STATUS_TRUNCATED, False)
    return result
