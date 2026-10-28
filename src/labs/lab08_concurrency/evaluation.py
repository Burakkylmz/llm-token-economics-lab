"""Deterministic, strict numeric evaluation for Lab08's arithmetic
workload.

Deliberately NOT substring matching: "18" contains "8" and "The answer is
not 8." and "8 or 9" are both wrong/ambiguous answers that a naive
`"8" in text` check would wrongly accept. After whitespace normalization,
the ENTIRE response text must be exactly one (optionally signed, optionally
decimal) number -- anything else is rejected as a format problem, never
silently treated as correct because it happens to contain the right
digits somewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_INVALID_FORMAT = "INVALID_FORMAT"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_MISSING_RESPONSE = "MISSING_RESPONSE"
STATUS_EVALUATION_UNAVAILABLE = "EVALUATION_UNAVAILABLE"

_STRICT_NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass
class EvaluationResult:
    status: str
    answer_correct: Optional[bool]
    answer_complete: Optional[bool]


def evaluate_numeric_answer(
    message_content: Optional[str], expected_answer: str, finish_reason: Optional[str]
) -> EvaluationResult:
    if message_content is None or not message_content.strip():
        return EvaluationResult(STATUS_MISSING_RESPONSE, False, False)

    normalized = message_content.strip()
    if not _STRICT_NUMBER_PATTERN.match(normalized):
        status = STATUS_TRUNCATED if finish_reason == "length" else STATUS_INVALID_FORMAT
        return EvaluationResult(status, False, False)

    try:
        actual = float(normalized)
        expected = float(expected_answer)
    except ValueError:
        return EvaluationResult(STATUS_EVALUATION_UNAVAILABLE, None, None)

    correct = actual == expected
    return EvaluationResult(STATUS_CORRECT if correct else STATUS_INCORRECT, correct, True)
