"""Deterministic, strict answer-correctness evaluation for Lab02's context
conditions.

Deliberately NOT substring matching: "The answer is not 4." contains "4"
but is not a valid answer to a question that demanded ONLY the integer, so
it must be rejected as `format_invalid`, never accepted as correct. The
rule is simple and auditable: after whitespace normalization, the ENTIRE
response text must be exactly one optionally-signed integer literal, which
is then compared to the expected answer as an integer (not a string, so
"4" and "+4" are both accepted as 4 while "4.0" is rejected).

API success and answer correctness are different concepts: a response can
succeed at the API level (status 200, usage reported) while still scoring
`incorrect` or `format_invalid` here.
"""

from __future__ import annotations

import re
from typing import Optional

STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_FORMAT_INVALID = "format_invalid"
STATUS_MISSING_RESPONSE = "missing_response"
STATUS_UNAVAILABLE = "unavailable"

_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")


def evaluate_integer_answer(message_content: Optional[str], expected_answer: str) -> tuple[str, Optional[bool]]:
    """Returns (status, correct). `correct` is True only for STATUS_CORRECT,
    False for STATUS_INCORRECT/STATUS_FORMAT_INVALID (both are definite wrong
    answers, just for different reasons), and None for STATUS_MISSING_RESPONSE
    (nothing to judge)."""
    if message_content is None:
        return STATUS_MISSING_RESPONSE, None

    normalized = message_content.strip()
    if not normalized:
        return STATUS_MISSING_RESPONSE, None

    if not _INTEGER_PATTERN.match(normalized):
        return STATUS_FORMAT_INVALID, False

    try:
        expected_int = int(expected_answer.strip())
        actual_int = int(normalized)
    except ValueError:
        return STATUS_FORMAT_INVALID, False

    return (STATUS_CORRECT, True) if actual_int == expected_int else (STATUS_INCORRECT, False)
