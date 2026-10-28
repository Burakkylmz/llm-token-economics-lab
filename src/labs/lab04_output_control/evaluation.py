"""Deterministic, field-based answer evaluation for Lab04.

Answer CORRECTNESS and answer COMPLETENESS are different concepts (Part E
of this lab's design): correctness asks whether each present field's value
matches the expected value; completeness asks how many of the required
fields are present at all, independent of whether their values are right.
Both are derived here, in one place, so every consumer (output_limit_sweep's
retry-policy extraction, invocation_comparison's combined 3-field parse,
and each of its 3 independent single-answer checks) uses exactly the same
rules -- this is what makes "equivalent correctness evaluation" between the
single- and multiple-invocation strategies true by construction rather than
by coincidence.

Rules (all from this lab's task spec, Part E):
  - A missing required field is counted, never silently ignored.
  - A wrong value is counted as incorrect, never as "close enough".
  - A DUPLICATE field (the same field name appears twice in the output) is
    counted incorrect even if one of its values would otherwise match --
    an output contract that repeats a field is not a valid contract.
  - An unparseable/empty response is `missing_response`, not `incomplete`:
    "nothing to judge" is a different fact from "judged and found wanting".
  - `overall_answer_accepted` is True only when every required field is
    present exactly once and every one of them is correct.
  - A response is never accepted merely for containing the expected text as
    a substring of a longer, format-violating answer (the retry-policy
    task's strict "key: value" line format rules this out structurally);
    the one place a looser, explicitly-labeled heuristic is used is
    `evaluate_single_answer`, for the invocation-comparison task's
    free-text single-question answers (see its docstring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

STATUS_ACCEPTED = "accepted"
STATUS_INCOMPLETE = "incomplete"
STATUS_MISSING_RESPONSE = "missing_response"

# --- Output-limit sweep: API Retry Policy Extraction task -----------------

RETRY_POLICY_DOCUMENT = """\
Initial request counts as attempt 1.
Maximum retry attempts: 4.
Maximum total attempts: 5.
Base retry delay: 1 second.
Nominal exponential backoff delays: 1, 2, 4, 8 seconds.
Retryable HTTP status for this exercise: 429.
"""

REQUIRED_RETRY_FIELDS = (
    "max_retry_attempts",
    "max_total_attempts",
    "base_delay_seconds",
    "last_nominal_delay_seconds",
    "retryable_http_status",
)

# Teaching-fixture values belonging to RETRY_POLICY_DOCUMENT above -- never
# presented as a universal real-world HTTP retry rule.
EXPECTED_RETRY_FIELD_VALUES = {
    "max_retry_attempts": "4",
    "max_total_attempts": "5",
    "base_delay_seconds": "1",
    "last_nominal_delay_seconds": "8",
    "retryable_http_status": "429",
}

RETRY_POLICY_SYSTEM_MESSAGE = f"""\
You are given a short API retry policy document. Read it carefully, then \
extract exactly the fields requested by the user, following the exact \
output format the user specifies. Do not include any text other than what \
is requested.

=== API RETRY POLICY DOCUMENT ===
{RETRY_POLICY_DOCUMENT}"""

RETRY_POLICY_USER_MESSAGE = (
    "Extract the following five fields from the retry policy document above. "
    "Respond with EXACTLY five lines, each in the form \"field_name: value\", "
    "in this exact order, and nothing else (no explanation, no restating the "
    "question, no extra fields):\n\n"
    "max_retry_attempts: <integer>\n"
    "max_total_attempts: <integer>\n"
    "base_delay_seconds: <integer>\n"
    "last_nominal_delay_seconds: <integer>\n"
    "retryable_http_status: <integer>"
)


@dataclass
class FieldExtractionResult:
    status: str
    required_field_count: int
    present_field_count: int
    correct_field_count: int
    incorrect_field_count: int
    missing_field_count: int
    duplicate_field_count: int
    extra_field_count: int
    field_completeness_ratio: Optional[float]
    field_correctness_ratio: Optional[float]
    overall_answer_accepted: bool


def _missing_response_result(required_field_count: int) -> FieldExtractionResult:
    return FieldExtractionResult(
        status=STATUS_MISSING_RESPONSE,
        required_field_count=required_field_count,
        present_field_count=0,
        correct_field_count=0,
        incorrect_field_count=0,
        missing_field_count=required_field_count,
        duplicate_field_count=0,
        extra_field_count=0,
        field_completeness_ratio=0.0 if required_field_count else None,
        field_correctness_ratio=0.0 if required_field_count else None,
        overall_answer_accepted=False,
    )


def _parse_key_value_lines(message_content: str) -> tuple[dict[str, str], dict[str, int]]:
    """Parses "key: value" lines. Returns (first-seen value per key, occurrence
    count per key) -- lines without a colon, or blank lines, are ignored
    rather than treated as a parse failure of the whole response."""
    values: dict[str, str] = {}
    counts: dict[str, int] = {}
    for raw_line in message_content.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in values:
            values[key] = value
    return values, counts


def _values_match(expected: str, actual: str) -> bool:
    expected_n = expected.strip()
    actual_n = actual.strip()
    if expected_n.lower() == actual_n.lower():
        return True
    try:
        return int(expected_n) == int(actual_n)
    except ValueError:
        return False


def evaluate_fields(
    message_content: Optional[str], required_fields: tuple[str, ...], expected_values: dict[str, str]
) -> FieldExtractionResult:
    """Strict "key: value" multi-field evaluator -- used for the retry-policy
    extraction task and for invocation_comparison's single-call combined
    answer (both ask for a fixed set of "field_name: value" lines)."""
    required_field_count = len(required_fields)
    if message_content is None or not message_content.strip():
        return _missing_response_result(required_field_count)

    values, counts = _parse_key_value_lines(message_content)

    present = 0
    correct = 0
    incorrect = 0
    duplicates = 0
    for name in required_fields:
        if name not in values:
            continue
        present += 1
        if counts.get(name, 0) > 1:
            # Duplicate field: never counted correct, regardless of value --
            # see module docstring.
            duplicates += 1
            incorrect += 1
            continue
        if _values_match(expected_values[name], values[name]):
            correct += 1
        else:
            incorrect += 1

    missing = required_field_count - present
    extra = len([k for k in values if k not in required_fields])

    overall_accepted = correct == required_field_count and duplicates == 0
    return FieldExtractionResult(
        status=STATUS_ACCEPTED if overall_accepted else STATUS_INCOMPLETE,
        required_field_count=required_field_count,
        present_field_count=present,
        correct_field_count=correct,
        incorrect_field_count=incorrect,
        missing_field_count=missing,
        duplicate_field_count=duplicates,
        extra_field_count=extra,
        field_completeness_ratio=present / required_field_count if required_field_count else None,
        field_correctness_ratio=correct / required_field_count if required_field_count else None,
        overall_answer_accepted=overall_accepted,
    )


_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def evaluate_single_answer(message_content: Optional[str], expected_value: str) -> FieldExtractionResult:
    """A deliberately weak, explicitly-labeled heuristic (method name says
    "heuristic", same project convention as
    labs.lab01_token_anatomy.factorial_runner.evaluate_quality_heuristic) --
    NOT a correctness guarantee. Used only for invocation_comparison's
    multiple-invocation strategy, where each call gets a bare question (no
    "key: value" contract) and may answer in a full sentence
    ("The capital of Japan is Tokyo."). Treated as one required field so the
    same required/present/correct/missing vocabulary applies uniformly
    across both invocation strategies (see module docstring)."""
    if message_content is None or not message_content.strip():
        return _missing_response_result(1)

    expected_words = _normalize_words(expected_value)
    actual_words = _normalize_words(message_content)
    correct = bool(expected_words) and all(w in actual_words for w in expected_words)

    return FieldExtractionResult(
        status=STATUS_ACCEPTED if correct else STATUS_INCOMPLETE,
        required_field_count=1,
        present_field_count=1,
        correct_field_count=1 if correct else 0,
        incorrect_field_count=0 if correct else 1,
        missing_field_count=0,
        duplicate_field_count=0,
        extra_field_count=0,
        field_completeness_ratio=1.0,
        field_correctness_ratio=1.0 if correct else 0.0,
        overall_answer_accepted=correct,
    )
