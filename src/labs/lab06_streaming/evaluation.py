"""Deterministic completeness evaluation for Lab06's Production API
Incident Explanation task, plus the seven-state completion-status
taxonomy shared by both the streaming and non-streaming conditions.

A required section is counted "present" only if its heading is found AND
the text following it (up to the next heading or end of response) is
non-empty -- the heading's mere presence is NOT semantic correctness (Part
E's explicit warning), only a structural completeness signal.

Seven states (Part I): COMPLETE, INCOMPLETE, TRUNCATED, NO_VISIBLE_CONTENT,
STREAM_INTERRUPTED, EVALUATION_UNAVAILABLE, API_ERROR. TRUNCATED is only
assigned when the response is incomplete AND the provider's own
`finish_reason == "length"` -- never inferred from length alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from labs.lab06_streaming.experiments import REQUIRED_SECTIONS

STATUS_COMPLETE = "COMPLETE"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_NO_VISIBLE_CONTENT = "NO_VISIBLE_CONTENT"
STATUS_STREAM_INTERRUPTED = "STREAM_INTERRUPTED"
STATUS_EVALUATION_UNAVAILABLE = "EVALUATION_UNAVAILABLE"
STATUS_API_ERROR = "API_ERROR"

_HEADING_PATTERN = re.compile(
    "(" + "|".join(re.escape(h) for h in REQUIRED_SECTIONS) + r")\s*:\s*", re.IGNORECASE
)


@dataclass
class EvaluationResult:
    status: str
    answer_complete: Optional[bool]
    present_sections: int
    required_sections: int
    completeness_ratio: Optional[float]


def _parse_sections(text: str) -> dict[str, str]:
    """Maps each canonical heading (as spelled in REQUIRED_SECTIONS) to the
    text found after it, up to the next heading or the end of the
    response. A heading that appears more than once keeps only its FIRST
    occurrence's body -- a repeated heading is never treated as replacing
    an earlier, possibly-empty one."""
    matches = list(_HEADING_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        canonical = next(h for h in REQUIRED_SECTIONS if h.lower() == match.group(1).lower())
        if canonical in sections:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[canonical] = text[start:end].strip()
    return sections


def evaluate_incident_response(
    message_content: Optional[str], finish_reason: Optional[str], stream_status: Optional[str] = None
) -> EvaluationResult:
    required = len(REQUIRED_SECTIONS)

    if stream_status == "STREAM_INTERRUPTED":
        return EvaluationResult(STATUS_STREAM_INTERRUPTED, None, 0, required, None)

    if stream_status == "EMPTY_STREAM" or message_content is None or not message_content.strip():
        return EvaluationResult(STATUS_NO_VISIBLE_CONTENT, False, 0, required, 0.0)

    sections = _parse_sections(message_content)
    present = sum(1 for heading in REQUIRED_SECTIONS if sections.get(heading))
    complete = present == required
    ratio = present / required

    if complete:
        status = STATUS_COMPLETE
    elif finish_reason == "length":
        status = STATUS_TRUNCATED
    else:
        status = STATUS_INCOMPLETE

    return EvaluationResult(status, complete, present, required, ratio)
