"""Context payload construction and structured-summary schema validation.

Both Full History and Summary conditions use the exact same OUTER request
shape (one system instruction message + one user message containing
"Conversation Context: ... / Question: ... / Answer Type: ..."); the ONLY
variable is what text fills CONTEXT_PAYLOAD (Part 6.2). This module builds
that payload for each of the three conditions and validates/renders the
structured summary JSON (Part 5.2).
"""

from __future__ import annotations

import json
from typing import Optional

from labs.lab10_conversation_summarization.models import SUMMARY_PENDING_GENERATION

REQUIRED_SUMMARY_KEYS = ("overview", "facts", "decisions", "open_issues", "timeline")


def validate_summary_schema(data: object) -> tuple[bool, list[str]]:
    """Structural validation only (Part 5.2/19.3) -- a structurally valid
    summary can still be factually wrong or incomplete; that is evaluated
    separately in evaluation.py's fact-retention checks."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["summary is not a JSON object"]

    for key in REQUIRED_SUMMARY_KEYS:
        if key not in data:
            errors.append(f"missing required key '{key}'")
    if errors:
        return False, errors

    if not isinstance(data["overview"], str):
        errors.append("'overview' must be a string")

    if not isinstance(data["facts"], list):
        errors.append("'facts' must be a list")
    else:
        for i, item in enumerate(data["facts"]):
            if not isinstance(item, dict) or not all(k in item for k in ("name", "value", "status")):
                errors.append(f"facts[{i}] must be an object with 'name', 'value', 'status'")

    if not isinstance(data["decisions"], list):
        errors.append("'decisions' must be a list")
    else:
        for i, item in enumerate(data["decisions"]):
            if not isinstance(item, dict) or not all(k in item for k in ("topic", "current_value", "superseded_value")):
                errors.append(f"decisions[{i}] must be an object with 'topic', 'current_value', 'superseded_value'")

    if not isinstance(data["open_issues"], list):
        errors.append("'open_issues' must be a list")
    else:
        for i, item in enumerate(data["open_issues"]):
            if not isinstance(item, dict) or not all(k in item for k in ("issue", "status")):
                errors.append(f"open_issues[{i}] must be an object with 'issue', 'status'")

    if not isinstance(data["timeline"], list):
        errors.append("'timeline' must be a list")
    else:
        for i, item in enumerate(data["timeline"]):
            if not isinstance(item, dict) or not all(k in item for k in ("event", "date")):
                errors.append(f"timeline[{i}] must be an object with 'event', 'date'")

    return (len(errors) == 0), errors


def parse_summary_json(raw_content: Optional[str]) -> tuple[Optional[dict], bool, list[str]]:
    """Returns (parsed_dict_or_None, json_valid, schema_errors)."""
    if raw_content is None or not raw_content.strip():
        return None, False, ["empty or missing response content"]
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return None, False, [f"invalid JSON: {exc}"]
    is_valid, errors = validate_summary_schema(data)
    return (data if is_valid else None), is_valid, errors


def render_summary_as_context(summary: dict) -> str:
    """Renders a validated structured summary as readable text, used as the
    CONTEXT_PAYLOAD for the two summarization conditions."""
    lines = ["Overview:", summary.get("overview", ""), ""]

    facts = summary.get("facts") or []
    if facts:
        lines.append("Facts:")
        for f in facts:
            lines.append(f"- {f.get('name')}: {f.get('value')} (status: {f.get('status')})")
        lines.append("")

    decisions = summary.get("decisions") or []
    if decisions:
        lines.append("Decisions:")
        for d in decisions:
            superseded = d.get("superseded_value")
            suffix = f" (superseded: {superseded})" if superseded else ""
            lines.append(f"- {d.get('topic')}: {d.get('current_value')}{suffix}")
        lines.append("")

    open_issues = summary.get("open_issues") or []
    if open_issues:
        lines.append("Open Issues:")
        for issue in open_issues:
            lines.append(f"- {issue.get('issue')}: {issue.get('status')}")
        lines.append("")

    timeline = summary.get("timeline") or []
    if timeline:
        lines.append("Timeline:")
        for event in timeline:
            date = event.get("date")
            suffix = f" ({date})" if date else ""
            lines.append(f"- {event.get('event')}{suffix}")

    return "\n".join(lines).strip()


def pending_generation_context() -> str:
    """A summary that has not yet been generated must be labeled this way,
    never replaced with fabricated content (Part 8.4)."""
    return SUMMARY_PENDING_GENERATION


def build_followup_user_message(context_payload: str, question: str, answer_type: str, allowed_enum_values: Optional[tuple[str, ...]]) -> str:
    enum_text = ", ".join(allowed_enum_values) if allowed_enum_values else "N/A"
    return (
        f"Conversation Context:\n{context_payload}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer Type: {answer_type}\n"
        f"Allowed Values: {enum_text}\n\n"
        'Expected Answer Format:\n{"answer": "..."}'
    )
