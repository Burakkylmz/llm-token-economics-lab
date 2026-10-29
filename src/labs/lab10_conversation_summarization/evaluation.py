"""Deterministic evaluation for Lab10 (Part 11) -- no LLM-as-judge, no
hidden API calls. Three distinct kinds of evaluation live here:

1. Downstream follow-up answer correctness (strict, per answer_type).
2. Summary fact retention against `golden_facts_v1.json`'s critical facts.
3. Known-fact-violation detection (a bounded, documented set of checks --
   never presented as a complete hallucination measurement, Part 11.5).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_INVALID_FORMAT = "INVALID_FORMAT"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_MISSING_RESPONSE = "MISSING_RESPONSE"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_NOT_STATED = "NOT_STATED"


@dataclass(frozen=True)
class AnswerEvaluation:
    status: str
    answer_correct: bool


def _extract_answer_string(raw_content: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Returns (answer_value_or_None, failure_reason_or_None)."""
    if raw_content is None or not raw_content.strip():
        return None, "missing_response"
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict) or "answer" not in data or not isinstance(data["answer"], str):
        return None, "missing_key"
    return data["answer"], None


def evaluate_followup_answer(
    raw_content: Optional[str],
    finish_reason: Optional[str],
    expected_answer: str,
    answer_type: str,
    allowed_enum_values: Optional[tuple[str, ...]] = None,
) -> AnswerEvaluation:
    """Strict, whole-value evaluation -- never a substring search, so a
    prose explanation that happens to CONTAIN the right value is always
    rejected (Part 11.1's explicit requirement)."""
    value, failure = _extract_answer_string(raw_content)

    if value is None:
        if finish_reason == "length":
            return AnswerEvaluation(STATUS_TRUNCATED, False)
        if failure == "missing_response":
            return AnswerEvaluation(STATUS_MISSING_RESPONSE, False)
        return AnswerEvaluation(STATUS_INVALID_FORMAT, False)

    value = value.strip()
    if not value:
        return AnswerEvaluation(STATUS_MISSING_RESPONSE, False)

    # The "not available" sentinel is accepted as a candidate value for ANY
    # answer_type (Part 6.1 allows it universally) -- it is only ever
    # CORRECT when the golden expected_answer itself is NOT_STATED.
    if value.upper() == _NOT_STATED:
        correct = expected_answer.upper() == _NOT_STATED
        return AnswerEvaluation(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)

    if answer_type == "unknown":
        # Expected a NOT_STATED sentinel but the model supplied some other
        # bare value instead -- a format violation only if it isn't even a
        # single well-formed token (prose, punctuation, etc.).
        if not _IDENTIFIER_PATTERN.match(value):
            return AnswerEvaluation(STATUS_INVALID_FORMAT, False)
        return AnswerEvaluation(STATUS_INCORRECT, False)

    if answer_type == "exact_identifier":
        if not _IDENTIFIER_PATTERN.match(value):
            return AnswerEvaluation(STATUS_INVALID_FORMAT, False)
        correct = value == expected_answer
        return AnswerEvaluation(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)

    if answer_type == "enum":
        normalized = value.upper()
        if allowed_enum_values is not None and normalized not in {v.upper() for v in allowed_enum_values}:
            return AnswerEvaluation(STATUS_INVALID_FORMAT, False)
        correct = normalized == expected_answer.upper()
        return AnswerEvaluation(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)

    if answer_type == "iso_date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return AnswerEvaluation(STATUS_INVALID_FORMAT, False)
        correct = value == expected_answer
        return AnswerEvaluation(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)

    if answer_type == "integer":
        if not _INTEGER_PATTERN.match(value):
            return AnswerEvaluation(STATUS_INVALID_FORMAT, False)
        correct = int(value) == int(expected_answer)
        return AnswerEvaluation(STATUS_CORRECT if correct else STATUS_INCORRECT, correct)

    raise ValueError(f"Unsupported answer_type: {answer_type}")


# --------------------------------------------------------------------------
# Summary fact retention (Part 11.3/11.4) and known-fact-violation checks
# (Part 11.5). Implemented against structured summary fields (never naive
# substring matching that could accept a negated/superseded statement) --
# explicit per-dataset checkers rather than a generic DSL, since each
# critical fact's location within the schema (facts/decisions/open_issues)
# differs and a generic matcher risks exactly the false-positive bugs this
# part of the spec warns against.
# --------------------------------------------------------------------------

FACT_PRESENT = "fact_present"
FACT_MISSING = "fact_missing"
FACT_INCORRECT = "fact_incorrect"
FACT_EVALUATION_UNAVAILABLE = "evaluation_unavailable"


@dataclass(frozen=True)
class FactCheckResult:
    fact_name: str
    status: str


def _all_text_values(summary: dict) -> list[str]:
    values = [str(summary.get("overview", ""))]
    for f in summary.get("facts") or []:
        values.append(str(f.get("value", "")))
    for d in summary.get("decisions") or []:
        values.append(str(d.get("current_value", "")))
    for i in summary.get("open_issues") or []:
        values.append(f"{i.get('issue', '')} {i.get('status', '')}")
    return values


def _decision_current_value(summary: dict, topic_keyword: str) -> Optional[str]:
    for d in summary.get("decisions") or []:
        if topic_keyword.lower() in str(d.get("topic", "")).lower():
            return str(d.get("current_value", ""))
    return None


def _fact_value(summary: dict, name_keyword: str) -> Optional[str]:
    for f in summary.get("facts") or []:
        if name_keyword.lower() in str(f.get("name", "")).lower():
            return str(f.get("value", ""))
    return None


def _open_issue_status(summary: dict, issue_keyword: str) -> Optional[str]:
    for i in summary.get("open_issues") or []:
        if issue_keyword.lower() in str(i.get("issue", "")).lower():
            return str(i.get("status", ""))
    return None


def _check_customer_support_facts(summary: dict) -> list[FactCheckResult]:
    results = []

    order_id = _fact_value(summary, "order")
    results.append(FactCheckResult("order_id", FACT_PRESENT if order_id and "ORD-74291" in order_id else FACT_MISSING))

    city = _fact_value(summary, "city") or _fact_value(summary, "delivery")
    present = bool(city) and "izmir" in city.lower()
    results.append(FactCheckResult("delivery_city", FACT_PRESENT if present else FACT_MISSING))

    date_current = _decision_current_value(summary, "date")
    if date_current is None:
        results.append(FactCheckResult("latest_delivery_date", FACT_MISSING))
    elif "2026-11-05" in date_current:
        results.append(FactCheckResult("latest_delivery_date", FACT_PRESENT))
    elif "2026-11-03" in date_current:
        results.append(FactCheckResult("latest_delivery_date", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("latest_delivery_date", FACT_MISSING))

    pref_current = _decision_current_value(summary, "preference") or _decision_current_value(summary, "time")
    if pref_current is None:
        results.append(FactCheckResult("latest_delivery_preference", FACT_MISSING))
    elif "afternoon" in pref_current.lower():
        results.append(FactCheckResult("latest_delivery_preference", FACT_PRESENT))
    elif "morning" in pref_current.lower():
        results.append(FactCheckResult("latest_delivery_preference", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("latest_delivery_preference", FACT_MISSING))

    ticket = _fact_value(summary, "ticket")
    results.append(FactCheckResult("support_ticket", FACT_PRESENT if ticket and "TKT-516" in ticket else FACT_MISSING))

    return_status = _open_issue_status(summary, "return")
    if return_status is None:
        results.append(FactCheckResult("return_status", FACT_MISSING))
    elif "pending" in return_status.lower():
        results.append(FactCheckResult("return_status", FACT_PRESENT))
    elif "approved" in return_status.lower():
        results.append(FactCheckResult("return_status", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("return_status", FACT_MISSING))

    tracking = _fact_value(summary, "tracking")
    if tracking and _IDENTIFIER_PATTERN.match(tracking.strip()) and tracking.strip().upper() != "NOT_STATED" and tracking.strip():
        results.append(FactCheckResult("tracking_number", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("tracking_number", FACT_PRESENT))

    return results


def _check_technical_incident_facts(summary: dict) -> list[FactCheckResult]:
    results = []

    incident_id = _fact_value(summary, "incident")
    results.append(FactCheckResult("incident_id", FACT_PRESENT if incident_id and "INC-2084" in incident_id else FACT_MISSING))

    service = _fact_value(summary, "service")
    results.append(FactCheckResult("service", FACT_PRESENT if service and "payments-api" in service.lower() else FACT_MISSING))

    region = _fact_value(summary, "region")
    results.append(FactCheckResult("region", FACT_PRESENT if region and "eu-west-1" in region.lower() else FACT_MISSING))

    error = _fact_value(summary, "error")
    results.append(FactCheckResult("primary_error", FACT_PRESENT if error and "504" in error else FACT_MISSING))

    root_cause = _decision_current_value(summary, "root cause") or _decision_current_value(summary, "cause")
    if root_cause is None:
        results.append(FactCheckResult("root_cause", FACT_MISSING))
    elif "database_connection_pool_exhaustion" in root_cause.lower().replace(" ", "_"):
        results.append(FactCheckResult("root_cause", FACT_PRESENT))
    elif "memory" in root_cause.lower() or "network" in root_cause.lower():
        results.append(FactCheckResult("root_cause", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("root_cause", FACT_MISSING))

    previous_limit = _fact_value(summary, "previous") or _fact_value(summary, "original pool")
    results.append(FactCheckResult("previous_pool_limit", FACT_PRESENT if previous_limit and "50" in previous_limit else FACT_MISSING))

    pool_current = _decision_current_value(summary, "pool")
    if pool_current is None:
        results.append(FactCheckResult("temporary_pool_limit", FACT_MISSING))
    elif "100" in pool_current:
        results.append(FactCheckResult("temporary_pool_limit", FACT_PRESENT))
    elif "50" in pool_current:
        results.append(FactCheckResult("temporary_pool_limit", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("temporary_pool_limit", FACT_MISSING))

    owner = _fact_value(summary, "owner")
    results.append(FactCheckResult("incident_owner", FACT_PRESENT if owner and "platform team" in owner.lower() else FACT_MISSING))

    fix_status = _open_issue_status(summary, "permanent fix") or _open_issue_status(summary, "fix")
    if fix_status is None:
        results.append(FactCheckResult("permanent_fix_deployed", FACT_MISSING))
    elif "not" in fix_status.lower() or "pending" in fix_status.lower() or "no" == fix_status.strip().lower():
        results.append(FactCheckResult("permanent_fix_deployed", FACT_PRESENT))
    elif "deployed" in fix_status.lower():
        results.append(FactCheckResult("permanent_fix_deployed", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("permanent_fix_deployed", FACT_MISSING))

    deployment_date = None
    for event in summary.get("timeline") or []:
        if "deploy" in str(event.get("event", "")).lower() and "permanent" in str(event.get("event", "")).lower():
            deployment_date = event.get("date")
    if deployment_date:
        results.append(FactCheckResult("permanent_fix_deployment_date", FACT_INCORRECT))
    else:
        results.append(FactCheckResult("permanent_fix_deployment_date", FACT_PRESENT))

    return results


_FACT_CHECKERS = {
    "customer_support_001": _check_customer_support_facts,
    "technical_incident_001": _check_technical_incident_facts,
}


@dataclass(frozen=True)
class FactRetentionResult:
    checks: tuple[FactCheckResult, ...]
    fact_retention_rate: Optional[float]
    known_fact_violation_count: int


def _known_fact_violations_customer_support(summary: dict) -> int:
    violations = 0
    pref = _decision_current_value(summary, "preference") or _decision_current_value(summary, "time")
    if pref and "morning" in pref.lower():
        violations += 1
    date_current = _decision_current_value(summary, "date")
    if date_current and "2026-11-03" in date_current:
        violations += 1
    return_status = _open_issue_status(summary, "return")
    if return_status and "approved" in return_status.lower():
        violations += 1
    tracking = _fact_value(summary, "tracking")
    if tracking and _IDENTIFIER_PATTERN.match(tracking.strip()) and tracking.strip().upper() != "NOT_STATED":
        violations += 1
    return violations


def _known_fact_violations_technical_incident(summary: dict) -> int:
    violations = 0
    fix_status = _open_issue_status(summary, "permanent fix") or _open_issue_status(summary, "fix")
    if fix_status and "deployed" in fix_status.lower() and "not" not in fix_status.lower():
        violations += 1
    for event in summary.get("timeline") or []:
        if "deploy" in str(event.get("event", "")).lower() and "permanent" in str(event.get("event", "")).lower() and event.get("date"):
            violations += 1
    root_cause = _decision_current_value(summary, "root cause") or _decision_current_value(summary, "cause")
    if root_cause and ("memory" in root_cause.lower() or "network" in root_cause.lower()):
        violations += 1
    pool_current = _decision_current_value(summary, "pool")
    if pool_current and "100" in pool_current and "permanent" in pool_current.lower():
        violations += 1
    return violations


_VIOLATION_CHECKERS = {
    "customer_support_001": _known_fact_violations_customer_support,
    "technical_incident_001": _known_fact_violations_technical_incident,
}


def evaluate_fact_retention(dataset_id: str, summary: Optional[dict]) -> FactRetentionResult:
    checker = _FACT_CHECKERS.get(dataset_id)
    violation_checker = _VIOLATION_CHECKERS.get(dataset_id)
    if checker is None or violation_checker is None:
        return FactRetentionResult(checks=(), fact_retention_rate=None, known_fact_violation_count=0)

    if summary is None:
        # A missing/invalid summary is never scored as fully retained
        # (Part 11.4) -- every fact is explicitly unavailable, not skipped.
        checks = tuple(FactCheckResult(name, FACT_EVALUATION_UNAVAILABLE) for name in _critical_fact_names(dataset_id))
        return FactRetentionResult(checks=checks, fact_retention_rate=None, known_fact_violation_count=0)

    checks = tuple(checker(summary))
    evaluable = [c for c in checks if c.status != FACT_EVALUATION_UNAVAILABLE]
    retained = [c for c in evaluable if c.status == FACT_PRESENT]
    rate = (len(retained) / len(evaluable)) if evaluable else None
    violation_count = violation_checker(summary)
    return FactRetentionResult(checks=checks, fact_retention_rate=rate, known_fact_violation_count=violation_count)


def _critical_fact_names(dataset_id: str) -> tuple[str, ...]:
    from labs.lab10_conversation_summarization.datasets import load_dataset_facts

    return tuple(f.name for f in load_dataset_facts(dataset_id).critical_facts)
