from __future__ import annotations

import json

from labs.lab10_conversation_summarization.evaluation import (
    FACT_INCORRECT,
    FACT_MISSING,
    FACT_PRESENT,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_INVALID_FORMAT,
    STATUS_MISSING_RESPONSE,
    STATUS_TRUNCATED,
    evaluate_fact_retention,
    evaluate_followup_answer,
)


def _answer(value: str) -> str:
    return json.dumps({"answer": value})


# -- Downstream answer evaluation (Part 11.1/19.4) --------------------------


def test_correct_exact_identifier():
    r = evaluate_followup_answer(_answer("ORD-74291"), "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_CORRECT and r.answer_correct is True


def test_incorrect_exact_identifier():
    r = evaluate_followup_answer(_answer("ORD-99999"), "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_INCORRECT and r.answer_correct is False


def test_prose_explanation_containing_expected_value_is_rejected_as_invalid_format():
    r = evaluate_followup_answer(_answer("The order ID is ORD-74291"), "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_INVALID_FORMAT
    assert r.answer_correct is False


def test_correct_enum():
    r = evaluate_followup_answer(_answer("AFTERNOON"), "stop", "AFTERNOON", "enum", ("MORNING", "AFTERNOON"))
    assert r.status == STATUS_CORRECT


def test_enum_case_insensitive_match():
    r = evaluate_followup_answer(_answer("afternoon"), "stop", "AFTERNOON", "enum", ("MORNING", "AFTERNOON"))
    assert r.status == STATUS_CORRECT


def test_enum_value_outside_allowed_set_is_invalid_format():
    r = evaluate_followup_answer(_answer("EVENING"), "stop", "AFTERNOON", "enum", ("MORNING", "AFTERNOON"))
    assert r.status == STATUS_INVALID_FORMAT


def test_incorrect_enum_within_allowed_set():
    r = evaluate_followup_answer(_answer("MORNING"), "stop", "AFTERNOON", "enum", ("MORNING", "AFTERNOON"))
    assert r.status == STATUS_INCORRECT


def test_correct_iso_date():
    r = evaluate_followup_answer(_answer("2026-11-05"), "stop", "2026-11-05", "iso_date")
    assert r.status == STATUS_CORRECT


def test_wrong_iso_date():
    r = evaluate_followup_answer(_answer("2026-11-03"), "stop", "2026-11-05", "iso_date")
    assert r.status == STATUS_INCORRECT


def test_malformed_iso_date_is_invalid_format():
    r = evaluate_followup_answer(_answer("Nov 5 2026"), "stop", "2026-11-05", "iso_date")
    assert r.status == STATUS_INVALID_FORMAT


def test_integer_normalization_accepts_numeric_equality():
    r = evaluate_followup_answer(_answer("100"), "stop", "100", "integer")
    assert r.status == STATUS_CORRECT
    r2 = evaluate_followup_answer(_answer("+100"), "stop", "100", "integer")
    assert r2.status == STATUS_CORRECT


def test_wrong_integer():
    r = evaluate_followup_answer(_answer("50"), "stop", "100", "integer")
    assert r.status == STATUS_INCORRECT


def test_non_numeric_integer_answer_is_invalid_format():
    r = evaluate_followup_answer(_answer("one hundred"), "stop", "100", "integer")
    assert r.status == STATUS_INVALID_FORMAT


def test_unknown_type_accepts_not_stated_as_correct():
    r = evaluate_followup_answer(_answer("NOT_STATED"), "stop", "NOT_STATED", "unknown")
    assert r.status == STATUS_CORRECT


def test_not_stated_is_incorrect_when_expected_answer_is_known():
    r = evaluate_followup_answer(_answer("NOT_STATED"), "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_INCORRECT


def test_substring_false_positive_rejected():
    # "18" contains "8" but must never be accepted as a match for "8".
    r = evaluate_followup_answer(_answer("18"), "stop", "8", "integer")
    assert r.status == STATUS_INCORRECT


def test_missing_response_is_none():
    r = evaluate_followup_answer(None, "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_MISSING_RESPONSE


def test_empty_string_response_is_missing():
    r = evaluate_followup_answer("", "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_MISSING_RESPONSE


def test_invalid_json_is_invalid_format():
    r = evaluate_followup_answer("not json at all", "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_INVALID_FORMAT


def test_missing_answer_key_is_invalid_format():
    r = evaluate_followup_answer(json.dumps({"result": "ORD-74291"}), "stop", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_INVALID_FORMAT


def test_truncated_output_with_length_finish_reason():
    r = evaluate_followup_answer('{"answ', "length", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_TRUNCATED


def test_correct_answer_never_marked_truncated_despite_length_finish_reason():
    r = evaluate_followup_answer(_answer("ORD-74291"), "length", "ORD-74291", "exact_identifier")
    assert r.status == STATUS_CORRECT


# -- Fact retention & known-fact-violation checks (Part 11.3/11.4/11.5) ----


def _customer_support_summary(**overrides) -> dict:
    base = {
        "overview": "o",
        "facts": [
            {"name": "order_id", "value": "ORD-74291", "status": "confirmed"},
            {"name": "support_ticket", "value": "TKT-516", "status": "confirmed"},
            {"name": "delivery_city", "value": "Izmir", "status": "confirmed"},
        ],
        "decisions": [
            {"topic": "delivery date", "current_value": "2026-11-05", "superseded_value": "2026-11-03"},
            {"topic": "delivery time preference", "current_value": "afternoon", "superseded_value": "morning"},
        ],
        "open_issues": [{"issue": "return request status", "status": "pending"}],
        "timeline": [],
    }
    base.update(overrides)
    return base


def test_fact_retention_all_present_gives_full_rate():
    result = evaluate_fact_retention("customer_support_001", _customer_support_summary())
    assert result.fact_retention_rate == 1.0
    assert result.known_fact_violation_count == 0


def test_fact_retention_missing_order_id_is_detected():
    summary = _customer_support_summary(facts=[{"name": "support_ticket", "value": "TKT-516", "status": "confirmed"}])
    result = evaluate_fact_retention("customer_support_001", summary)
    order_check = next(c for c in result.checks if c.fact_name == "order_id")
    assert order_check.status == FACT_MISSING
    assert result.fact_retention_rate < 1.0


def test_fact_retention_superseded_preference_marked_current_is_incorrect_not_present():
    summary = _customer_support_summary(
        decisions=[{"topic": "delivery time preference", "current_value": "morning", "superseded_value": None}]
    )
    result = evaluate_fact_retention("customer_support_001", summary)
    pref_check = next(c for c in result.checks if c.fact_name == "latest_delivery_preference")
    assert pref_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_fact_retention_original_date_as_current_is_a_known_violation():
    summary = _customer_support_summary(
        decisions=[{"topic": "delivery date", "current_value": "2026-11-03", "superseded_value": None}]
    )
    result = evaluate_fact_retention("customer_support_001", summary)
    date_check = next(c for c in result.checks if c.fact_name == "latest_delivery_date")
    assert date_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_fact_retention_return_marked_approved_is_a_known_violation():
    summary = _customer_support_summary(open_issues=[{"issue": "return request status", "status": "approved"}])
    result = evaluate_fact_retention("customer_support_001", summary)
    return_check = next(c for c in result.checks if c.fact_name == "return_status")
    assert return_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_fact_retention_invented_tracking_number_is_incorrect_and_a_violation():
    summary = _customer_support_summary(facts=_customer_support_summary()["facts"] + [{"name": "tracking_number", "value": "TRK-998877", "status": "confirmed"}])
    result = evaluate_fact_retention("customer_support_001", summary)
    tracking_check = next(c for c in result.checks if c.fact_name == "tracking_number")
    assert tracking_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_fact_retention_correctly_omitted_tracking_number_is_present():
    result = evaluate_fact_retention("customer_support_001", _customer_support_summary())
    tracking_check = next(c for c in result.checks if c.fact_name == "tracking_number")
    assert tracking_check.status == FACT_PRESENT


def test_fact_retention_missing_summary_is_never_scored_as_fully_retained():
    result = evaluate_fact_retention("customer_support_001", None)
    assert result.fact_retention_rate is None
    assert all(c.status == "evaluation_unavailable" for c in result.checks)


def _technical_incident_summary(**overrides) -> dict:
    base = {
        "overview": "o",
        "facts": [
            {"name": "incident_id", "value": "INC-2084", "status": "confirmed"},
            {"name": "service", "value": "payments-api", "status": "confirmed"},
            {"name": "region", "value": "eu-west-1", "status": "confirmed"},
            {"name": "primary_error", "value": "HTTP 504", "status": "confirmed"},
            {"name": "previous_pool_limit", "value": "50", "status": "confirmed"},
            {"name": "incident_owner", "value": "Platform Team", "status": "confirmed"},
        ],
        "decisions": [
            {"topic": "root cause", "current_value": "DATABASE_CONNECTION_POOL_EXHAUSTION", "superseded_value": None},
            {"topic": "pool limit", "current_value": "100", "superseded_value": "50"},
        ],
        "open_issues": [{"issue": "permanent fix deployment", "status": "not deployed"}],
        "timeline": [],
    }
    base.update(overrides)
    return base


def test_technical_incident_rejected_hypothesis_as_root_cause_is_a_violation():
    summary = _technical_incident_summary(
        decisions=[{"topic": "root cause", "current_value": "memory leak", "superseded_value": None}]
    )
    result = evaluate_fact_retention("technical_incident_001", summary)
    root_cause_check = next(c for c in result.checks if c.fact_name == "root_cause")
    assert root_cause_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_technical_incident_permanent_fix_marked_deployed_is_a_violation():
    summary = _technical_incident_summary(open_issues=[{"issue": "permanent fix deployment", "status": "deployed"}])
    result = evaluate_fact_retention("technical_incident_001", summary)
    fix_check = next(c for c in result.checks if c.fact_name == "permanent_fix_deployed")
    assert fix_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1


def test_technical_incident_invented_deployment_date_is_a_violation():
    summary = _technical_incident_summary(timeline=[{"event": "permanent fix deployed", "date": "2026-12-01"}])
    result = evaluate_fact_retention("technical_incident_001", summary)
    date_check = next(c for c in result.checks if c.fact_name == "permanent_fix_deployment_date")
    assert date_check.status == FACT_INCORRECT
    assert result.known_fact_violation_count >= 1
