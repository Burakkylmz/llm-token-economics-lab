from __future__ import annotations

from labs.lab06_streaming.evaluation import (
    STATUS_COMPLETE,
    STATUS_INCOMPLETE,
    STATUS_NO_VISIBLE_CONTENT,
    STATUS_STREAM_INTERRUPTED,
    STATUS_TRUNCATED,
    evaluate_incident_response,
)

ALL_FIVE_SECTIONS = """\
Incident Summary: A brief outage occurred in the checkout service.
Possible Causes: Connection pool exhaustion after a recent deployment.
User Impact: Roughly 3,200 checkout requests failed.
Immediate Mitigation: On-call engineer restarted the client pods.
Prevention Measures: Restore the connection timeout configuration.
"""


def test_all_five_sections_present_is_complete():
    r = evaluate_incident_response(ALL_FIVE_SECTIONS, "stop")
    assert r.status == STATUS_COMPLETE
    assert r.answer_complete is True
    assert r.present_sections == 5
    assert r.required_sections == 5
    assert r.completeness_ratio == 1.0


def test_missing_section_is_incomplete():
    text = ALL_FIVE_SECTIONS.replace("Prevention Measures: Restore the connection timeout configuration.\n", "")
    r = evaluate_incident_response(text, "stop")
    assert r.status == STATUS_INCOMPLETE
    assert r.present_sections == 4
    assert r.answer_complete is False


def test_empty_section_body_does_not_count_as_present():
    text = ALL_FIVE_SECTIONS.replace(
        "Prevention Measures: Restore the connection timeout configuration.\n", "Prevention Measures:\n"
    )
    r = evaluate_incident_response(text, "stop")
    assert r.present_sections == 4


def test_truncated_when_incomplete_and_finish_reason_is_length():
    text = ALL_FIVE_SECTIONS.replace("Prevention Measures: Restore the connection timeout configuration.\n", "")
    r = evaluate_incident_response(text, "length")
    assert r.status == STATUS_TRUNCATED


def test_complete_response_is_never_marked_truncated_even_with_finish_reason_length():
    r = evaluate_incident_response(ALL_FIVE_SECTIONS, "length")
    assert r.status == STATUS_COMPLETE


def test_no_visible_content_for_none_message():
    r = evaluate_incident_response(None, "stop")
    assert r.status == STATUS_NO_VISIBLE_CONTENT
    assert r.answer_complete is False


def test_no_visible_content_for_empty_string():
    r = evaluate_incident_response("   ", "stop")
    assert r.status == STATUS_NO_VISIBLE_CONTENT


def test_no_visible_content_for_empty_stream_status():
    r = evaluate_incident_response("anything", "stop", stream_status="EMPTY_STREAM")
    assert r.status == STATUS_NO_VISIBLE_CONTENT


def test_stream_interrupted_status_overrides_everything():
    r = evaluate_incident_response(ALL_FIVE_SECTIONS, "stop", stream_status="STREAM_INTERRUPTED")
    assert r.status == STATUS_STREAM_INTERRUPTED
    assert r.answer_complete is None


def test_duplicate_heading_keeps_only_first_occurrence_body():
    text = ALL_FIVE_SECTIONS + "Incident Summary: a second, different summary.\n"
    r = evaluate_incident_response(text, "stop")
    # Still complete (duplicate heading doesn't remove the first body),
    # but this asserts the parser doesn't crash or double count.
    assert r.present_sections == 5


def test_headings_found_even_with_different_case():
    text = ALL_FIVE_SECTIONS.replace("Incident Summary:", "INCIDENT SUMMARY:")
    r = evaluate_incident_response(text, "stop")
    assert r.present_sections == 5


def test_heading_presence_alone_is_not_treated_as_semantic_correctness():
    # Headings present, bodies are nonsense -- still COMPLETE structurally,
    # because this evaluator checks completeness, not factual accuracy.
    text = "\n".join(f"{h}: xyz" for h in ("Incident Summary", "Possible Causes", "User Impact", "Immediate Mitigation", "Prevention Measures"))
    r = evaluate_incident_response(text, "stop")
    assert r.status == STATUS_COMPLETE
