from __future__ import annotations

from labs.lab10_conversation_summarization.datasets import (
    DATASET_IDS,
    check_dataset_integrity,
    dataset_version_hashes,
    load_conversation,
    load_dataset_facts,
    load_golden_questions,
    render_transcript,
)


def test_both_dataset_ids_are_registered():
    assert DATASET_IDS == ("customer_support_001", "technical_incident_001")


def test_every_dataset_passes_integrity_checks():
    for dataset_id in DATASET_IDS:
        assert check_dataset_integrity(dataset_id) == []


def test_customer_support_conversation_has_at_least_24_messages():
    conversation = load_conversation("customer_support_001")
    assert len(conversation.messages) >= 24


def test_technical_incident_conversation_has_at_least_24_messages():
    conversation = load_conversation("technical_incident_001")
    assert len(conversation.messages) >= 24


def test_customer_support_timeline_shows_superseded_delivery_date():
    transcript = render_transcript(load_conversation("customer_support_001"))
    assert "2026-11-03" in transcript
    assert "2026-11-05" in transcript
    assert "no longer applies" in transcript


def test_customer_support_tracking_number_never_appears():
    transcript = render_transcript(load_conversation("customer_support_001"))
    assert "tracking number has not been generated" in transcript.lower()
    # No fabricated concrete tracking number value anywhere in the fixture.
    assert "trk-" not in transcript.lower()


def test_customer_support_return_status_stays_pending_never_approved():
    transcript = render_transcript(load_conversation("customer_support_001"))
    assert "pending" in transcript.lower()
    assert "return request" in transcript.lower()
    assert "approved" not in transcript.lower()


def test_technical_incident_rejects_initial_hypotheses_before_confirming_root_cause():
    transcript = render_transcript(load_conversation("technical_incident_001"))
    memory_leak_index = transcript.lower().index("memory leak")
    confirmed_index = transcript.index("DATABASE_CONNECTION_POOL_EXHAUSTION")
    assert memory_leak_index < confirmed_index
    assert "does not match" in transcript.lower() or "ruling out" in transcript.lower()


def test_technical_incident_distinguishes_temporary_from_permanent():
    transcript = render_transcript(load_conversation("technical_incident_001"))
    assert "temporary mitigation" in transcript.lower()
    assert "has not been deployed" in transcript.lower()


def test_technical_incident_permanent_fix_deployment_date_never_appears():
    transcript = render_transcript(load_conversation("technical_incident_001"))
    assert "don't have a confirmed deployment date" in transcript.lower()


def test_golden_questions_cover_four_distinct_answer_types_per_dataset():
    expected_types = {
        "customer_support_001": {"exact_identifier", "enum", "iso_date", "unknown"},
        "technical_incident_001": {"exact_identifier", "enum", "integer", "unknown"},
    }
    for dataset_id, expected in expected_types.items():
        questions = load_golden_questions(dataset_id)
        assert len(questions) == 4
        assert {q.answer_type for q in questions} == expected
        unknown_questions = [q for q in questions if q.answer_type == "unknown"]
        assert len(unknown_questions) == 1
        assert unknown_questions[0].expected_answer == "NOT_STATED"


def test_golden_question_ids_are_unique_per_dataset():
    for dataset_id in DATASET_IDS:
        ids = [q.question_id for q in load_golden_questions(dataset_id)]
        assert len(ids) == len(set(ids))


def test_golden_facts_include_at_least_one_deliberately_unstated_fact():
    for dataset_id in DATASET_IDS:
        facts = load_dataset_facts(dataset_id)
        assert any(f.canonical_value is None for f in facts.critical_facts)


def test_golden_facts_distinguish_superseded_from_current():
    facts = load_dataset_facts("customer_support_001")
    date_fact = next(f for f in facts.critical_facts if f.name == "latest_delivery_date")
    assert date_fact.canonical_value == "2026-11-05"
    assert date_fact.superseded_value == "2026-11-03"
    assert date_fact.canonical_value != date_fact.superseded_value


def test_dataset_version_hashes_are_stable_sha256_hex_digests():
    hashes = dataset_version_hashes("v1")
    assert set(hashes.keys()) == {"conversations", "golden_questions", "golden_facts"}
    for value in hashes.values():
        assert len(value) == 64
    # Deterministic: hashing the same file twice gives the same digest.
    assert dataset_version_hashes("v1") == hashes


def test_golden_expected_answers_never_leak_evaluation_rules_into_dataset():
    # The dataset files themselves carry no reference to summarization/
    # evaluation internals -- only questions/facts/conversation content.
    for dataset_id in DATASET_IDS:
        for q in load_golden_questions(dataset_id):
            assert "golden" not in q.question.lower()
            assert "expected_answer" not in q.question.lower()
