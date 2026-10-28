from __future__ import annotations

from labs.lab06_streaming.experiments import INCIDENT_DESCRIPTION, REQUIRED_SECTIONS, build_incident_messages


def test_required_sections_has_exactly_five_headings():
    assert len(REQUIRED_SECTIONS) == 5


def test_incident_messages_are_byte_identical_across_calls():
    a = build_incident_messages()
    b = build_incident_messages()
    assert a == b


def test_incident_messages_contain_the_incident_description_and_every_heading():
    messages = build_incident_messages()
    content = messages[0]["content"]
    assert INCIDENT_DESCRIPTION.strip()[:30] in content
    for heading in REQUIRED_SECTIONS:
        assert heading in content


def test_incident_messages_is_a_single_user_message():
    messages = build_incident_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
