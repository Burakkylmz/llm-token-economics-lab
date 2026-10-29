from __future__ import annotations

import json

import pytest

from labs.lab09_batch_processing.batch import (
    build_batch_requests,
    parse_batch_output_jsonl,
    write_batch_jsonl,
)
from labs.lab09_batch_processing.datasets import (
    DatasetIntegrityError,
    PROFILE_TASK_COUNTS,
    PROFILES,
    batch_request_count_for_profile,
    check_dataset_integrity,
    load_dataset,
    realtime_call_count_for_profile,
    task_ids_for_profile,
    tasks_by_id,
)


# -- Dataset tests --


def test_load_dataset_returns_at_least_sixteen_tasks():
    tasks = load_dataset("v1")
    assert len(tasks) >= 16


def test_dataset_passes_its_own_integrity_check():
    tasks = load_dataset("v1")
    check_dataset_integrity(tasks)  # must not raise


def test_dataset_has_unique_task_ids():
    tasks = load_dataset("v1")
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_dataset_integrity_rejects_duplicate_task_ids():
    tasks = load_dataset("v1")
    duplicated = tasks[:1] + tasks[:1] + tasks[1:]
    with pytest.raises(DatasetIntegrityError):
        check_dataset_integrity(duplicated)


def test_dataset_integrity_rejects_unsupported_evaluation_type():
    tasks = load_dataset("v1")
    bad = tasks[0].__class__(
        task_id=tasks[0].task_id, category=tasks[0].category, prompt=tasks[0].prompt,
        expected_answer=tasks[0].expected_answer, evaluation_type="fuzzy_match", metadata={},
    )
    with pytest.raises(DatasetIntegrityError):
        check_dataset_integrity((bad,) + tasks[1:])


def test_dataset_integrity_rejects_numeric_match_with_non_numeric_expected_answer():
    tasks = load_dataset("v1")
    bad = tasks[0].__class__(
        task_id=tasks[0].task_id, category=tasks[0].category, prompt=tasks[0].prompt,
        expected_answer="not-a-number", evaluation_type="numeric_match", metadata={},
    )
    with pytest.raises(DatasetIntegrityError):
        check_dataset_integrity((bad,) + tasks[1:])


@pytest.mark.parametrize("profile", PROFILES)
def test_task_ids_for_profile_matches_documented_counts(profile):
    tasks = load_dataset("v1")
    ids = task_ids_for_profile(tasks, profile)
    assert len(ids) == PROFILE_TASK_COUNTS[profile]


def test_profiles_are_nested_prefixes():
    tasks = load_dataset("v1")
    pilot = task_ids_for_profile(tasks, "pilot")
    default = task_ids_for_profile(tasks, "default")
    extended = task_ids_for_profile(tasks, "extended")
    assert set(pilot).issubset(set(default))
    assert set(default).issubset(set(extended))


@pytest.mark.parametrize(
    "profile,expected_realtime,expected_batch",
    [("pilot", 8, 4), ("default", 16, 8), ("extended", 32, 16)],
)
def test_call_counts_match_spec_table(profile, expected_realtime, expected_batch):
    assert realtime_call_count_for_profile(profile) == expected_realtime
    assert batch_request_count_for_profile(profile) == expected_batch


def test_tasks_by_id_round_trips():
    tasks = load_dataset("v1")
    by_id = tasks_by_id(tasks)
    assert by_id[tasks[0].task_id] is tasks[0]


# -- Batch JSONL helpers --


def test_build_batch_requests_keys_by_task_id():
    tasks = load_dataset("v1")[:2]
    requests = build_batch_requests(tasks, "llama-3.1-8b-instant")
    assert len(requests) == 2
    assert requests[0]["custom_id"] == tasks[0].task_id
    assert requests[0]["method"] == "POST"
    assert requests[0]["url"] == "/v1/chat/completions"
    assert requests[0]["body"]["model"] == "llama-3.1-8b-instant"
    assert requests[0]["body"]["messages"][0]["content"] == tasks[0].prompt


def test_write_batch_jsonl_round_trips(tmp_path):
    tasks = load_dataset("v1")[:3]
    requests = build_batch_requests(tasks, "llama-3.1-8b-instant")
    path = tmp_path / "batch_input.jsonl"
    write_batch_jsonl(requests, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["custom_id"] == tasks[0].task_id


def test_parse_batch_output_jsonl():
    raw = (
        '{"custom_id": "task_001", "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 10}}}, "error": null}\n'
        '{"custom_id": "task_002", "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 20}}}, "error": null}\n'
    )
    parsed = parse_batch_output_jsonl(raw)
    assert len(parsed) == 2
    assert parsed[1]["custom_id"] == "task_002"


def test_parse_batch_output_jsonl_marks_malformed_lines_instead_of_dropping_them():
    raw = '{"custom_id": "task_001"}\nnot valid json\n'
    parsed = parse_batch_output_jsonl(raw)
    assert len(parsed) == 2
    assert parsed[1]["_malformed_line"] == "not valid json"
