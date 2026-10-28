from __future__ import annotations

from labs.lab08_concurrency.experiments import DEFAULT_TASK_IDS, PILOT_TASK_IDS, WORKLOAD, WORKLOAD_BY_ID


def test_workload_has_eight_tasks():
    assert len(WORKLOAD) == 8


def test_every_task_id_is_unique():
    ids = [t.task_id for t in WORKLOAD]
    assert len(ids) == len(set(ids))


def test_every_task_has_a_distinct_expected_answer():
    answers = [t.expected_answer for t in WORKLOAD]
    assert len(answers) == len(set(answers))


def test_expected_answers_are_correct_sums():
    expected_by_id = {
        "task_001": "2", "task_002": "4", "task_003": "7", "task_004": "11",
        "task_005": "15", "task_006": "21", "task_007": "27", "task_008": "41",
    }
    for task_id, expected in expected_by_id.items():
        assert WORKLOAD_BY_ID[task_id].expected_answer == expected


def test_pilot_is_a_stable_four_task_subset():
    assert PILOT_TASK_IDS == tuple(t.task_id for t in WORKLOAD[:4])
    assert len(PILOT_TASK_IDS) == 4


def test_default_task_ids_cover_the_full_workload():
    assert DEFAULT_TASK_IDS == tuple(t.task_id for t in WORKLOAD)
    assert len(DEFAULT_TASK_IDS) == 8


def test_workload_by_id_round_trips_every_task():
    for task in WORKLOAD:
        assert WORKLOAD_BY_ID[task.task_id] is task
