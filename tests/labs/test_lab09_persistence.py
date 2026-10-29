from __future__ import annotations

import json

import pytest

from labs.lab09_batch_processing.models import (
    LOCAL_STATE_NOT_SUBMITTED,
    LOCAL_STATE_SUBMITTED,
    RECON_COMPLETED,
    STATUS_SUCCESS,
    BatchJobMetadata,
    BatchRequestMeasurement,
    RealtimeMeasurement,
)
from labs.lab09_batch_processing.persistence import (
    RunIdAlreadyUsed,
    lab09_run_dir,
    read_batch_job,
    read_batch_measurements,
    read_manifest,
    read_realtime_measurements,
    refuse_if_already_has_measurements,
    write_batch_job,
    write_batch_measurements,
    write_manifest,
    write_realtime_measurements,
)


def _measurement(task_id="task_001", answer_correct=True):
    return RealtimeMeasurement(
        experiment_run_id="run1", execution_index=1, task_id=task_id, category="classification_sentiment",
        condition="sequential", max_concurrent=1, timestamp="2026-10-29T00:00:00Z", model="m",
        reasoning_effort="low", max_completion_tokens=100, request_status=STATUS_SUCCESS,
        answer_correct=answer_correct,
    )


def test_lab09_run_dir_is_isolated_per_run_id(tmp_path):
    dir1 = lab09_run_dir(tmp_path, "run1")
    dir2 = lab09_run_dir(tmp_path, "run2")
    assert dir1 != dir2
    assert dir1.parent == dir2.parent


def test_manifest_round_trips(tmp_path):
    run_dir = tmp_path / "run1"
    manifest = {"experiment_run_id": "run1", "profile": "default"}
    write_manifest(run_dir, manifest)
    loaded = read_manifest(run_dir)
    assert loaded["experiment_run_id"] == "run1"


def test_read_manifest_returns_none_when_missing(tmp_path):
    assert read_manifest(tmp_path / "nope") is None


def test_realtime_measurements_round_trip_preserves_optional_bool(tmp_path):
    run_dir = tmp_path / "run1"
    measurements = [_measurement(answer_correct=True), _measurement("task_002", answer_correct=False), _measurement("task_003", answer_correct=None)]
    write_realtime_measurements(run_dir, "sequential", measurements)
    loaded = read_realtime_measurements(run_dir, "sequential")
    assert [m.answer_correct for m in loaded] == [True, False, None]
    assert (run_dir / "sequential_measurements.csv").exists()


def test_refuse_if_already_has_measurements(tmp_path):
    run_dir = tmp_path / "run1"
    write_realtime_measurements(run_dir, "sequential", [_measurement()])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_if_already_has_measurements_allows_empty_run(tmp_path):
    run_dir = tmp_path / "run1"
    write_realtime_measurements(run_dir, "sequential", [])
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_batch_job_round_trips_and_is_atomic(tmp_path):
    run_dir = tmp_path / "run1"
    job = BatchJobMetadata(experiment_run_id="run1", workload_version="v1", model="m", local_lifecycle_state=LOCAL_STATE_SUBMITTED, batch_id="batch_123")
    write_batch_job(run_dir, job)
    loaded = read_batch_job(run_dir)
    assert loaded.batch_id == "batch_123"
    assert loaded.local_lifecycle_state == LOCAL_STATE_SUBMITTED
    assert not (run_dir / "batch_job.json.tmp").exists()


def test_read_batch_job_returns_none_when_missing(tmp_path):
    assert read_batch_job(tmp_path / "nope") is None


def test_batch_measurements_round_trip(tmp_path):
    run_dir = tmp_path / "run1"
    measurements = [
        BatchRequestMeasurement(experiment_run_id="run1", task_id="task_001", category="x", custom_id="task_001", reconciliation_status=RECON_COMPLETED, answer_correct=True),
    ]
    write_batch_measurements(run_dir, measurements)
    loaded = read_batch_measurements(run_dir)
    assert loaded[0].custom_id == "task_001"
    assert loaded[0].answer_correct is True


def test_batch_job_default_state_is_not_submitted():
    job = BatchJobMetadata(experiment_run_id="run1", workload_version="v1", model="m")
    assert job.local_lifecycle_state == LOCAL_STATE_NOT_SUBMITTED
    assert job.batch_id is None
