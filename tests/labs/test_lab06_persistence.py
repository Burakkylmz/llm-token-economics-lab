from __future__ import annotations

import json

import pytest

from labs.lab06_streaming.models import CONDITION_STREAMING, StreamEventRecord, StreamMeasurement
from labs.lab06_streaming.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab06_run_dir,
    read_stream_measurements,
    refuse_if_already_has_measurements,
    write_manifest,
    write_stream_events_csv,
    write_stream_measurements,
)


def _measurement(execution_index=1) -> StreamMeasurement:
    return StreamMeasurement(
        experiment_run_id="run-1", execution_index=execution_index, pair_id=1, condition=CONDITION_STREAMING,
        timestamp="t", model="m", reasoning_effort="low", max_completion_tokens=100, request_hash="h",
        succeeded=True, attempts_used=1,
    )


def _event(execution_index=1, event_index=1) -> StreamEventRecord:
    return StreamEventRecord(
        execution_index=execution_index, event_index=event_index, relative_arrival_ms=10.0, event_type="VISIBLE_CONTENT",
        visible_content_character_count=5, cumulative_visible_character_count=5, has_reasoning_delta=False,
        has_finish_reason=False, has_usage=False,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab06_run_dir(tmp_path, "run-a")
    dir_b = lab06_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent


def test_manifest_round_trip(tmp_path):
    from lab_core.config import Lab06Config

    config = Lab06Config(
        enabled=True, model="openai/gpt-oss-20b", default_profile="default", paired_repetitions=5,
        reasoning_effort="low", max_completion_tokens=768, randomize_condition_order=False, randomization_seed=42,
        request_usage_when_supported=True, record_event_timeline=True, max_api_calls=16, max_estimated_cost_usd=0.05,
    )
    run_dir = lab06_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config, "default", 10)
    write_manifest(run_dir, manifest)

    with open(run_dir / "experiment_manifest.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 10
    assert loaded["profile"] == "default"


def test_stream_measurements_json_and_csv_have_matching_counts(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_measurements(run_dir, [_measurement(1), _measurement(2)])

    assert (run_dir / "stream_measurements.json").exists()
    assert (run_dir / "stream_measurements.csv").exists()
    with open(run_dir / "stream_measurements.json", encoding="utf-8") as f:
        json_rows = json.load(f)
    with open(run_dir / "stream_measurements.csv", encoding="utf-8") as f:
        csv_rows = f.read().strip().splitlines()
    assert len(json_rows) == 2
    assert len(csv_rows) - 1 == 2


def test_read_stream_measurements_round_trips(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_measurements(run_dir, [_measurement(1)])
    loaded = read_stream_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].condition == CONDITION_STREAMING


def test_stream_events_csv_written_when_events_present(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_events_csv(run_dir, [_event(1, 1), _event(1, 2)])
    assert (run_dir / "stream_events.csv").exists()
    with open(run_dir / "stream_events.csv", encoding="utf-8") as f:
        rows = f.read().strip().splitlines()
    assert len(rows) - 1 == 2


def test_stream_events_csv_not_written_when_no_events(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_events_csv(run_dir, [])
    assert not (run_dir / "stream_events.csv").exists()


def test_refuse_allows_a_fresh_run_dir(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    refuse_if_already_has_measurements(run_dir)


def test_refuse_blocks_a_duplicate_run_id(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_measurements(run_dir, [_measurement(1)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_does_not_block_on_empty_measurement_list(tmp_path):
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_measurements(run_dir, [])
    refuse_if_already_has_measurements(run_dir)


def test_legacy_flat_file_and_new_run_dir_do_not_collide(tmp_path):
    (tmp_path / "lab06_streaming.json").write_text("[]", encoding="utf-8")
    run_dir = lab06_run_dir(tmp_path, "run-1")
    write_stream_measurements(run_dir, [_measurement(1)])
    assert (tmp_path / "lab06_streaming.json").read_text(encoding="utf-8") == "[]"
