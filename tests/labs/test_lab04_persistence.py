from __future__ import annotations

import json

import pytest

from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement
from labs.lab04_output_control.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab04_run_dir,
    read_output_limit_measurements,
    refuse_if_already_has_measurements,
    write_invocation_measurements,
    write_manifest,
    write_output_limit_measurements,
)


def _ol_measurement(execution_index=1, succeeded=True) -> OutputLimitMeasurement:
    return OutputLimitMeasurement(
        experiment_run_id="run-1", condition_id="restrictive", execution_index=execution_index,
        repetition_index=1, timestamp="t", model="m", max_completion_tokens=64, reasoning_effort="low",
        succeeded=succeeded, attempts_used=1, prompt_tokens=100, completion_tokens=64,
    )


def _inv_measurement(execution_index=1) -> InvocationMeasurement:
    return InvocationMeasurement(
        experiment_run_id="run-1", strategy="single", execution_index=execution_index, question_index=None,
        timestamp="t", model="m", max_completion_tokens=2048, reasoning_effort="low",
        succeeded=True, attempts_used=1,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab04_run_dir(tmp_path, "run-a")
    dir_b = lab04_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent


def test_manifest_round_trip(tmp_path):
    from lab_core.config import Lab04Config

    config = Lab04Config(
        enabled=True, model="openai/gpt-oss-20b", output_limit_values=(64, 256, 1024),
        output_limit_repetitions=2, reasoning_effort="low", invocation_max_completion_tokens=2048,
        invocation_execution_mode="sequential", max_api_calls=10, max_estimated_cost_usd=0.05,
    )
    run_dir = lab04_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config, "openai/gpt-oss-20b", 10)
    write_manifest(run_dir, manifest)

    with open(run_dir / "experiment_manifest.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 10


def test_output_limit_measurements_json_and_csv_have_matching_counts(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_output_limit_measurements(run_dir, [_ol_measurement(1), _ol_measurement(2)])

    assert (run_dir / "output_limit_measurements.json").exists()
    assert (run_dir / "output_limit_measurements.csv").exists()
    with open(run_dir / "output_limit_measurements.json", encoding="utf-8") as f:
        json_rows = json.load(f)
    with open(run_dir / "output_limit_measurements.csv", encoding="utf-8") as f:
        csv_rows = f.read().strip().splitlines()
    assert len(json_rows) == 2
    assert len(csv_rows) - 1 == 2


def test_invocation_measurements_written_separately_from_output_limit(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_invocation_measurements(run_dir, [_inv_measurement(1)])
    assert (run_dir / "invocation_measurements.json").exists()
    assert not (run_dir / "output_limit_measurements.json").exists()


def test_read_output_limit_measurements_round_trips(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_output_limit_measurements(run_dir, [_ol_measurement(1)])
    loaded = read_output_limit_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].condition_id == "restrictive"


def test_refuse_allows_a_fresh_run_dir(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_refuse_blocks_a_duplicate_run_id_via_output_limit_file(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_output_limit_measurements(run_dir, [_ol_measurement(1)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_blocks_a_duplicate_run_id_via_invocation_file(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_invocation_measurements(run_dir, [_inv_measurement(1)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_does_not_block_on_empty_measurement_lists(tmp_path):
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_output_limit_measurements(run_dir, [])
    write_invocation_measurements(run_dir, [])
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_legacy_flat_file_and_new_run_dir_do_not_collide(tmp_path):
    (tmp_path / "lab04_output_control.json").write_text("[]", encoding="utf-8")
    run_dir = lab04_run_dir(tmp_path, "run-1")
    write_output_limit_measurements(run_dir, [_ol_measurement(1)])
    assert (tmp_path / "lab04_output_control.json").read_text(encoding="utf-8") == "[]"
