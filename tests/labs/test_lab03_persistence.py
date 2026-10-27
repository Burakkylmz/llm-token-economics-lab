from __future__ import annotations

import json

import pytest

from labs.lab03_prompt_caching.models import CacheMeasurement
from labs.lab03_prompt_caching.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab03_run_dir,
    read_cache_measurements,
    refuse_if_already_has_measurements,
    write_cache_measurements,
    write_manifest,
)


def _measurement(execution_index=1, succeeded=True) -> CacheMeasurement:
    return CacheMeasurement(
        experiment_run_id="run-1",
        condition_id="repeated_prefix",
        execution_index=execution_index,
        pair_index=1,
        question_index=0,
        timestamp="t",
        model="m",
        max_completion_tokens=100,
        succeeded=succeeded,
        attempts_used=1,
        prompt_tokens=100,
        cached_tokens=50,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab03_run_dir(tmp_path, "run-a")
    dir_b = lab03_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent


def test_manifest_round_trip(tmp_path):
    from lab_core.config import Lab03Config

    config = Lab03Config(
        caching_capable_models=("openai/gpt-oss-20b",),
        repetitions=3,
        max_completion_tokens=2048,
        randomize_condition_order_within_pair=False,
        randomization_seed=42,
        max_api_calls=8,
        max_estimated_cost_usd=0.05,
        cache_probe_calls=3,
    )
    run_dir = lab03_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config, "openai/gpt-oss-20b", 6)
    write_manifest(run_dir, manifest)

    with open(run_dir / "experiment_manifest.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 6


def test_cache_measurements_json_and_csv_have_matching_counts(tmp_path):
    run_dir = lab03_run_dir(tmp_path, "run-1")
    measurements = [_measurement(1), _measurement(2)]
    write_cache_measurements(run_dir, measurements)

    assert (run_dir / "cache_measurements.json").exists()
    assert (run_dir / "cache_measurements.csv").exists()
    with open(run_dir / "cache_measurements.json", encoding="utf-8") as f:
        json_rows = json.load(f)
    with open(run_dir / "cache_measurements.csv", encoding="utf-8") as f:
        csv_rows = f.read().strip().splitlines()
    assert len(json_rows) == 2
    assert len(csv_rows) - 1 == 2  # minus header


def test_read_cache_measurements_round_trips(tmp_path):
    run_dir = lab03_run_dir(tmp_path, "run-1")
    write_cache_measurements(run_dir, [_measurement(1)])
    loaded = read_cache_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].condition_id == "repeated_prefix"


def test_refuse_allows_a_fresh_run_dir(tmp_path):
    run_dir = lab03_run_dir(tmp_path, "run-1")
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_refuse_blocks_a_duplicate_run_id(tmp_path):
    run_dir = lab03_run_dir(tmp_path, "run-1")
    write_cache_measurements(run_dir, [_measurement(1)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_does_not_block_on_empty_measurement_list(tmp_path):
    run_dir = lab03_run_dir(tmp_path, "run-1")
    write_cache_measurements(run_dir, [])
    refuse_if_already_has_measurements(run_dir)  # must not raise -- empty list is not "has measurements"


def test_legacy_flat_file_and_new_run_dir_do_not_collide(tmp_path):
    (tmp_path / "lab03_prompt_caching.json").write_text("[]", encoding="utf-8")
    run_dir = lab03_run_dir(tmp_path, "run-1")
    write_cache_measurements(run_dir, [_measurement(1)])
    assert (tmp_path / "lab03_prompt_caching.json").read_text(encoding="utf-8") == "[]"
