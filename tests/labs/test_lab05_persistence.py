from __future__ import annotations

import json

import pytest

from labs.lab05_model_routing.models import RoutingMeasurement
from labs.lab05_model_routing.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab05_run_dir,
    read_routing_measurements,
    refuse_if_already_has_measurements,
    write_manifest,
    write_routing_measurements,
)


def _measurement(execution_index=1) -> RoutingMeasurement:
    return RoutingMeasurement(
        experiment_run_id="run-1", execution_index=execution_index, strategy="always_small",
        task_id="task_001", task_category="factual_retrieval", complexity_label="simple", timestamp="t",
        selected_model="m", model_tier="small", routing_reason="r", succeeded=True, attempts_used=1,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab05_run_dir(tmp_path, "run-a")
    dir_b = lab05_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent


def test_manifest_round_trip(tmp_path):
    from lab_core.config import Lab05Config

    config = Lab05Config(
        small_model="small-m", large_model="large-m", default_profile="default", reasoning_effort="low",
        max_completion_tokens=1024, randomize_task_order=False, randomization_seed=42, max_api_calls=18,
        max_estimated_cost_usd=0.05,
    )
    run_dir = lab05_run_dir(tmp_path, "run-1")
    from lab_core.config import ModelPricing

    pricing = {
        "small-m": ModelPricing(0.1, 0.2, 0.05, "2026-10-31"),
        "large-m": ModelPricing(0.3, 0.6, 0.15, "2026-10-31"),
    }
    manifest = build_initial_manifest("run-1", config, "default", 12, pricing)
    write_manifest(run_dir, manifest)

    with open(run_dir / "experiment_manifest.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 12
    assert loaded["profile"] == "default"
    assert loaded["pricing_snapshot"]["small-m"]["input_per_1m_usd"] == 0.1
    assert loaded["pricing_snapshot"]["large-m"]["output_per_1m_usd"] == 0.6
    assert loaded["pricing_snapshot"]["small-m"]["price_verified_date"] == "2026-10-31"


def test_routing_measurements_json_and_csv_have_matching_counts(tmp_path):
    run_dir = lab05_run_dir(tmp_path, "run-1")
    write_routing_measurements(run_dir, [_measurement(1), _measurement(2)])

    assert (run_dir / "routing_measurements.json").exists()
    assert (run_dir / "routing_measurements.csv").exists()
    with open(run_dir / "routing_measurements.json", encoding="utf-8") as f:
        json_rows = json.load(f)
    with open(run_dir / "routing_measurements.csv", encoding="utf-8") as f:
        csv_rows = f.read().strip().splitlines()
    assert len(json_rows) == 2
    assert len(csv_rows) - 1 == 2


def test_read_routing_measurements_round_trips(tmp_path):
    run_dir = lab05_run_dir(tmp_path, "run-1")
    write_routing_measurements(run_dir, [_measurement(1)])
    loaded = read_routing_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].task_id == "task_001"


def test_refuse_allows_a_fresh_run_dir(tmp_path):
    run_dir = lab05_run_dir(tmp_path, "run-1")
    refuse_if_already_has_measurements(run_dir)


def test_refuse_blocks_a_duplicate_run_id(tmp_path):
    run_dir = lab05_run_dir(tmp_path, "run-1")
    write_routing_measurements(run_dir, [_measurement(1)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_does_not_block_on_empty_measurement_list(tmp_path):
    run_dir = lab05_run_dir(tmp_path, "run-1")
    write_routing_measurements(run_dir, [])
    refuse_if_already_has_measurements(run_dir)


def test_legacy_flat_file_and_new_run_dir_do_not_collide(tmp_path):
    (tmp_path / "lab05_model_routing.json").write_text("[]", encoding="utf-8")
    run_dir = lab05_run_dir(tmp_path, "run-1")
    write_routing_measurements(run_dir, [_measurement(1)])
    assert (tmp_path / "lab05_model_routing.json").read_text(encoding="utf-8") == "[]"
