from __future__ import annotations

import json

import pytest

from labs.lab07_rate_limiting.models import AttemptRecord, ClientSimulationResult, OUTCOME_SUCCEEDED
from labs.lab07_rate_limiting.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab07_run_dir,
    read_raw_measurements,
    refuse_if_already_has_measurements,
    write_attempt_timeline_csv,
    write_manifest,
    write_raw_measurements,
    write_scenario_comparison_csv,
    write_strategy_summary_csv,
)
from labs.lab07_rate_limiting.stats import StrategySummary


def _client_result(client_id=0) -> ClientSimulationResult:
    return ClientSimulationResult(
        scenario_id="transient_failures", strategy_id="no_backoff", simulation_run_id="run-1", client_id=client_id,
        succeeded=True, outcome=OUTCOME_SUCCEEDED, total_attempts=1, retry_count=0, rate_limit_response_count=0,
        total_scheduled_wait_seconds=0.0, virtual_completion_time_seconds=0.0,
    )


def _attempt_record() -> AttemptRecord:
    return AttemptRecord(
        scenario_id="transient_failures", strategy_id="no_backoff", simulation_run_id="run-1", client_id=0,
        attempt_number=1, retry_number=0, virtual_timestamp_seconds=0.0, http_status=200, retry_after_header=None,
        scheduled_delay_seconds=None, delay_reason="none", final_outcome=OUTCOME_SUCCEEDED,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab07_run_dir(tmp_path, "run-a")
    dir_b = lab07_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent


def test_manifest_round_trip(tmp_path):
    from lab_core.config import Lab07Config

    config = Lab07Config(
        enabled=True, default_profile="default", randomization_seed=42, max_simulation_events=10000, max_clients=50,
        max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=30.0, max_total_wait_seconds=120.0,
        simulation_deadline_seconds=None, transient_fail_count=2, recovery_time_seconds=3.0, contention_client_count=8,
        contention_capacity_per_window=2, contention_window_seconds=1.0, quota_client_count=3, quota_rpm=5, quota_tpm=250,
        quota_token_demand_per_request=100, quota_window_seconds=60.0,
    )
    run_dir = lab07_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config, "default", 12, 42)
    write_manifest(run_dir, manifest)

    with open(run_dir / "experiment_manifest.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_simulation_runs"] == 12
    assert loaded["real_api_calls"] == 0
    assert loaded["real_token_usage"] is None


def test_raw_measurements_json_and_csv_have_matching_counts(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [_client_result(0), _client_result(1)])

    assert (run_dir / "raw_measurements.json").exists()
    assert (run_dir / "raw_measurements.csv").exists()
    with open(run_dir / "raw_measurements.json", encoding="utf-8") as f:
        json_rows = json.load(f)
    assert len(json_rows) == 2


def test_read_raw_measurements_round_trips(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [_client_result(0)])
    loaded = read_raw_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].scenario_id == "transient_failures"


def test_attempt_timeline_csv_written(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_attempt_timeline_csv(run_dir, [_attempt_record(), _attempt_record()])
    assert (run_dir / "attempt_timeline.csv").exists()


def test_scenario_comparison_pivots_by_strategy(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    summaries = [
        StrategySummary("s1", "a", 1, 1, 0, 1.0, 1, 0, 0, 1.0, 0.0, 0.0, 0.0, None, 5.0, 0),
        StrategySummary("s2", "a", 1, 1, 0, 1.0, 1, 0, 0, 1.0, 0.0, 0.0, 0.0, None, 7.0, 0),
    ]
    write_scenario_comparison_csv(run_dir, summaries)
    with open(run_dir / "scenario_comparison.csv", encoding="utf-8") as f:
        content = f.read()
    assert "s1.completion_time_seconds" in content
    assert "s2.completion_time_seconds" in content
    assert content.count("\n") == 2  # header + 1 row (strategy "a")


def test_refuse_allows_a_fresh_run_dir(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    refuse_if_already_has_measurements(run_dir)


def test_refuse_blocks_a_duplicate_run_id(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [_client_result(0)])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_does_not_block_on_empty_measurement_list(tmp_path):
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [])
    refuse_if_already_has_measurements(run_dir)


def test_legacy_flat_file_and_new_run_dir_do_not_collide(tmp_path):
    (tmp_path / "lab07_rate_limiting.json").write_text("[]", encoding="utf-8")
    run_dir = lab07_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [_client_result(0)])
    assert (tmp_path / "lab07_rate_limiting.json").read_text(encoding="utf-8") == "[]"
