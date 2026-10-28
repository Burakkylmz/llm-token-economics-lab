from __future__ import annotations

import json

import pytest

from lab_core.config import Lab08Config
from labs.lab08_concurrency.models import (
    EVENT_REQUEST_COMPLETED,
    ConcurrencyMeasurement,
    PlannedConcurrencyCall,
    RequestTimelineEvent,
)
from labs.lab08_concurrency.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab08_run_dir,
    read_manifest,
    read_raw_measurements,
    refuse_if_already_has_measurements,
    write_condition_summary_csv,
    write_concurrency_comparison_csv,
    write_experiment_plan_csv,
    write_manifest,
    write_raw_measurements,
    write_request_timeline_csv,
)
from labs.lab08_concurrency.stats import ConcurrencyComparison, summarize_condition


def _config() -> Lab08Config:
    return Lab08Config(
        enabled=True, model="openai/gpt-oss-20b", default_profile="default", concurrency_levels=(1, 5),
        repetitions=1, reasoning_effort="low", max_completion_tokens=256, randomize_condition_order=False,
        randomization_seed=42, max_allowed_concurrency=10, sdk_max_retries=0, max_api_calls=32,
        max_estimated_cost_usd=0.05,
    )


def _measurement(task_id="task_001", max_concurrent=1) -> ConcurrencyMeasurement:
    return ConcurrencyMeasurement(
        experiment_run_id="run-1", execution_index=1, repetition_index=1, task_id=task_id,
        max_concurrent=max_concurrent, execution_mode="async_sequential", timestamp="2026-01-01T00:00:00+00:00",
        model="m", reasoning_effort="low", max_completion_tokens=100, request_status="SUCCESS",
        queue_wait_ms=1.0, request_service_latency_ms=100.0, task_end_to_end_latency_ms=101.0,
        prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost_usd=0.001,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    dir_a = lab08_run_dir(tmp_path, "run-a")
    dir_b = lab08_run_dir(tmp_path, "run-b")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent
    assert dir_a.parent.name == "lab08_concurrency"


def test_manifest_round_trip(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", _config(), "default", 16)
    write_manifest(run_dir, manifest)
    loaded = read_manifest(run_dir)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 16
    assert loaded["profile"] == "default"


def test_read_manifest_returns_none_when_absent(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-missing")
    assert read_manifest(run_dir) is None


def test_refuse_if_already_has_measurements_blocks_reuse(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    write_raw_measurements(run_dir, [_measurement()])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_if_already_has_measurements_allows_fresh_run_id(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-fresh")
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_raw_measurements_json_and_csv_round_trip(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    measurements = [_measurement("task_001"), _measurement("task_002")]
    write_raw_measurements(run_dir, measurements)
    assert (run_dir / "raw_measurements.json").exists()
    assert (run_dir / "raw_measurements.csv").exists()
    loaded = read_raw_measurements(run_dir)
    assert len(loaded) == 2
    assert loaded[0].task_id == "task_001"
    assert loaded[0].estimated_cost_usd == 0.001


def test_read_raw_measurements_returns_empty_list_when_absent(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-missing")
    assert read_raw_measurements(run_dir) == []


def test_experiment_plan_csv_written(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    plan = [PlannedConcurrencyCall(execution_index=1, repetition_index=1, max_concurrent=1, task_id="task_001", task_index=0)]
    write_experiment_plan_csv(run_dir, plan)
    content = (run_dir / "experiment_plan.csv").read_text(encoding="utf-8")
    assert "task_001" in content
    assert "execution_index" in content


def test_request_timeline_csv_written(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    events = [RequestTimelineEvent(1, 1, 1, "task_001", EVENT_REQUEST_COMPLETED, 10.0)]
    write_request_timeline_csv(run_dir, events)
    content = (run_dir / "request_timeline.csv").read_text(encoding="utf-8")
    assert "REQUEST_COMPLETED" in content


def test_condition_summary_csv_handles_heterogeneous_nested_fields(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    summary_one = summarize_condition(1, [_measurement("task_001", 1)], [], batch_wall_clock_ms=100.0, observed_peak_in_flight=1)
    summary_five = summarize_condition(5, [], [], batch_wall_clock_ms=None, observed_peak_in_flight=0)
    # Union-of-fieldnames: must not raise even when rows have differently
    # shaped nested dict values (e.g. one summary has no measurements at all).
    write_condition_summary_csv(run_dir, [summary_one, summary_five])
    content = (run_dir / "condition_summary.csv").read_text(encoding="utf-8")
    assert "queue_wait_ms.mean" in content


def test_condition_summary_csv_skips_write_when_no_summaries(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    write_condition_summary_csv(run_dir, [])
    assert not (run_dir / "condition_summary.csv").exists()


def test_concurrency_comparison_csv_written(tmp_path):
    run_dir = lab08_run_dir(tmp_path, "run-1")
    comparison = ConcurrencyComparison(
        baseline_concurrency=1, compared_concurrency=5, throughput_speedup_ratio=2.0,
        batch_time_reduction_percent=40.0, median_service_latency_delta_percent=10.0,
        median_task_end_to_end_latency_delta_percent=-5.0, success_rate_delta=0.0, cost_delta_percent=0.0,
    )
    write_concurrency_comparison_csv(run_dir, [comparison])
    content = (run_dir / "concurrency_comparison.csv").read_text(encoding="utf-8")
    assert "throughput_speedup_ratio" in content
