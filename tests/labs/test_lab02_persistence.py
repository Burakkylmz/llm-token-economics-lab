from __future__ import annotations

from lab_core.config import Lab02Config
from labs.lab02_context_growth.models import ContextMeasurement, TurnMeasurement
from labs.lab02_context_growth.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab02_run_dir,
    read_context_measurements,
    read_manifest,
    read_multi_turn_measurements,
    refuse_if_already_has_measurements,
    write_context_measurements,
    write_manifest,
    write_multi_turn_measurements,
)


def _lab02_config(**overrides) -> Lab02Config:
    defaults = dict(
        enabled=True, repetitions=2, multi_turn_turns=4, max_completion_tokens=2048,
        randomize_context_order=False, randomization_seed=42, max_api_calls=10, max_estimated_cost_usd=0.05,
    )
    defaults.update(overrides)
    return Lab02Config(**defaults)


def _context_measurement(condition_id="minimal_context", succeeded=True):
    return ContextMeasurement(
        experiment_run_id="run-1", condition_id=condition_id, execution_index=1, repetition_index=1,
        timestamp="2026-10-27T00:00:00+00:00", model="m", max_completion_tokens=2048, succeeded=succeeded, attempts_used=1,
    )


def _turn_measurement(turn_index=1):
    return TurnMeasurement(
        experiment_run_id="run-1", turn_index=turn_index, timestamp="2026-10-27T00:00:00+00:00",
        model="m", max_completion_tokens=2048, messages_count=2, succeeded=True, attempts_used=1,
    )


def test_run_dir_is_isolated_per_experiment_run_id(tmp_path):
    results_dir = tmp_path / "results"
    dir_a = lab02_run_dir(results_dir, "lab02-aaa")
    dir_b = lab02_run_dir(results_dir, "lab02-bbb")
    assert dir_a != dir_b
    assert dir_a.parent == dir_b.parent == results_dir / "lab02_context_growth"


def test_manifest_round_trips(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    manifest = build_initial_manifest("run-1", _lab02_config(), "m", planned_context_calls=6, planned_turn_calls=4)
    write_manifest(run_dir, manifest)
    reloaded = read_manifest(run_dir)
    assert reloaded["experiment_run_id"] == "run-1"
    assert reloaded["planned_calls"] == 10


def test_context_measurements_json_and_csv_counts_match(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    measurements = [_context_measurement(), _context_measurement()]
    write_context_measurements(run_dir, measurements)

    assert (run_dir / "context_measurements.json").exists()
    assert (run_dir / "context_measurements.csv").exists()
    reloaded = read_context_measurements(run_dir)
    assert len(reloaded) == 2

    import csv

    with open(run_dir / "context_measurements.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_multi_turn_measurements_round_trip(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    write_multi_turn_measurements(run_dir, [_turn_measurement(1), _turn_measurement(2)])
    reloaded = read_multi_turn_measurements(run_dir)
    assert [t.turn_index for t in reloaded] == [1, 2]


def test_refuse_if_already_has_measurements_allows_fresh_run_dir(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    refuse_if_already_has_measurements(run_dir)  # must not raise: nothing written yet


def test_refuse_if_already_has_measurements_blocks_duplicate_run_id(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    write_context_measurements(run_dir, [_context_measurement()])

    try:
        refuse_if_already_has_measurements(run_dir)
        assert False, "expected RunIdAlreadyUsed"
    except RunIdAlreadyUsed:
        pass


def test_refuse_if_already_has_measurements_checks_multi_turn_too(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    write_multi_turn_measurements(run_dir, [_turn_measurement()])

    try:
        refuse_if_already_has_measurements(run_dir)
        assert False, "expected RunIdAlreadyUsed"
    except RunIdAlreadyUsed:
        pass


def test_empty_measurement_files_do_not_block_a_run(tmp_path):
    run_dir = tmp_path / "results" / "lab02_context_growth" / "run-1"
    write_context_measurements(run_dir, [])  # e.g. a run that stopped before any call completed
    refuse_if_already_has_measurements(run_dir)  # must not raise


def test_legacy_flat_files_are_a_distinct_filesystem_path(tmp_path):
    results_dir = tmp_path / "results"
    legacy_file = results_dir / "lab02_context_growth.json"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("[]", encoding="utf-8")

    run_dir = lab02_run_dir(results_dir, "lab02-new-run")
    write_context_measurements(run_dir, [_context_measurement()])

    # The legacy file (a file) and the new per-run directory (a directory
    # with the same stem) coexist without collision.
    assert legacy_file.is_file()
    assert legacy_file.read_text(encoding="utf-8") == "[]"  # untouched
    assert run_dir.is_dir()
