"""Offline v1 validation and schema-compatibility regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_core.experiment_validation import (
    MAX_ISSUES,
    NORMALIZED_CONTRACT_VERSION,
    _safe_run_id,
    main,
    validate_run,
)
from labs.lab07_rate_limiting.persistence import build_initial_manifest


def make_run(tmp_path: Path, *, manifest_version=1, records=None, write_manifest=True):
    run = tmp_path / "results" / "lab07_rate_limiting" / "one-run"
    run.mkdir(parents=True)
    if write_manifest:
        manifest = {"experiment_run_id": "one-run", "randomization_seed": 42}
        if manifest_version is not None:
            manifest["normalized_contract_version"] = manifest_version
        (run / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if records is None:
        records = [{
            "scenario_id": "transient_failures",
            "strategy_id": "no_backoff",
            "client_id": 0,
            "simulation_run_id": "one-run-transient_failures-no_backoff",
            "succeeded": True,
            "outcome": "SUCCEEDED",
            "total_attempts": 3,
            "retry_count": 2,
            "rate_limit_response_count": 2,
            "total_scheduled_wait_seconds": 0,
            "virtual_completion_time_seconds": 0,
            "secret_prompt": "NEVER EXPORT PRIVATE PROMPT",
        }]
    (run / "raw_measurements.json").write_text(json.dumps(records), encoding="utf-8")
    return run


def codes(result):
    return [diagnostic.code for diagnostic in result.diagnostics]


def test_new_versioned_lab07_run_valid_zero_preserved_and_no_leaks(tmp_path):
    run = make_run(tmp_path)
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    report = result.to_report()
    assert report["valid"] is True
    assert report["normalized_contract_version"] == NORMALIZED_CONTRACT_VERSION == 1
    assert report["declared_normalized_contract_version"] == 1
    assert report["compatibility"] == "v1"
    assert report["files_checked"] == 1
    assert report["records_checked"] == 1
    assert report["errors"] == 0 and report["warnings"] == 0
    assert "PRIVATE PROMPT" not in json.dumps(report)
    assert (run / "raw_measurements.json").exists()


def test_legacy_unversioned_can_be_read_but_strict_mode_blocks(tmp_path, capsys):
    make_run(tmp_path, manifest_version=None)
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert result.to_report()["valid"] is True
    assert result.compatibility == "legacy_unversioned"
    assert "unversioned_manifest" in codes(result)
    args = [
        "--results-dir", str(tmp_path / "results"),
        "--experiment", "lab07_rate_limiting", "--run-id", "one-run",
    ]
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1
    assert "unversioned_manifest" in capsys.readouterr().out


@pytest.mark.parametrize("version,expected", [
    (2, "unsupported_contract_version"),
    (0, "invalid_contract_version"),
    (-1, "invalid_contract_version"),
    ("1", "invalid_contract_version"),
    (True, "invalid_contract_version"),
])
def test_unsupported_or_malformed_versions_fail_closed(tmp_path, version, expected):
    make_run(tmp_path, manifest_version=version)
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert result.to_report()["valid"] is False
    assert expected in codes(result)
    assert result.record_count == 0  # never fall back silently to unversioned


def test_missing_manifest_is_legacy_with_warning(tmp_path):
    make_run(tmp_path, write_manifest=False)
    report = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert report.to_report()["valid"]
    assert report.compatibility == "legacy_unversioned"


def test_bad_manifest_is_error_not_silent_missing(tmp_path):
    run = make_run(tmp_path)
    (run / "experiment_manifest.json").write_text("{", encoding="utf-8")
    report = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert not report.to_report()["valid"]
    assert "invalid_manifest" in codes(report)


def test_corrupt_raw_numbers_rejected_even_if_normalizer_would_hide_them(tmp_path):
    make_run(tmp_path, records=[{
        "scenario_id": "transient_failures",
        "strategy_id": "no_backoff",
        "client_id": 0,
        "succeeded": True,
        "total_attempts": "not-a-number",
        "retry_count": -1,
        "rate_limit_response_count": 3,
        "total_scheduled_wait_seconds": "NaN",
    }])
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert not result.to_report()["valid"]
    found = [item for item in result.diagnostics if item.code == "invalid_raw_value"]
    assert {i.field for i in found} == {
        "total_attempts", "retry_count", "total_scheduled_wait_seconds",
    }
    assert "not-a-number" not in json.dumps(result.to_report())


def test_lab07_cross_field_invariants(tmp_path):
    make_run(tmp_path, records=[{
        "scenario_id": "transient_failures",
        "strategy_id": "no_backoff",
        "client_id": 0,
        "succeeded": True,
        "total_attempts": 2,
        "retry_count": 3,
        "rate_limit_response_count": 4,
        "total_scheduled_wait_seconds": 0,
        "total_tokens": 99,
        "estimated_cost_usd": 0.002,
        "total_latency_ms": 100,
    }])
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert not result.to_report()["valid"]
    assert {
        "retry_count_exceeds_attempts",
        "rate_limits_exceed_attempts",
        "simulation_has_provider_measurement",
    }.issubset(set(codes(result)))


def test_v1_missing_required_lab07_fields_vs_legacy_warning(tmp_path):
    run = make_run(tmp_path, records=[{"succeeded": True}])
    root = tmp_path / "results"
    result = validate_run(root, experiment="lab07_rate_limiting", run_id="one-run")
    assert not result.to_report()["valid"]
    assert "missing_lab07_field" in codes(result)
    (run / "experiment_manifest.json").unlink()
    legacy = validate_run(root, experiment="lab07_rate_limiting", run_id="one-run")
    assert legacy.to_report()["valid"]
    assert legacy.warnings > 0


def test_bad_measurement_and_no_measurement_are_diagnostics(tmp_path):
    run = make_run(tmp_path)
    (run / "raw_measurements.json").write_text("[{bad json", encoding="utf-8")
    result = validate_run(tmp_path / "results", experiment="lab07_rate_limiting", run_id="one-run")
    assert "invalid_measurement_file" in codes(result)
    assert "no_measurement_files" in codes(result)


def test_strict_path_policy_and_nonexistent_run(tmp_path):
    for value in ("../x", "..", ".", "a/b", "a\\b", "C:test", ""):
        assert not _safe_run_id(value)
        with pytest.raises(ValueError):
            validate_run(tmp_path, experiment="lab07_rate_limiting", run_id=value)
    with pytest.raises(ValueError, match="Unknown experiment"):
        validate_run(tmp_path, experiment="made_up", run_id="legit")
    with pytest.raises(ValueError, match="does not exist"):
        validate_run(tmp_path, experiment="lab07_rate_limiting", run_id="legit")


def test_input_symlink_does_not_escape_results_root(tmp_path):
    make_run(tmp_path)
    root = tmp_path / "results"
    external = tmp_path / "external.json"
    external.write_text('{"private":"never display"}', encoding="utf-8")
    file_path = root / "lab07_rate_limiting" / "one-run" / "raw_measurements.json"
    file_path.unlink()
    try:
        file_path.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink unavailable on this system")
    result = validate_run(root, experiment="lab07_rate_limiting", run_id="one-run")
    assert not result.to_report()["valid"]
    assert "invalid_measurement_file" in codes(result)
    assert "private" not in json.dumps(result.to_report())


def test_json_cli_is_deterministic_read_only_and_output_guarded(tmp_path, capsys):
    run = make_run(tmp_path)
    before = (run / "raw_measurements.json").read_bytes()
    args = [
        "--results-dir", str(tmp_path / "results"),
        "--experiment", "lab07_rate_limiting", "--run-id", "one-run", "--format", "json",
    ]
    assert main(args) == 0
    output1 = capsys.readouterr().out
    assert main(args) == 0
    output2 = capsys.readouterr().out
    assert output1 == output2
    assert json.loads(output1)["valid"] is True
    assert (run / "raw_measurements.json").read_bytes() == before
    with pytest.raises(SystemExit) as invalid:
        main([*args, "--output", str(run / "experiment_manifest.json")])
    assert invalid.value.code == 2
    assert (run / "raw_measurements.json").read_bytes() == before
    report_path = tmp_path / "reports" / "v1.json"
    assert main([*args, "--output", str(report_path)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"]


def test_legacy_provider_family_does_not_fabricate_required_model_usage(tmp_path):
    run = tmp_path / "results" / "lab05_model_routing" / "old"
    run.mkdir(parents=True)
    (run / "routing_measurements.json").write_text(json.dumps([
        {
            "model": "provider-model",
            "request_status": "SUCCESS",
            "total_tokens": "0",
            "total_latency_ms": "0",
        },
    ]), encoding="utf-8")
    result = validate_run(tmp_path / "results", experiment="lab05_model_routing", run_id="old")
    assert result.to_report()["valid"]
    assert result.compatibility == "legacy_unversioned"
    assert result.record_count == 1


def test_contract_field_groups_align_with_existing_normalized_export():
    from lab_core.dashboard_data import EXPORT_COLUMNS
    from lab_core.experiment_validation import (
        BOOLEAN_FIELDS, INTEGER_FIELDS, NUMBER_FIELDS, TEXT_FIELDS,
    )
    assert (TEXT_FIELDS | BOOLEAN_FIELDS | INTEGER_FIELDS | NUMBER_FIELDS) == set(EXPORT_COLUMNS)
    assert not any(
        x & y for x, y in (
            (TEXT_FIELDS, BOOLEAN_FIELDS), (TEXT_FIELDS, INTEGER_FIELDS),
            (TEXT_FIELDS, NUMBER_FIELDS), (BOOLEAN_FIELDS, INTEGER_FIELDS),
            (BOOLEAN_FIELDS, NUMBER_FIELDS), (INTEGER_FIELDS, NUMBER_FIELDS),
        )
    )


def test_report_diagnostics_bounded(tmp_path):
    from lab_core.experiment_validation import ValidationResult
    result = ValidationResult(experiment="lab07_rate_limiting", run_id="one")
    for i in range(MAX_ISSUES + 10):
        result.issue("warning", "test_only", "raw_measurements.json", row=i)
    report = result.to_report()
    assert len(report["diagnostics"]) == MAX_ISSUES
    assert report["diagnostics_truncated"] == 10
    assert report["valid"] is False  # truncation requires investigation


def test_new_lab07_manifests_declare_normalized_contract_version():
    # Build a light config stand-in; only asdict() is used by this function.
    from dataclasses import dataclass

    @dataclass
    class Config:
        enabled: bool = True

    manifest = build_initial_manifest("run", Config(), "pilot", 4, 42)
    assert manifest["normalized_contract_version"] == 1
    assert manifest["simulation_version"] == "v1.0-virtual-clock"
