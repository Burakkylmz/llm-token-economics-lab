"""Regression tests for matched-condition, descriptive comparison and CLI safety."""

from __future__ import annotations

import json

import pytest

from lab_core.benchmark_report import build_benchmark
from lab_core.experiment_comparison import compare_benchmark, main, render_markdown


def row(**changes):
    base = {
        "lab": "lab07", "experiment": "lab07_rate_limiting", "run_id": "pilot",
        "source": "raw_measurements.json", "scenario": "transient_failures",
        "model": "Simulation (no LLM)", "condition": "no_backoff",
        "cost_basis": "Unavailable", "simulated": True,
        "succeeded": True, "answer_correct": None,
        "retry_count": 1, "rate_limit_response_count": 1,
        "total_scheduled_wait_seconds": 0.0,
        "total_tokens": None, "estimated_cost_usd": None, "latency_ms": None,
    }
    return {**base, **changes}


def sample_report():
    records = [
        row(succeeded=True, retry_count=1, rate_limit_response_count=1),
        row(succeeded=False, retry_count=2, rate_limit_response_count=2),
        row(condition="rate_limit_aware", succeeded=True, retry_count=1,
            rate_limit_response_count=1, total_scheduled_wait_seconds=2.0),
        row(condition="rate_limit_aware", succeeded=True, retry_count=1,
            rate_limit_response_count=0, total_scheduled_wait_seconds=4.0),
    ]
    return build_benchmark(records, lab="lab07", run_id="pilot", scenario="transient_failures")


def test_lab07_deltas_rates_denominators_and_zero_relative():
    result = compare_benchmark(
        sample_report(),
        baseline_condition="no_backoff",
        candidate_condition="rate_limit_aware",
    )
    assert result["simulated"]
    assert result["comparison_report_version"] == 1
    assert result["baseline"]["requests"] == 2
    assert result["candidate"]["requests"] == 2
    success = result["metrics"]["success_rate"]
    assert success["baseline"] == 0.5 and success["candidate"] == 1.0
    assert success["absolute_delta"] == 0.5
    assert success["percentage_point_delta"] == 50
    assert success["relative_delta_percent"] == 100
    retry = result["metrics"]["retry_count"]
    assert retry["baseline"] == 1.5 and retry["candidate"] == 1
    assert retry["absolute_delta"] == -0.5
    wait = result["metrics"]["total_scheduled_wait_seconds"]
    assert wait["baseline"] == 0
    assert wait["candidate"] == 3
    assert wait["absolute_delta"] == 3
    assert wait["relative_delta_percent"] is None
    assert result["metrics"]["total_tokens"]["absolute_delta"] is None
    assert result["metrics"]["total_tokens"]["baseline_observed"] == 0
    assert result["metrics"]["answer_accuracy"]["absolute_delta"] is None
    assert "synthetic" in " ".join(result["notes"]).lower()


@pytest.mark.parametrize("baseline,candidate", [
    ("no_backoff", "no_backoff"),
    ("not-found", "rate_limit_aware"),
    ("no_backoff", "not-found"),
])
def test_conditions_must_exist_and_differ(baseline, candidate):
    with pytest.raises(ValueError):
        compare_benchmark(sample_report(), baseline_condition=baseline,
                          candidate_condition=candidate)


def test_multiple_experiments_refuses_ambiguity_then_accepts_filter():
    report = sample_report()
    report["groups"] += [
        {**group, "experiment": "another_experiment"} for group in report["groups"]
    ]
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_benchmark(report, baseline_condition="no_backoff",
                          candidate_condition="rate_limit_aware")
    fixed = compare_benchmark(
        report, baseline_condition="no_backoff", candidate_condition="rate_limit_aware",
        experiment="lab07_rate_limiting",
    )
    assert fixed["context"]["experiment"] == "lab07_rate_limiting"


@pytest.mark.parametrize("field", ["experiment", "source", "scenario", "cost_basis"])
def test_unmatched_provenance_rejected(field):
    report = sample_report()
    report["groups"][1][field] = "different"
    with pytest.raises(ValueError, match=field):
        compare_benchmark(report, baseline_condition="no_backoff",
                          candidate_condition="rate_limit_aware")


def test_different_models_require_acknowledgement_and_remain_exploratory():
    records = [
        row(lab="lab05", experiment="lab05_model_routing", simulated=False,
            source="routing_measurements.json", model="small-model",
            condition="always_small", cost_basis="Estimated",
            total_tokens=10, latency_ms=100, estimated_cost_usd=0.001),
        row(lab="lab05", experiment="lab05_model_routing", simulated=False,
            source="routing_measurements.json", model="large-model",
            condition="always_large", cost_basis="Estimated",
            total_tokens=20, latency_ms=200, estimated_cost_usd=0.003),
    ]
    report = build_benchmark(records, lab="lab05", run_id="pilot")
    with pytest.raises(ValueError, match="Model identifiers differ"):
        compare_benchmark(report, baseline_condition="always_small",
                          candidate_condition="always_large")
    result = compare_benchmark(
        report, baseline_condition="always_small", candidate_condition="always_large",
        allow_model_difference=True,
    )
    assert result["model_difference_acknowledged"]
    assert result["metrics"]["total_tokens"]["absolute_delta"] == 10
    assert result["metrics"]["estimated_cost_usd"]["absolute_delta"] == pytest.approx(0.002)
    assert result["metrics"]["latency_ms"]["relative_delta_percent"] == 100
    assert result["metrics"]["latency_p50_ms"]["absolute_delta"] == 100


def test_incomplete_data_does_not_invent_zero_or_percentage():
    report = sample_report()
    report["groups"][1]["metrics"]["retry_count"] = {
        "observed": 0, "mean": None, "p50": None, "p95": None,
    }
    compared = compare_benchmark(report, baseline_condition="no_backoff",
                                 candidate_condition="rate_limit_aware")
    metric = compared["metrics"]["retry_count"]
    assert metric["candidate"] is None
    assert metric["baseline_observed"] == 2
    assert metric["candidate_observed"] == 0
    assert metric["absolute_delta"] is None
    assert metric["relative_delta_percent"] is None


def test_markdown_deterministic_safe_and_documents_limitations():
    report = sample_report()
    report["groups"][1]["condition"] = "rate_limit|aware<script>"
    result = compare_benchmark(
        report, baseline_condition="no_backoff", candidate_condition="rate_limit|aware<script>"
    )
    text = render_markdown(result)
    assert text == render_markdown(result)
    assert "rate_limit\\|aware&lt;script&gt;" in text
    assert "50 pp" in text
    assert "N/A" in text
    assert "not matched-request effects" in text


def _make_lab07_run(tmp_path, malformed=False):
    run = tmp_path / "results" / "lab07_rate_limiting" / "pilot"
    run.mkdir(parents=True)
    (run / "experiment_manifest.json").write_text(
        json.dumps({"normalized_contract_version": 1}), encoding="utf-8"
    )
    records = [
        {
            "scenario_id": "transient_failures",
            "strategy_id": condition,
            "client_id": i,
            "succeeded": success,
            "total_attempts": 2,
            "retry_count": retries,
            "rate_limit_response_count": 1,
            "total_scheduled_wait_seconds": wait,
        }
        for condition, success, retries, wait, i in [
            ("no_backoff", True, 1, 0.0, 0),
            ("no_backoff", False, 1, 0.0, 1),
            ("rate_limit_aware", True, 1, 1.0, 0),
            ("rate_limit_aware", True, 0, 2.0, 1),
        ]
    ]
    if malformed:
        records[0]["total_attempts"] = "INVALID"
    (run / "raw_measurements.json").write_text(json.dumps(records), encoding="utf-8")
    return run


def test_cli_saved_run_reports_json_and_md_without_mutating_inputs(tmp_path, capsys):
    run = _make_lab07_run(tmp_path)
    measurements = run / "raw_measurements.json"
    original = measurements.read_bytes()
    args = [
        "--results-dir", str(tmp_path / "results"), "--lab", "lab07",
        "--run-id", "pilot", "--scenario", "transient_failures",
        "--baseline-condition", "no_backoff",
        "--candidate-condition", "rate_limit_aware",
    ]
    output = tmp_path / "results" / "comparisons" / "report.json"
    assert main([*args, "--format", "json", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["success_rate"]["percentage_point_delta"] == 50
    assert measurements.read_bytes() == original
    copied = output.read_bytes()
    assert main([*args, "--format", "json", "--output", str(output)]) == 0
    assert output.read_bytes() == copied
    assert main(args) == 0
    assert "Condition Comparison" in capsys.readouterr().out
    with pytest.raises(SystemExit) as err:
        main([*args, "--output", str(run / "experiment_manifest.json")])
    assert err.value.code == 2


def test_cli_blocks_corrupt_source_even_if_normalizer_silences_bad_values(tmp_path):
    _make_lab07_run(tmp_path, malformed=True)
    args = [
        "--results-dir", str(tmp_path / "results"), "--lab", "lab07",
        "--run-id", "pilot", "--scenario", "transient_failures",
        "--baseline-condition", "no_backoff",
        "--candidate-condition", "rate_limit_aware",
    ]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_cli_refuses_missing_and_ambiguous_scenarios(tmp_path):
    _make_lab07_run(tmp_path)
    args = [
        "--results-dir", str(tmp_path / "results"), "--lab", "lab07",
        "--run-id", "pilot", "--baseline-condition", "no_backoff",
        "--candidate-condition", "rate_limit_aware",
    ]
    assert main(args) == 0  # one scenario, scenario flag optional
    with pytest.raises(SystemExit):
        main([*args, "--scenario", "not-existing"])
