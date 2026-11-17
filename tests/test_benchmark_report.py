"""Offline benchmark report contracts: safety, separation and reproducibility."""

from __future__ import annotations

import json

import pytest

from lab_core.benchmark_report import (
    _metric_stats,
    build_benchmark,
    main,
    render_markdown,
)


def row(**changes):
    baseline = {
        "lab": "lab01",
        "experiment": "lab01_factorial",
        "run_id": "matched-run",
        "scenario": "Unspecified",
        "source": "raw_measurements.json",
        "model": "model-a",
        "condition": "temperature-0",
        "cost_basis": "Estimated",
        "simulated": False,
        "succeeded": True,
        "answer_correct": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "latency_ms": None,
        "retry_count": None,
        "rate_limit_response_count": None,
        "total_scheduled_wait_seconds": None,
    }
    return {**baseline, **changes}


def test_observed_denominators_and_nearest_rank_percentiles():
    report = build_benchmark([
        row(total_tokens=10, estimated_cost_usd=0.1, latency_ms=50, answer_correct=True),
        row(total_tokens=None, estimated_cost_usd=None, latency_ms=70, answer_correct=None),
        row(total_tokens=30, estimated_cost_usd=0.3, latency_ms=90, answer_correct=False),
    ], lab="lab01", run_id="matched-run")
    assert report["measurement_count"] == 3
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["metrics"]["total_tokens"]["mean"] == 20
    assert group["metrics"]["total_tokens"]["observed"] == 2
    assert group["metrics"]["estimated_cost_usd"]["observed"] == 2
    assert group["metrics"]["latency_ms"]["p50"] == 70
    assert group["metrics"]["latency_ms"]["p95"] == 90
    assert group["answer_accuracy"]["evaluated"] == 2
    assert group["answer_accuracy"]["rate"] == 0.5
    md = render_markdown(report)
    assert "20.00 (n=2)" in md
    assert "50.0% (n=2)" in md
    assert "50.00 / 90.00 (n=3)" not in md


def test_run_isolation_and_source_and_cost_basis_separation():
    report = build_benchmark([
        row(total_tokens=12),
        row(source="raw_measurements.csv", total_tokens=200),
        row(cost_basis="Batch discounted estimate", total_tokens=17),
        row(run_id="different-run", total_tokens=10000),
        row(lab="lab02", total_tokens=10000),
    ], lab="lab01", run_id="matched-run")
    assert report["measurement_count"] == 3
    assert report["group_count"] == 3
    assert sorted(g["metrics"]["total_tokens"]["mean"] for g in report["groups"]) == [12, 17, 200]


def test_simulation_requires_matched_scenario_and_never_invents_billing():
    records = [
        row(lab="lab07", run_id="sim", scenario="transient_failures",
            model="Simulation (no LLM)", condition="no_backoff", simulated=True,
            succeeded=True, retry_count=2, rate_limit_response_count=2,
            total_scheduled_wait_seconds=0),
        row(lab="lab07", run_id="sim", scenario="transient_failures",
            model="Simulation (no LLM)", condition="no_backoff", simulated=True,
            succeeded=False, retry_count=3, rate_limit_response_count=4,
            total_scheduled_wait_seconds=0),
        row(lab="lab07", run_id="sim", scenario="server_recovery_window",
            model="Simulation (no LLM)", simulated=True, succeeded=True),
    ]
    with pytest.raises(ValueError, match="multiple scenarios"):
        build_benchmark(records, lab="lab07", run_id="sim")

    report = build_benchmark(records, lab="lab07", run_id="sim", scenario="transient_failures")
    assert report["measurement_count"] == 2
    m = report["groups"][0]["metrics"]
    assert m["retry_count"]["mean"] == 2.5
    assert m["rate_limit_response_count"]["mean"] == 3
    assert m["total_scheduled_wait_seconds"]["mean"] == 0
    assert m["estimated_cost_usd"]["mean"] is None
    rendered = render_markdown(report)
    assert "Offline retry simulation" in rendered
    assert "synthetic HTTP/client" in rendered
    assert "50.0% (n=2)" in rendered


def test_missing_metrics_and_invalid_numeric_values_remain_missing():
    m = _metric_stats([])
    assert m == {"observed": 0, "mean": None, "p50": None, "p95": None}
    report = build_benchmark([
        row(total_tokens=None, estimated_cost_usd=None, latency_ms=None),
        row(total_tokens=float("nan"), estimated_cost_usd=float("inf"), latency_ms=-1),
    ], lab="lab01", run_id="matched-run")
    summary = report["groups"][0]["metrics"]
    for key in ("total_tokens", "estimated_cost_usd", "latency_ms"):
        assert summary[key]["observed"] == 0
        assert summary[key]["mean"] is None
    assert "N/A (n=0)" in render_markdown(report)


def test_markdown_stable_and_safe_for_untrusted_local_identifiers():
    records = [
        row(condition="a|b", run_id="local<run>", model="model\nnext", total_tokens=5,
            secret_prompt="DO NOT LEAK THIS CONTENT")
    ]
    report = build_benchmark(records, lab="lab01", run_id="local<run>")
    content = render_markdown(report)
    assert content == render_markdown(report)
    assert "DO NOT LEAK THIS CONTENT" not in content
    assert "a\\|b" in content
    assert "local&lt;run&gt;" in content
    assert "model next" in content


def test_cli_reads_allowlisted_file_and_writes_deterministic_json(tmp_path, capsys):
    results = tmp_path / "results"
    run_dir = results / "lab07_rate_limiting" / "pilot-run"
    run_dir.mkdir(parents=True)
    (run_dir / "raw_measurements.json").write_text(json.dumps([
        {
            "scenario_id": "transient_failures",
            "strategy_id": "no_backoff",
            "succeeded": True,
            "retry_count": 2,
            "rate_limit_response_count": 2,
            "total_scheduled_wait_seconds": 0,
        },
    ]), encoding="utf-8")
    output = tmp_path / "reports" / "offline.json"
    args = [
        "--results-dir", str(results), "--lab", "lab07",
        "--run-id", "pilot-run", "--format", "json", "--output", str(output),
    ]
    assert main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measurement_count"] == 1
    assert report["groups"][0]["success"]["rate"] == 1
    first = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == first
    assert "Saved report" in capsys.readouterr().out


def test_cli_absent_run_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["--results-dir", str(tmp_path), "--lab", "lab07", "--run-id", "not-here"])
    assert caught.value.code == 2
