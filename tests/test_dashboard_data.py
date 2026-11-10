"""Offline regression tests for the read-only dashboard metrics layer."""

from __future__ import annotations

import csv
import io
import json

from lab_core.dashboard_data import (
    EXPORT_COLUMNS, export_csv, filter_records, scan_results, summarize, summarize_lab07,
)


def _save_json(root, experiment, run_id, name, payload):
    directory = root / experiment / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_missing_results_is_empty(tmp_path):
    rows, warnings = scan_results(tmp_path / "missing")
    assert rows == []
    assert warnings == []


def test_lab05_and_lab10_normalization_without_imputing_unknown_values(tmp_path):
    _save_json(tmp_path, "lab05_model_routing", "run-a", "routing_measurements.json", [
        {
            "model": "misleading", "selected_model": "small",
            "strategy": "always_small", "succeeded": True,
            "prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20,
            "estimated_cost_usd": 0.002, "total_latency_ms": 120,
            "answer_correct": False,
            "final_answer_text": "SECRET MODEL RESPONSE",
        },
        {
            "selected_model": "large", "succeeded": False,
            "estimated_cost_usd": None, "answer_correct": None,
        },
    ])
    target = tmp_path / "lab10_conversation_summarization" / "run-b"
    target.mkdir(parents=True)
    with (target / "followup_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "condition_id", "request_status", "followup_prompt_tokens",
            "followup_completion_tokens", "followup_estimated_cost_usd",
            "followup_latency_ms", "answer_correct",
        ])
        writer.writeheader()
        writer.writerow({
            "model": "large", "condition_id": "summary", "request_status": "SUCCESS",
            "followup_prompt_tokens": "50", "followup_completion_tokens": "10",
            "followup_estimated_cost_usd": "0.007", "followup_latency_ms": "200",
            "answer_correct": "True",
        })
    records, warnings = scan_results(tmp_path)
    assert warnings == []
    assert len(records) == 3
    assert records[0]["model"] == "small"
    assert records[0]["answer_correct"] is False
    assert records[1]["estimated_cost_usd"] is None
    assert records[1]["answer_correct"] is None
    assert records[2]["prompt_tokens"] == 50
    assert records[2]["estimated_cost_usd"] == 0.007
    assert records[2]["succeeded"] is True
    assert "SECRET MODEL RESPONSE" not in export_csv(records)
    assert "final_answer_text" not in EXPORT_COLUMNS


def test_summary_uses_known_only_denominators_and_preserves_zero(tmp_path):
    _save_json(tmp_path, "lab05_model_routing", "sample", "routing_measurements.json", [
        {"selected_model": "small", "strategy": "router", "succeeded": True,
         "answer_correct": True, "estimated_cost_usd": 0, "total_latency_ms": 10},
        {"selected_model": "small", "strategy": "router", "succeeded": False,
         "answer_correct": None, "estimated_cost_usd": None, "total_latency_ms": None},
    ])
    records, _ = scan_results(tmp_path)
    stats = summarize(records)
    assert len(stats) == 1
    s = stats[0]
    assert s["requests"] == 2
    assert s["success_evaluated"] == 2
    assert s["success_rate"] == 0.5
    assert s["quality_evaluated"] == 1
    assert s["accuracy"] == 1.0
    assert s["cost_observed"] == 1
    assert s["estimated_cost_usd"] == 0.0
    assert s["mean_latency_ms"] == 10
    assert s["latency_observed"] == 1


def test_invalid_numeric_values_not_fabricated(tmp_path):
    _save_json(tmp_path, "lab08_concurrency", "bad", "raw_measurements.json", [
        {"model": "m", "request_status": "UNKNOWN_ERROR",
         "estimated_cost_usd": -1, "total_tokens": "NaN",
         "task_end_to_end_latency_ms": "inf"},
        {"model": "m", "request_status": "SUCCESS",
         "estimated_cost_usd": "not-a-cost", "total_tokens": -10},
    ])
    records, _ = scan_results(tmp_path)
    assert len(records) == 2
    assert records[0]["succeeded"] is False
    assert records[1]["succeeded"] is True
    for row in records:
        assert row["estimated_cost_usd"] is None
        assert row["total_tokens"] is None
        assert row["latency_ms"] is None


def test_unreadable_file_does_not_crash_or_block_other_runs(tmp_path):
    folder = tmp_path / "lab05_model_routing" / "bad"
    folder.mkdir(parents=True)
    (folder / "routing_measurements.json").write_text("{broken", encoding="utf-8")
    _save_json(tmp_path, "lab05_model_routing", "good", "routing_measurements.json", [
        {"selected_model": "m", "answer_correct": True},
    ])
    records, warnings = scan_results(tmp_path)
    assert len(records) == 1
    assert len(warnings) == 1
    assert "bad" in warnings[0]


def test_duplicate_json_csv_family_not_counted_twice(tmp_path):
    folder = _save_json(tmp_path, "lab01_effort", "run1", "raw_measurements.json", [
        {"model": "m", "estimated_cost_usd": 0.001},
    ])
    (folder / "raw_measurements.csv").write_text(
        "model,estimated_cost_usd\nm,0.001\n", encoding="utf-8",
    )
    records, warnings = scan_results(tmp_path)
    assert len(records) == 1
    assert warnings == []


def test_batch_uses_discounted_cost_not_hypothetical_standard(tmp_path):
    folder = _save_json(tmp_path, "lab09_batch_processing", "run2", "batch_measurements.json", [
        {"task_id": "t1", "estimated_batch_cost_usd": 0.15,
         "estimated_standard_cost_usd": 0.30},
    ])
    (folder / "experiment_manifest.json").write_text(json.dumps({
        "experiment_configuration": {"model": "batch-model"},
    }), encoding="utf-8")
    records, _ = scan_results(tmp_path)
    assert len(records) == 1
    assert records[0]["estimated_cost_usd"] == 0.15
    assert records[0]["cost_basis"] == "Batch discounted estimate"
    assert records[0]["model"] == "batch-model"


def test_filters_and_csv_formula_escaping(tmp_path):
    _save_json(tmp_path, "lab05_model_routing", "run3", "routing_measurements.json", [
        {"selected_model": "=HYPERLINK('evil')", "strategy": "router"},
        {"selected_model": "safe", "strategy": "always_small"},
    ])
    records, _ = scan_results(tmp_path)
    assert len(filter_records(records, models=["safe"])) == 1
    exported = list(csv.DictReader(io.StringIO(export_csv(records))))
    assert exported[0]["model"].startswith("'=")
    assert exported[1]["model"] == "safe"


def test_legacy_rows_are_metrics_only(tmp_path):
    (tmp_path / "lab01_token_anatomy.json").write_text(json.dumps([
        {"metadata": {"model": "m", "experiment_id": "lab01_short", "timestamp": "t"},
         "tokens_and_cost": {"prompt_tokens": 5, "completion_tokens": 3,
                             "estimated_cost_usd": 0.0001},
         "performance": {"client_elapsed_ms": 20},
         "text": "PRIVATE PROMPT"},
    ]), encoding="utf-8")
    records, _ = scan_results(tmp_path)
    assert len(records) == 1
    assert records[0]["prompt_tokens"] == 5
    assert records[0]["latency_ms"] == 20
    assert "PRIVATE PROMPT" not in export_csv(records)


def test_lab07_simulation_metrics_and_strategy_comparisons(tmp_path):
    _save_json(tmp_path, "lab07_rate_limiting", "demo", "raw_measurements.json", [
        {
            "scenario_id": "transient_failures", "strategy_id": "no_backoff",
            "succeeded": False, "outcome": "FAILED_MAX_ATTEMPTS",
            "total_attempts": 3, "retry_count": 2, "rate_limit_response_count": 3,
            "total_scheduled_wait_seconds": 0, "virtual_completion_time_seconds": None,
        },
        {
            "scenario_id": "transient_failures", "strategy_id": "rate_limit_aware",
            "succeeded": True, "outcome": "SUCCEEDED",
            "total_attempts": 2, "retry_count": 1, "rate_limit_response_count": 1,
            "total_scheduled_wait_seconds": 2.5, "virtual_completion_time_seconds": 3.5,
        },
        {
            "scenario_id": "transient_failures", "strategy_id": "rate_limit_aware",
            "succeeded": True, "outcome": "SUCCEEDED",
            "total_attempts": 1, "retry_count": 0, "rate_limit_response_count": 0,
            "total_scheduled_wait_seconds": 0, "virtual_completion_time_seconds": 0.5,
        },
    ])
    rows, warnings = scan_results(tmp_path)
    assert warnings == []
    assert len(rows) == 3
    assert all(r["model"] == "Simulation (no LLM)" for r in rows)
    assert all(r["simulated"] is True for r in rows)
    assert all(r["estimated_cost_usd"] is None for r in rows)
    assert all(r["latency_ms"] is None for r in rows)
    assert rows[0]["status"] == "FAILED_MAX_ATTEMPTS"
    assert rows[1]["status"] == "SUCCEEDED"
    assert len(filter_records(rows, conditions=["rate_limit_aware"])) == 2
    assert len(filter_records(rows, scenarios=["transient_failures"])) == 3

    by_strategy = {r["strategy"]: r for r in summarize_lab07(rows)}
    assert len(by_strategy) == 2
    assert by_strategy["no_backoff"]["success_rate_percent"] == 0
    assert by_strategy["no_backoff"]["successful_clients"] == 0
    assert by_strategy["rate_limit_aware"]["successful_clients"] == 2
    assert by_strategy["no_backoff"]["total_retries"] == 2
    assert by_strategy["no_backoff"]["total_429_responses"] == 3
    aware = by_strategy["rate_limit_aware"]
    assert aware["success_rate_percent"] == 100
    assert aware["total_attempts"] == 3
    assert aware["total_retries"] == 1
    assert aware["total_virtual_wait_seconds"] == 2.5
    assert aware["mean_successful_virtual_completion_seconds"] == 2.0
    assert "rate_limit_response_count" in EXPORT_COLUMNS
    assert "retry_count" in EXPORT_COLUMNS


def test_lab07_missing_metrics_stay_missing(tmp_path):
    _save_json(tmp_path, "lab07_rate_limiting", "demo", "raw_measurements.json", [
        {"scenario_id": "quota", "strategy_id": "no_backoff", "succeeded": False},
    ])
    rows, _ = scan_results(tmp_path)
    summary = summarize_lab07(rows)[0]
    assert summary["success_rate_percent"] == 0.0
    assert summary["total_attempts"] is None
    assert summary["total_retries"] is None
    assert summary["total_virtual_wait_seconds"] is None
    assert summary["total_429_responses"] is None


def test_lab07_does_not_repurpose_real_model_columns(tmp_path):
    _save_json(tmp_path, "lab05_model_routing", "real", "routing_measurements.json", [
        {"model": "a", "strategy": "always_small", "succeeded": True,
         "total_attempts": 999, "rate_limit_response_count": 800},
    ])
    rows, _ = scan_results(tmp_path)
    assert summarize_lab07(rows) == []
    assert rows[0]["simulated"] is False
    assert rows[0]["total_attempts"] is None
    assert rows[0]["rate_limit_response_count"] is None
