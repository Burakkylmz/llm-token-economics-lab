from __future__ import annotations

import json
from pathlib import Path

from labs.lab10_conversation_summarization.charts import generate_summarization_charts


def _condition(dataset_id, condition_id, **overrides):
    base = {
        "dataset_id": dataset_id, "condition_id": condition_id, "total_requests": 4, "successful_requests": 4,
        "evaluated_answers": 4, "correct_answers": 3, "answer_accuracy": 0.75, "invalid_format_count": 0,
        "missing_answer_count": 0, "summary_generation_prompt_tokens": 1800, "summary_generation_completion_tokens": 300,
        "summary_generation_reasoning_tokens": 100, "summary_generation_non_reasoning_completion_tokens": 200,
        "followup_total_prompt_tokens": 800, "followup_total_completion_tokens": 60, "followup_total_reasoning_tokens": 20,
        "summary_generation_cost_usd": 0.01, "followup_cost_usd": 0.001, "total_workflow_cost_usd": 0.011,
        "net_savings_usd": -0.005, "net_savings_percent": -50.0, "cost_per_correct_answer_usd": 0.0037,
        "fact_retention_rate": 0.9, "known_fact_violation_count": 0, "mean_context_reduction_percent": 80.0,
        "mean_compression_ratio": 5.0, "mean_request_prompt_tokens": 600,
    }
    base.update(overrides)
    return base


def _write_summary(run_dir: Path, conditions, cumulative=None, break_even=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_run_id": run_dir.name, "datasets": ["customer_support_001"], "conditions": conditions,
        "cumulative": cumulative or [], "break_even": break_even or [],
    }
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _full_conditions():
    return [
        _condition("customer_support_001", "full_history_baseline", summary_generation_cost_usd=None, summary_generation_prompt_tokens=None, fact_retention_rate=None, known_fact_violation_count=None, mean_context_reduction_percent=None, mean_compression_ratio=None, mean_request_prompt_tokens=4000),
        _condition("customer_support_001", "summarization_low"),
        _condition("customer_support_001", "summarization_medium"),
    ]


def test_generate_charts_writes_all_six_pngs(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_summary(run_dir, _full_conditions())
    written = generate_summarization_charts(run_dir)
    assert len(written) == 6
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_charts_returns_empty_when_no_summary_file(tmp_path):
    assert generate_summarization_charts(tmp_path / "missing") == []


def test_generate_charts_returns_empty_when_no_conditions(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_summary(run_dir, [])
    assert generate_summarization_charts(run_dir) == []


def test_cumulative_chart_handles_populated_data(tmp_path):
    run_dir = tmp_path / "run-1"
    cumulative = [
        {"dataset_id": "customer_support_001", "condition_id": "full_history_baseline", "call_index": 1, "cumulative_input_tokens": 4000},
        {"dataset_id": "customer_support_001", "condition_id": "full_history_baseline", "call_index": 2, "cumulative_input_tokens": 8000},
        {"dataset_id": "customer_support_001", "condition_id": "summarization_low", "call_index": 1, "cumulative_input_tokens": 2300},
        {"dataset_id": "customer_support_001", "condition_id": "summarization_low", "call_index": 2, "cumulative_input_tokens": 2800},
    ]
    _write_summary(run_dir, _full_conditions(), cumulative=cumulative)
    written = generate_summarization_charts(run_dir)
    assert len(written) == 6


def test_break_even_chart_handles_populated_data_with_reached_marker(tmp_path):
    run_dir = tmp_path / "run-1"
    break_even = [
        {"dataset_id": "customer_support_001", "condition_id": "summarization_low", "call_index": 1, "baseline_cumulative_cost_usd": 0.001, "summary_cumulative_cost_usd": 0.011, "break_even_call_index": 2, "break_even_status": "REACHED"},
        {"dataset_id": "customer_support_001", "condition_id": "summarization_low", "call_index": 2, "baseline_cumulative_cost_usd": 0.002, "summary_cumulative_cost_usd": 0.0012, "break_even_call_index": 2, "break_even_status": "REACHED"},
    ]
    _write_summary(run_dir, _full_conditions(), break_even=break_even)
    written = generate_summarization_charts(run_dir)
    assert len(written) == 6


def test_generate_charts_handles_missing_cost_data_without_crashing(tmp_path):
    run_dir = tmp_path / "run-1"
    conditions = _full_conditions()
    for c in conditions:
        c["total_workflow_cost_usd"] = None
    _write_summary(run_dir, conditions)
    written = generate_summarization_charts(run_dir)
    assert len(written) == 6
