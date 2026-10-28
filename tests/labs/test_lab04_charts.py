from __future__ import annotations

import json

from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab04_output_control.charts import generate_output_control_charts
from labs.lab04_output_control.legacy_charts import generate_legacy_output_control_charts

EXPECTED_CHART_FILES = {
    "completion_tokens_vs_budget.png",
    "reasoning_vs_non_reasoning_tokens.png",
    "finish_reason_distribution.png",
    "answer_completeness_vs_budget.png",
    "output_limit_vs_estimated_cost.png",
    "invocation_token_breakdown.png",
    "invocation_cost_comparison.png",
    "workflow_wall_clock_latency.png",
    "quality_constrained_optimization_dashboard.png",
}


def _stats(values) -> dict:
    return vars(compute_descriptive_stats(values))


def _condition(condition_id, max_tokens, completion_tokens, reasoning_tokens, non_reasoning, cost, accepted, completeness, correctness, truncated=0):
    successful = len(completion_tokens)
    return {
        "condition_id": condition_id,
        "max_completion_tokens": max_tokens,
        "successful_calls": successful,
        "truncated_calls": truncated,
        "truncation_rate": (truncated / successful) if successful else None,
        "completion_tokens": _stats(completion_tokens),
        "reasoning_tokens": _stats(reasoning_tokens),
        "non_reasoning_completion_tokens": _stats(non_reasoning),
        "mean_estimated_cost_usd": (sum(cost) / len(cost)) if cost else None,
        "accepted_count": accepted,
        "mean_field_completeness_ratio": completeness,
        "mean_field_correctness_ratio": correctness,
    }


def _measurement_record(execution_index, condition_id, finish_reason, succeeded=True):
    return {"execution_index": execution_index, "condition_id": condition_id, "succeeded": succeeded, "finish_reason": finish_reason}


def _strategy(api_calls, prompt, completion, reasoning, cost, latency):
    return {
        "api_call_count": api_calls,
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_reasoning_tokens": reasoning,
        "total_estimated_cost_usd": cost,
        "workflow_wall_clock_ms": latency,
        "individual_request_latencies_ms": [latency],
        "correct_answers": api_calls,
        "missing_answers": 0,
        "invalid_answers": 0,
        "all_answers_accepted": True,
    }


def _write_summary(run_dir, conditions=None, records=None, comparison=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_run_id": run_dir.name,
        "output_limit_conditions": conditions or [],
        "output_limit_measurements": records or [],
        "invocation_comparison": comparison,
    }
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestGenerateOutputControlCharts:
    def test_full_run_produces_every_chart(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-1"
        conditions = [
            _condition("restrictive", 64, [64, 64], [60, 60], [4, 4], [0.0001] * 2, 0, 0.4, 0.4, truncated=2),
            _condition("moderate", 256, [100, 110], [30, 35], [70, 75], [0.0003] * 2, 2, 1.0, 1.0),
            _condition("generous", 1024, [120, 130], [40, 45], [80, 85], [0.0006] * 2, 2, 1.0, 1.0),
        ]
        records = [
            _measurement_record(1, "restrictive", "length"),
            _measurement_record(2, "restrictive", "length"),
            _measurement_record(3, "moderate", "stop"),
            _measurement_record(4, "moderate", "stop"),
        ]
        comparison = {
            "single": _strategy(1, 120, 60, 10, 0.0001, 500.0),
            "multiple": _strategy(3, 150, 45, 15, 0.00015, 1200.0),
            "absolute_cost_difference_usd": 0.00005,
            "relative_cost_difference_percent": 50.0,
            "workflow_latency_difference_ms": 700.0,
            "quality_comparison": "BOTH_STRATEGIES_ACCEPTED",
        }
        _write_summary(run_dir, conditions, records, comparison)

        written = generate_output_control_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_CHART_FILES
        for p in written:
            assert p.exists()

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-empty"
        run_dir.mkdir(parents=True)
        assert generate_output_control_charts(run_dir) == []

    def test_zero_data_at_all_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-zero"
        _write_summary(run_dir, [], [], None)
        assert generate_output_control_charts(run_dir) == []

    def test_output_limit_only_run_still_renders_all_charts_with_no_data_invocation_panels(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-ol-only"
        conditions = [_condition("restrictive", 64, [64], [60], [4], [0.0001], 0, 0.4, 0.4, truncated=1)]
        _write_summary(run_dir, conditions, [_measurement_record(1, "restrictive", "length")], None)
        written = generate_output_control_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_invocation_only_run_still_renders_all_charts_with_no_data_output_limit_panels(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-inv-only"
        comparison = {
            "single": _strategy(1, 120, 60, 10, 0.0001, 500.0),
            "multiple": _strategy(3, 150, 45, 15, 0.00015, 1200.0),
            "absolute_cost_difference_usd": 0.00005,
            "relative_cost_difference_percent": 50.0,
            "workflow_latency_difference_ms": 700.0,
            "quality_comparison": "BOTH_STRATEGIES_ACCEPTED",
        }
        _write_summary(run_dir, [], [], comparison)
        written = generate_output_control_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_never_modifies_input_summary_file(self, tmp_path):
        run_dir = tmp_path / "lab04_output_control" / "run-readonly"
        conditions = [_condition("restrictive", 64, [64], [60], [4], [0.0001], 0, 0.4, 0.4)]
        _write_summary(run_dir, conditions, [], None)
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_output_control_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _sweep_record(self, max_tokens, completion_tokens, reasoning_tokens, finish_reason, cost):
        return {
            "metadata": {"experiment_id": "lab04_output_limit_sweep", "model_parameters": {"max_tokens": max_tokens}, "error": None},
            "tokens_and_cost": {
                "prompt_tokens": 88, "completion_tokens": completion_tokens, "reasoning_tokens": reasoning_tokens,
                "finish_reason": finish_reason, "estimated_cost_usd": cost,
            },
            "performance": {"client_elapsed_ms": 500.0},
            "derived": {"non_reasoning_completion_tokens": (completion_tokens - reasoning_tokens) if reasoning_tokens is not None else None},
        }

    def _invocation_record(self, experiment_id, prompt_tokens, completion_tokens, reasoning_tokens, cost, latency):
        return {
            "metadata": {"experiment_id": experiment_id, "error": None},
            "tokens_and_cost": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "reasoning_tokens": reasoning_tokens,
                "estimated_cost_usd": cost,
            },
            "performance": {"client_elapsed_ms": latency},
        }

    def test_legacy_sweep_data_reconstructs_truncation_from_real_finish_reason(self, tmp_path):
        records = [
            self._sweep_record(16, 16, 14, "length", 0.0000114),
            self._sweep_record(64, 64, 20, "length", 0.0000288),
            self._sweep_record(256, 180, 30, "stop", 0.0001),
        ]
        (tmp_path / "lab04_output_control.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_output_control_charts(tmp_path)
        assert len(written) == 4  # 4 output-limit charts, no invocation data present
        assert any(p.name == "finish_reason_distribution.png" for p in written)

    def test_legacy_invocation_data_reconstructs_cost_and_token_breakdown(self, tmp_path):
        records = [
            self._invocation_record("lab04_single_invocation", 116, 110, None, 0.0000417, 300.0),
            self._invocation_record("lab04_multiple_invocations", 80, 70, None, 0.00003, 150.0),
            self._invocation_record("lab04_multiple_invocations", 80, 70, None, 0.00003, 150.0),
            self._invocation_record("lab04_multiple_invocations", 80, 70, None, 0.00003, 150.0),
        ]
        (tmp_path / "lab04_output_control.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_output_control_charts(tmp_path)
        names = {p.name for p in written}
        assert "invocation_token_breakdown.png" in names
        assert "invocation_cost_comparison.png" in names

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_output_control_charts(tmp_path) == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        (tmp_path / "lab04_output_control.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_output_control_charts(tmp_path) == []

    def test_legacy_backfill_never_modifies_source_json(self, tmp_path):
        records = [self._sweep_record(16, 16, 14, "length", 0.0000114)]
        path = tmp_path / "lab04_output_control.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        generate_legacy_output_control_charts(tmp_path)
        assert path.read_text(encoding="utf-8") == before

    def test_legacy_backfill_never_fabricates_workflow_latency(self, tmp_path):
        records = [
            self._invocation_record("lab04_single_invocation", 116, 110, None, 0.0000417, 300.0),
            self._invocation_record("lab04_multiple_invocations", 80, 70, None, 0.00003, 150.0),
        ]
        (tmp_path / "lab04_output_control.json").write_text(json.dumps(records), encoding="utf-8")
        from labs.lab04_output_control.legacy_charts import _build_legacy_invocation_comparison

        comparison = _build_legacy_invocation_comparison(records)
        assert comparison["single"]["workflow_wall_clock_ms"] is None
        assert comparison["multiple"]["workflow_wall_clock_ms"] is None
