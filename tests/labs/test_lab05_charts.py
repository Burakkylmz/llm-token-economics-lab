from __future__ import annotations

import json

from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab05_model_routing.charts import generate_routing_charts
from labs.lab05_model_routing.legacy_charts import generate_legacy_routing_charts

EXPECTED_CHART_FILES = {
    "model_selection_distribution.png",
    "accuracy_by_strategy.png",
    "cost_by_strategy.png",
    "latency_by_strategy.png",
    "cost_per_task.png",
    "quality_regression_matrix.png",
    "token_consumption_breakdown.png",
    "routing_decision_analysis.png",
    "quality_cost_tradeoff_dashboard.png",
}


def _strategy(strategy, total, correct, small_calls, large_calls, cost, latency_values, rule_distribution=None):
    return {
        "strategy": strategy,
        "total_tasks": total,
        "correct_tasks": correct,
        "incorrect_tasks": total - correct,
        "incomplete_tasks": 0,
        "invalid_format_tasks": 0,
        "truncated_tasks": 0,
        "structural_validation_only_tasks": 0,
        "accuracy": (correct / total) if total else None,
        "total_estimated_cost_usd": cost,
        "mean_cost_per_task_usd": (cost / total) if cost and total else None,
        "latency_ms": vars(compute_descriptive_stats(latency_values)),
        "total_workflow_wall_clock_ms": None,
        "small_model_calls": small_calls,
        "large_model_calls": large_calls,
        "small_model_selection_ratio": (small_calls / total) if total else None,
        "large_model_selection_ratio": (large_calls / total) if total else None,
        "routing_rule_distribution": rule_distribution or {},
    }


def _comparison(task_id, baseline_cost, routed_cost, outcome):
    return {
        "task_id": task_id, "baseline_strategy": "always_small", "routed_strategy": "rule_based_routing",
        "baseline_model": "small-m", "routed_model": "large-m", "baseline_correct": True, "routed_correct": True,
        "baseline_cost": baseline_cost, "routed_cost": routed_cost, "baseline_latency": 100.0, "routed_latency": 150.0,
        "cost_delta": (routed_cost - baseline_cost) if baseline_cost is not None and routed_cost is not None else None,
        "latency_delta": 50.0, "quality_outcome": outcome,
    }


def _record(strategy, prompt_tokens, reasoning_tokens, non_reasoning, succeeded=True):
    return {
        "strategy": strategy, "succeeded": succeeded, "prompt_tokens": prompt_tokens,
        "reasoning_tokens": reasoning_tokens, "non_reasoning_completion_tokens": non_reasoning,
    }


def _write_summary(run_dir, strategies=None, comparisons=None, records=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_run_id": run_dir.name,
        "strategies": strategies or [],
        "task_comparisons": comparisons or [],
        "measurements": records or [],
    }
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestGenerateRoutingCharts:
    def test_full_run_produces_every_chart(self, tmp_path):
        run_dir = tmp_path / "lab05_model_routing" / "run-1"
        strategies = [
            _strategy("always_small", 6, 6, 6, 0, 0.0002, [100, 110, 90, 95, 105, 100]),
            _strategy("rule_based_routing", 6, 5, 4, 2, 0.0003, [100, 150, 90, 200, 105, 100],
                      rule_distribution={"multi_step_reasoning_indicators": 1, "conditional_constraints": 1}),
        ]
        comparisons = [
            _comparison("task_001", 0.00003, 0.00003, "QUALITY_PRESERVED"),
            _comparison("task_009", 0.00003, 0.00006, "ROUTING_QUALITY_IMPROVEMENT"),
        ]
        records = [_record("always_small", 50, 5, 5), _record("rule_based_routing", 80, 20, 10)]
        _write_summary(run_dir, strategies, comparisons, records)

        written = generate_routing_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_CHART_FILES
        for p in written:
            assert p.exists()

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab05_model_routing" / "run-empty"
        run_dir.mkdir(parents=True)
        assert generate_routing_charts(run_dir) == []

    def test_zero_successful_strategies_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab05_model_routing" / "run-zero"
        _write_summary(run_dir, [_strategy("always_small", 0, 0, 0, 0, None, [])], [], [])
        assert generate_routing_charts(run_dir) == []

    def test_router_with_no_matched_rules_renders_honest_no_data_panel(self, tmp_path):
        run_dir = tmp_path / "lab05_model_routing" / "run-no-rules"
        strategies = [
            _strategy("always_small", 4, 4, 4, 0, 0.0002, [100, 110, 90, 95]),
            _strategy("rule_based_routing", 4, 4, 4, 0, 0.0002, [100, 110, 90, 95], rule_distribution={}),
        ]
        _write_summary(run_dir, strategies, [], [])
        written = generate_routing_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)  # still renders, never crashes

    def test_never_modifies_input_summary_file(self, tmp_path):
        run_dir = tmp_path / "lab05_model_routing" / "run-readonly"
        strategies = [_strategy("always_small", 2, 2, 2, 0, 0.0001, [100, 110])]
        _write_summary(run_dir, strategies, [], [])
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_routing_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _legacy_record(self, experiment_id, task_id, model, prompt_tokens, reasoning_tokens, completion_tokens, latency, cost, passed):
        return {
            "metadata": {"experiment_id": experiment_id, "model_id": model, "error": None, "model_parameters": {"task_id": task_id, "passed": passed}},
            "tokens_and_cost": {
                "prompt_tokens": prompt_tokens, "reasoning_tokens": reasoning_tokens, "completion_tokens": completion_tokens,
                "estimated_cost_usd": cost,
            },
            "performance": {"client_elapsed_ms": latency},
        }

    def test_legacy_file_produces_four_charts(self, tmp_path):
        records = [
            self._legacy_record("lab05_fixed_model_baseline", "simple_capital", "small-m", 85, 26, 36, 280.0, 0.00002, True),
            self._legacy_record("lab05_rule_based_routing", "simple_capital", "small-m", 85, 26, 36, 200.0, 0.00002, True),
            self._legacy_record("lab05_rule_based_routing", "complex_code_generation", "large-m", 90, 40, 50, 400.0, 0.00006, True),
        ]
        (tmp_path / "lab05_model_routing.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_routing_charts(tmp_path)
        assert len(written) == 4
        names = {p.name for p in written}
        assert "cost_by_strategy.png" in names
        assert "legacy_pass_rate_by_strategy.png" in names

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_routing_charts(tmp_path) == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        (tmp_path / "lab05_model_routing.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_routing_charts(tmp_path) == []

    def test_legacy_backfill_never_modifies_source_json(self, tmp_path):
        records = [self._legacy_record("lab05_fixed_model_baseline", "simple_capital", "small-m", 85, 26, 36, 280.0, 0.00002, True)]
        path = tmp_path / "lab05_model_routing.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        generate_legacy_routing_charts(tmp_path)
        assert path.read_text(encoding="utf-8") == before

    def test_legacy_backfill_never_fabricates_model_tier_counts(self, tmp_path):
        records = [self._legacy_record("lab05_rule_based_routing", "task", "large-m", 85, 26, 36, 280.0, 0.00002, True)]
        (tmp_path / "lab05_model_routing.json").write_text(json.dumps(records), encoding="utf-8")
        from labs.lab05_model_routing.legacy_charts import _build_legacy_strategy_summaries_and_records

        strategies_by_id, _ = _build_legacy_strategy_summaries_and_records(records)
        assert strategies_by_id["rule_based_routing"]["small_model_calls"] == 0
        assert strategies_by_id["rule_based_routing"]["large_model_calls"] == 0
