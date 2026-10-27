"""Offline tests for Lab01's chart generation: labs.lab01_token_anatomy.
effort_charts, .factorial_charts, and the `lab01 backfill` CLI.

No real API calls anywhere -- every test builds synthetic measurements
in-process (via the real model/stats/persistence code, so the pipeline
from measurement -> summary JSON -> chart PNG is exercised end to end) and
asserts against files on disk in tmp_path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from labs.lab01_token_anatomy.effort_models import EffortCondition, EffortMeasurement
from labs.lab01_token_anatomy.effort_stats import summarize_all_effort_conditions
from labs.lab01_token_anatomy.effort_persistence import write_effort_summary_json
from labs.lab01_token_anatomy.effort_charts import (
    _conditions_by_prompt_and_effort,
    _usable_efforts,
    generate_effort_charts,
)

from labs.lab01_token_anatomy.factorial_models import FactorialCondition, FactorialMeasurement
from labs.lab01_token_anatomy.factorial_stats import summarize_all_conditions
from labs.lab01_token_anatomy.factorial_persistence import write_factorial_summary_json
from labs.lab01_token_anatomy.factorial_charts import generate_factorial_charts


def _file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_effort_measurement(condition_id, prompt_type, effort, repetition, completion, reasoning, succeeded=True):
    non_reasoning = completion - reasoning if succeeded else None
    return EffortMeasurement(
        experiment_run_id="effort-test",
        condition_id=condition_id,
        execution_index=repetition,
        repetition_index=repetition,
        timestamp="2026-10-27T00:00:00+00:00",
        prompt_type=prompt_type,
        model="openai/gpt-oss-20b",
        reasoning_effort=effort,
        max_completion_tokens=4096,
        api_seed=None,
        succeeded=succeeded,
        attempts_used=1,
        error=None if succeeded else "simulated failure",
        prompt_tokens=50 if succeeded else None,
        completion_tokens=completion if succeeded else None,
        reasoning_tokens=reasoning if succeeded else None,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=(50 + completion) if succeeded else None,
        finish_reason="stop" if succeeded else None,
        total_latency_ms=100.0 + completion if succeeded else None,
        estimated_cost_usd=0.00001 * completion if succeeded else None,
    )


class TestEffortCharts:
    def test_full_pipeline_single_prompt_writes_six_charts(self, tmp_path):
        conditions = [EffortCondition(f"short_prompt__{e}", "short_prompt", e) for e in ("low", "medium", "high")]
        measurements = []
        base = {"low": 20, "medium": 40, "high": 70}
        for cond in conditions:
            for rep in (1, 2):
                completion = base[cond.reasoning_effort] + rep
                reasoning = completion // 2
                measurements.append(
                    _make_effort_measurement(cond.condition_id, cond.prompt_type, cond.reasoning_effort, rep, completion, reasoning)
                )
        summaries = summarize_all_effort_conditions(conditions, {c.condition_id: 2 for c in conditions}, measurements)
        run_dir = tmp_path / "effort-test"
        write_effort_summary_json(
            run_dir,
            {
                "experiment_run_id": "effort-test",
                "planned_calls": 6,
                "successful_calls": 6,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.001,
                "conditions": [asdict(s) for s in summaries],
            },
        )
        summary_path = run_dir / "effort_summary.json"
        before_hash = _file_hash(summary_path)

        written = generate_effort_charts(run_dir)

        assert len(written) == 6
        for path in written:
            assert path.exists()
            assert path.stat().st_size > 0
            assert path.parent.name == "charts"
        # Reading existing results must never modify them.
        assert _file_hash(summary_path) == before_hash

    def test_two_repetitions_has_no_single_measurement_caveat_blocker(self, tmp_path):
        # With >=2 repetitions everywhere, sample_size_caption should be None --
        # verified at the unit level (chart PNGs can't be asserted on by text).
        from lab_core.charts import sample_size_caption

        assert sample_size_caption([2, 2, 2]) is None
        assert sample_size_caption([1, 2, 2]) is not None
        assert sample_size_caption([]) is None

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "effort-missing"
        run_dir.mkdir()
        assert generate_effort_charts(run_dir) == []

    def test_all_zero_successful_calls_returns_empty(self, tmp_path):
        conditions = [EffortCondition("short_prompt__low", "short_prompt", "low")]
        measurements = [_make_effort_measurement("short_prompt__low", "short_prompt", "low", 1, None, None, succeeded=False)]
        summaries = summarize_all_effort_conditions(conditions, {"short_prompt__low": 1}, measurements)
        run_dir = tmp_path / "effort-allfail"
        write_effort_summary_json(
            run_dir,
            {
                "experiment_run_id": "effort-allfail",
                "planned_calls": 1,
                "successful_calls": 0,
                "failed_calls": 1,
                "stopped_reason": None,
                "total_estimated_cost_usd": None,
                "conditions": [asdict(s) for s in summaries],
            },
        )
        assert generate_effort_charts(run_dir) == []

    def test_multiple_prompt_types_are_faceted_not_averaged(self):
        # short_prompt and long_prompt get very different token counts;
        # faceting means each prompt_type's own conditions are grouped
        # separately rather than collapsed into one combined mean.
        conditions = [
            {"condition_id": "short_prompt__low", "prompt_type": "short_prompt", "reasoning_effort": "low", "successful_calls": 1},
            {"condition_id": "long_prompt__low", "prompt_type": "long_prompt", "reasoning_effort": "low", "successful_calls": 1},
        ]
        by_prompt = _conditions_by_prompt_and_effort(conditions)
        assert set(by_prompt.keys()) == {"short_prompt", "long_prompt"}
        assert by_prompt["short_prompt"]["low"]["condition_id"] == "short_prompt__low"
        assert by_prompt["long_prompt"]["low"]["condition_id"] == "long_prompt__low"

    def test_zero_successful_calls_excluded_from_usable_efforts(self):
        cond_by_effort = {
            "low": {"successful_calls": 1, "completion_tokens": {"mean": 20.0}},
            "medium": {"successful_calls": 0, "completion_tokens": {"mean": None}},
        }
        usable = _usable_efforts(cond_by_effort, ["low", "medium"], "completion_tokens")
        assert usable == ["low"]

    def test_missing_effort_entirely_is_not_fabricated(self):
        # "high" was never planned/present at all for this prompt_type.
        cond_by_effort = {"low": {"successful_calls": 1, "completion_tokens": {"mean": 20.0}}}
        usable = _usable_efforts(cond_by_effort, ["low", "medium", "high"], "completion_tokens")
        assert usable == ["low"]

    def test_extended_profile_two_prompts_writes_six_charts_faceted(self, tmp_path):
        conditions = []
        measurements = []
        for prompt_type, base in (("short_prompt", 20), ("long_prompt", 200)):
            for effort, bump in (("low", 0), ("medium", 20), ("high", 50)):
                cond_id = f"{prompt_type}__{effort}"
                conditions.append(EffortCondition(cond_id, prompt_type, effort))
                completion = base + bump
                measurements.append(_make_effort_measurement(cond_id, prompt_type, effort, 1, completion, completion // 3))
        summaries = summarize_all_effort_conditions(conditions, {c.condition_id: 1 for c in conditions}, measurements)
        run_dir = tmp_path / "effort-extended"
        write_effort_summary_json(
            run_dir,
            {
                "experiment_run_id": "effort-extended",
                "planned_calls": 6,
                "successful_calls": 6,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.002,
                "conditions": [asdict(s) for s in summaries],
            },
        )
        written = generate_effort_charts(run_dir)
        assert len(written) == 6


def _make_factorial_measurement(condition_id, prompt_type, effort, temp, top_p, repetition, completion, reasoning):
    return FactorialMeasurement(
        experiment_run_id="factorial-test",
        condition_id=condition_id,
        execution_index=repetition,
        repetition_index=repetition,
        timestamp="2026-10-27T00:00:00+00:00",
        prompt_type=prompt_type,
        model="openai/gpt-oss-20b",
        reasoning_effort=effort,
        temperature=temp,
        top_p=top_p,
        max_completion_tokens=4096,
        api_seed=None,
        succeeded=True,
        attempts_used=1,
        error=None,
        prompt_tokens=50,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        non_reasoning_completion_tokens=completion - reasoning,
        total_tokens=50 + completion,
        finish_reason="stop",
        total_latency_ms=120.0 + completion,
        estimated_cost_usd=0.00001 * completion,
    )


class TestFactorialCharts:
    def test_sparse_grid_only_plots_present_cells(self, tmp_path):
        # Deliberately NOT a full temperature x top_p grid -- only 2 of the
        # possible 9 (temp, top_p) cells have data for this effort level.
        from labs.lab01_token_anatomy.factorial_models import make_condition_id

        cells = [(0.5, 0.7), (1.5, 1.0)]
        conditions = [
            FactorialCondition(make_condition_id("short_prompt", "high", t, p), "short_prompt", "high", t, p)
            for t, p in cells
        ]
        measurements = [
            _make_factorial_measurement(c.condition_id, c.prompt_type, c.reasoning_effort, c.temperature, c.top_p, 1, 60, 30)
            for c in conditions
        ]
        summaries = summarize_all_conditions(conditions, {c.condition_id: 1 for c in conditions}, measurements)
        run_dir = tmp_path / "factorial-sparse"
        write_factorial_summary_json(
            run_dir,
            {
                "experiment_run_id": "factorial-sparse",
                "planned_calls": 2,
                "successful_calls": 2,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.0012,
                "conditions": [asdict(s) for s in summaries],
            },
        )
        summary_path = run_dir / "factorial_summary.json"
        before_hash = _file_hash(summary_path)

        written = generate_factorial_charts(run_dir)

        # 4 metric heatmaps + 1 interaction chart, for the single prompt_type present.
        assert len(written) == 5
        for path in written:
            assert path.exists() and path.stat().st_size > 0
        assert _file_hash(summary_path) == before_hash

    def test_field_read_is_top_p_not_top_k(self, tmp_path):
        from labs.lab01_token_anatomy.factorial_charts import _heatmap_grid

        conditions = [
            {"temperature": 0.5, "top_p": 0.9, "top_k": 999, "reasoning_tokens": {"mean": 42.0}},
        ]
        grid = _heatmap_grid(conditions, temps=[0.5], top_ps=[0.9], field="reasoning_tokens")
        assert not grid.mask[0, 0]
        assert grid[0, 0] == 42.0

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "factorial-missing"
        run_dir.mkdir()
        assert generate_factorial_charts(run_dir) == []

    def test_two_prompt_types_produce_separate_heatmap_files(self, tmp_path):
        from labs.lab01_token_anatomy.factorial_models import make_condition_id

        conditions = []
        measurements = []
        for prompt_type in ("short_prompt", "long_prompt"):
            cond = FactorialCondition(
                make_condition_id(prompt_type, "medium", 1.0, 0.9), prompt_type, "medium", 1.0, 0.9
            )
            conditions.append(cond)
            measurements.append(
                _make_factorial_measurement(cond.condition_id, prompt_type, "medium", 1.0, 0.9, 1, 50, 20)
            )
        summaries = summarize_all_conditions(conditions, {c.condition_id: 1 for c in conditions}, measurements)
        run_dir = tmp_path / "factorial-two-prompts"
        write_factorial_summary_json(
            run_dir,
            {
                "experiment_run_id": "factorial-two-prompts",
                "planned_calls": 2,
                "successful_calls": 2,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.001,
                "conditions": [asdict(s) for s in summaries],
            },
        )
        written = generate_factorial_charts(run_dir)
        names = {p.name for p in written}
        assert any("short_prompt" in n for n in names)
        assert any("long_prompt" in n for n in names)
        # 2 prompt types x (4 heatmaps + 1 interaction chart) = 10 files.
        assert len(written) == 10


class TestBackfillCli:
    def _build_app_parser(self):
        from lab_core.cli import build_parser

        return build_parser()

    def test_backfill_generates_charts_for_both_families_and_skips_dry_run_only(self, tmp_path, monkeypatch):
        results_dir = tmp_path / "results"

        # A completed effort run (has effort_summary.json).
        effort_conditions = [EffortCondition("short_prompt__low", "short_prompt", "low")]
        effort_measurements = [_make_effort_measurement("short_prompt__low", "short_prompt", "low", 1, 20, 10)]
        effort_summaries = summarize_all_effort_conditions(effort_conditions, {"short_prompt__low": 1}, effort_measurements)
        effort_run_dir = results_dir / "lab01_effort" / "effort-backfilltest"
        write_effort_summary_json(
            effort_run_dir,
            {
                "experiment_run_id": "effort-backfilltest",
                "planned_calls": 1,
                "successful_calls": 1,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.0002,
                "conditions": [asdict(s) for s in effort_summaries],
            },
        )

        # A dry-run-only effort run (only experiment_plan.csv, no summary) -- must be skipped, not crash.
        dry_run_dir = results_dir / "lab01_effort" / "effort-dryrunonly"
        dry_run_dir.mkdir(parents=True)
        (dry_run_dir / "experiment_plan.csv").write_text("execution_index\n1\n", encoding="utf-8")

        # A completed factorial run.
        from labs.lab01_token_anatomy.factorial_models import make_condition_id

        factorial_cond = FactorialCondition(make_condition_id("short_prompt", "high", 1.0, 0.9), "short_prompt", "high", 1.0, 0.9)
        factorial_measurements = [
            _make_factorial_measurement(factorial_cond.condition_id, "short_prompt", "high", 1.0, 0.9, 1, 60, 30)
        ]
        factorial_summaries = summarize_all_conditions(
            [factorial_cond], {factorial_cond.condition_id: 1}, factorial_measurements
        )
        factorial_run_dir = results_dir / "lab01_factorial" / "factorial-backfilltest"
        write_factorial_summary_json(
            factorial_run_dir,
            {
                "experiment_run_id": "factorial-backfilltest",
                "planned_calls": 1,
                "successful_calls": 1,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.0003,
                "conditions": [asdict(s) for s in factorial_summaries],
            },
        )

        parser = self._build_app_parser()
        args = parser.parse_args(["lab01", "backfill", "--results-dir", str(results_dir)])
        args.func(args)

        assert (effort_run_dir / "charts").is_dir()
        assert len(list((effort_run_dir / "charts").glob("*.png"))) == 6
        assert not (dry_run_dir / "charts").exists()
        assert (factorial_run_dir / "charts").is_dir()
        assert len(list((factorial_run_dir / "charts").glob("*.png"))) == 5

    def test_backfill_run_id_filter_targets_one_run_only(self, tmp_path):
        results_dir = tmp_path / "results"
        for run_id in ("effort-one", "effort-two"):
            conditions = [EffortCondition("short_prompt__low", "short_prompt", "low")]
            measurements = [_make_effort_measurement("short_prompt__low", "short_prompt", "low", 1, 20, 10)]
            summaries = summarize_all_effort_conditions(conditions, {"short_prompt__low": 1}, measurements)
            write_effort_summary_json(
                results_dir / "lab01_effort" / run_id,
                {
                    "experiment_run_id": run_id,
                    "planned_calls": 1,
                    "successful_calls": 1,
                    "failed_calls": 0,
                    "stopped_reason": None,
                    "total_estimated_cost_usd": 0.0001,
                    "conditions": [asdict(s) for s in summaries],
                },
            )

        parser = self._build_app_parser()
        args = parser.parse_args(
            ["lab01", "backfill", "--results-dir", str(results_dir), "--run-id", "effort-one"]
        )
        args.func(args)

        assert (results_dir / "lab01_effort" / "effort-one" / "charts").is_dir()
        assert not (results_dir / "lab01_effort" / "effort-two" / "charts").exists()

    def test_backfill_never_modifies_existing_result_files(self, tmp_path):
        results_dir = tmp_path / "results"
        conditions = [EffortCondition("short_prompt__low", "short_prompt", "low")]
        measurements = [_make_effort_measurement("short_prompt__low", "short_prompt", "low", 1, 20, 10)]
        summaries = summarize_all_effort_conditions(conditions, {"short_prompt__low": 1}, measurements)
        run_dir = results_dir / "lab01_effort" / "effort-untouched"
        payload = {
            "experiment_run_id": "effort-untouched",
            "planned_calls": 1,
            "successful_calls": 1,
            "failed_calls": 0,
            "stopped_reason": None,
            "total_estimated_cost_usd": 0.0001,
            "conditions": [asdict(s) for s in summaries],
        }
        write_effort_summary_json(run_dir, payload)
        before = (run_dir / "effort_summary.json").read_bytes()

        parser = self._build_app_parser()
        args = parser.parse_args(["lab01", "backfill", "--results-dir", str(results_dir)])
        args.func(args)

        after = (run_dir / "effort_summary.json").read_bytes()
        assert before == after

    def test_backfill_effort_only_skips_factorial(self, tmp_path):
        results_dir = tmp_path / "results"
        from labs.lab01_token_anatomy.factorial_models import make_condition_id

        factorial_cond = FactorialCondition(make_condition_id("short_prompt", "high", 1.0, 0.9), "short_prompt", "high", 1.0, 0.9)
        factorial_measurements = [
            _make_factorial_measurement(factorial_cond.condition_id, "short_prompt", "high", 1.0, 0.9, 1, 60, 30)
        ]
        factorial_summaries = summarize_all_conditions(
            [factorial_cond], {factorial_cond.condition_id: 1}, factorial_measurements
        )
        factorial_run_dir = results_dir / "lab01_factorial" / "factorial-skip"
        write_factorial_summary_json(
            factorial_run_dir,
            {
                "experiment_run_id": "factorial-skip",
                "planned_calls": 1,
                "successful_calls": 1,
                "failed_calls": 0,
                "stopped_reason": None,
                "total_estimated_cost_usd": 0.0001,
                "conditions": [asdict(s) for s in factorial_summaries],
            },
        )

        parser = self._build_app_parser()
        args = parser.parse_args(["lab01", "backfill", "--results-dir", str(results_dir), "--effort-only"])
        args.func(args)

        assert not (factorial_run_dir / "charts").exists()
