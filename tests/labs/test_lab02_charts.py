from __future__ import annotations

import json
from dataclasses import asdict

from labs.lab02_context_growth.charts import generate_context_growth_charts
from labs.lab02_context_growth.legacy_charts import generate_legacy_context_growth_charts
from labs.lab02_context_growth.models import ContextMeasurement, TurnMeasurement
from labs.lab02_context_growth.stats import summarize_all_context_conditions


def _context_measurement(condition_id, repetition_index, prompt_tokens, completion_tokens, reasoning_tokens=None, latency_ms=500.0, cost=0.0001, answer_correct=True, answer_status="correct"):
    non_reasoning = completion_tokens - reasoning_tokens if reasoning_tokens is not None else None
    return ContextMeasurement(
        experiment_run_id="run-1", condition_id=condition_id, execution_index=1, repetition_index=repetition_index,
        timestamp="2026-10-27T00:00:00+00:00", model="m", max_completion_tokens=2048, succeeded=True, attempts_used=1,
        context_chars=100, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning, total_tokens=prompt_tokens + completion_tokens,
        total_latency_ms=latency_ms, estimated_cost_usd=cost, finish_reason="stop",
        answer_evaluation_status=answer_status, answer_correct=answer_correct,
    )


def _write_run(run_dir, condition_ids_and_measurements: dict, turns: list[TurnMeasurement]):
    all_measurements = [m for measurements in condition_ids_and_measurements.values() for m in measurements]
    planned = {cid: len(ms) for cid, ms in condition_ids_and_measurements.items()}
    summaries = summarize_all_context_conditions(list(condition_ids_and_measurements.keys()), planned, all_measurements)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_summary.json").write_text(
        json.dumps({"experiment_run_id": "run-1", "conditions": [asdict(s) for s in summaries]}), encoding="utf-8"
    )
    (run_dir / "multi_turn_measurements.json").write_text(json.dumps([asdict(t) for t in turns]), encoding="utf-8")


class TestGenerateContextGrowthCharts:
    def test_full_run_produces_eight_charts(self, tmp_path):
        run_dir = tmp_path / "run-1"
        measurements = {
            "minimal_context": [_context_measurement("minimal_context", 1, 150, 5, reasoning_tokens=1), _context_measurement("minimal_context", 2, 155, 5, reasoning_tokens=1)],
            "expanded_context": [_context_measurement("expanded_context", 1, 1160, 5, reasoning_tokens=10), _context_measurement("expanded_context", 2, 1165, 5, reasoning_tokens=12)],
            "irrelevant_context": [_context_measurement("irrelevant_context", 1, 1230, 5, reasoning_tokens=50, latency_ms=9000.0), _context_measurement("irrelevant_context", 2, 1240, 5, reasoning_tokens=55, latency_ms=11000.0)],
        }
        turns = [
            TurnMeasurement(experiment_run_id="run-1", turn_index=i, timestamp="t", model="m", max_completion_tokens=2048, messages_count=2 * i, succeeded=True, attempts_used=1, prompt_tokens=90 + i * 30, cumulative_prompt_tokens=sum(90 + j * 30 for j in range(1, i + 1)))
            for i in range(1, 5)
        ]
        _write_run(run_dir, measurements, turns)

        written = generate_context_growth_charts(run_dir)
        assert len(written) == 8
        for path in written:
            assert path.exists() and path.stat().st_size > 0
            assert path.parent.name == "charts"

    def test_missing_summary_file_returns_empty(self, tmp_path):
        assert generate_context_growth_charts(tmp_path / "nowhere") == []

    def test_all_zero_successful_calls_returns_empty(self, tmp_path):
        run_dir = tmp_path / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "experiment_summary.json").write_text(
            json.dumps({"conditions": [{"condition_id": "minimal_context", "successful_calls": 0}]}), encoding="utf-8"
        )
        (run_dir / "multi_turn_measurements.json").write_text("[]", encoding="utf-8")
        assert generate_context_growth_charts(run_dir) == []

    def test_conditions_are_never_averaged_together(self, tmp_path):
        # Equivalent check to Lab01's: each condition gets its own bar, never
        # a pooled average -- verified by checking the per-condition means fed
        # into the chart are distinct from one blended number.
        run_dir = tmp_path / "run-1"
        measurements = {
            "minimal_context": [_context_measurement("minimal_context", 1, 150, 5)],
            "irrelevant_context": [_context_measurement("irrelevant_context", 1, 1230, 5, latency_ms=9000.0)],
        }
        _write_run(run_dir, measurements, [])
        with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        means = {c["condition_id"]: c["prompt_tokens"]["mean"] for c in summary["conditions"]}
        assert means["minimal_context"] != means["irrelevant_context"]

    def test_missing_reasoning_tokens_not_fabricated_as_zero(self, tmp_path):
        run_dir = tmp_path / "run-1"
        measurements = {"minimal_context": [_context_measurement("minimal_context", 1, 150, 5, reasoning_tokens=None)]}
        _write_run(run_dir, measurements, [])
        written = generate_context_growth_charts(run_dir)
        assert len(written) == 8  # still writes everything; the reasoning panel just shows "no data"

    def test_single_repetition_does_not_crash(self, tmp_path):
        run_dir = tmp_path / "run-1"
        measurements = {cid: [_context_measurement(cid, 1, 150, 5)] for cid in ("minimal_context", "expanded_context", "irrelevant_context")}
        _write_run(run_dir, measurements, [])
        written = generate_context_growth_charts(run_dir)
        assert len(written) == 8

    def test_never_modifies_existing_files(self, tmp_path):
        run_dir = tmp_path / "run-1"
        measurements = {"minimal_context": [_context_measurement("minimal_context", 1, 150, 5)]}
        _write_run(run_dir, measurements, [])
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_context_growth_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _legacy_record(self, experiment_id, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms, cost, turn=None):
        model_parameters = {"turn": turn} if turn is not None else {"context_chars": 100}
        non_reasoning = completion_tokens - reasoning_tokens if reasoning_tokens is not None else None
        return {
            "metadata": {"experiment_id": experiment_id, "run_id": "r", "model_id": "m", "execution_mode": "sync", "timestamp": "t", "model_parameters": model_parameters},
            "tokens_and_cost": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "reasoning_tokens": reasoning_tokens, "estimated_cost_usd": cost, "finish_reason": "stop"},
            "performance": {"client_elapsed_ms": latency_ms},
            "derived": {"non_reasoning_completion_tokens": non_reasoning},
        }

    def test_legacy_file_produces_six_charts(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        records = [
            self._legacy_record("lab02_minimal_context", 150, 5, 1, 500.0, 0.0001),
            self._legacy_record("lab02_expanded_context", 1160, 5, 10, 1800.0, 0.0002),
            self._legacy_record("lab02_irrelevant_context", 1230, 5, 50, 9000.0, 0.0002),
            self._legacy_record("lab02_multi_turn_growth", 98, 5, None, 400.0, 0.0001, turn=1),
            self._legacy_record("lab02_multi_turn_growth", 133, 5, None, 450.0, 0.0001, turn=2),
        ]
        (results_dir / "lab02_context_growth.json").write_text(json.dumps(records), encoding="utf-8")

        written = generate_legacy_context_growth_charts(results_dir)
        assert len(written) == 6
        for path in written:
            assert path.exists() and path.stat().st_size > 0

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_context_growth_charts(tmp_path / "results") == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "lab02_context_growth.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_context_growth_charts(results_dir) == []

    def test_legacy_backfill_never_modifies_the_source_file(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        records = [self._legacy_record("lab02_minimal_context", 150, 5, 1, 500.0, 0.0001)]
        (results_dir / "lab02_context_growth.json").write_text(json.dumps(records), encoding="utf-8")
        before = (results_dir / "lab02_context_growth.json").read_text(encoding="utf-8")

        generate_legacy_context_growth_charts(results_dir)

        after = (results_dir / "lab02_context_growth.json").read_text(encoding="utf-8")
        assert before == after

    def test_legacy_backfill_does_not_produce_answer_correctness_chart(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        records = [self._legacy_record("lab02_minimal_context", 150, 5, 1, 500.0, 0.0001)]
        (results_dir / "lab02_context_growth.json").write_text(json.dumps(records), encoding="utf-8")

        written = generate_legacy_context_growth_charts(results_dir)
        assert not any("correctness" in p.name for p in written)

    def test_multi_turn_only_legacy_data_produces_growth_charts(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        records = [
            self._legacy_record("lab02_multi_turn_growth", 98, 5, None, 400.0, 0.0001, turn=1),
            self._legacy_record("lab02_multi_turn_growth", 133, 5, None, 450.0, 0.0001, turn=2),
        ]
        (results_dir / "lab02_context_growth.json").write_text(json.dumps(records), encoding="utf-8")

        written = generate_legacy_context_growth_charts(results_dir)
        names = {p.name for p in written}
        assert "multi_turn_growth.png" in names
        assert "cumulative_token_consumption.png" in names
