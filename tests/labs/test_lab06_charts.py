from __future__ import annotations

import json

from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab06_streaming.charts import generate_streaming_charts
from labs.lab06_streaming.legacy_charts import generate_legacy_streaming_charts

EXPECTED_CHART_FILES = {
    "first_event_vs_first_visible.png",
    "perceived_latency_comparison.png",
    "total_latency_comparison.png",
    "paired_perceived_latency_gain.png",
    "event_to_visible_gap.png",
    "streaming_content_arrival_timeline.png",
    "effective_token_throughput.png",
    "token_cost_comparison.png",
    "streaming_performance_dashboard.png",
}


def _stats(values) -> dict:
    return vars(compute_descriptive_stats(values))


def _condition(condition, total, successful, usage_available, cost):
    return {
        "condition": condition,
        "total_calls": total,
        "successful_calls": successful,
        "first_stream_event_ms": _stats([200.0] * successful),
        "first_visible_content_ms": _stats([500.0] * successful),
        "total_latency_ms": _stats([900.0] * successful),
        "completed_count": successful,
        "incomplete_count": 0,
        "truncated_count": 0,
        "no_visible_content_count": 0,
        "stream_interrupted_count": 0,
        "api_error_count": total - successful,
        "evaluation_unavailable_count": 0,
        "usage_available_count": usage_available,
        "usage_unavailable_count": successful - usage_available,
        "total_estimated_cost_usd": cost,
        "mean_estimated_cost_usd": (cost / successful) if cost and successful else None,
    }


def _comparison(pair_id, first_visible, s_total, n_total, gain, gain_pct, diff):
    return {
        "pair_id": pair_id, "streaming_first_visible_content_ms": first_visible, "streaming_total_latency_ms": s_total,
        "non_streaming_total_latency_ms": n_total, "perceived_latency_gain_ms": gain,
        "perceived_latency_gain_percent": gain_pct, "total_latency_difference_ms": diff, "quality_comparison": "BOTH_COMPLETE",
    }


def _record(execution_index, pair_id, condition, effective_tps=None, first_event_to_visible_gap_ms=None):
    return {
        "execution_index": execution_index, "pair_id": pair_id, "condition": condition,
        "effective_completion_tokens_per_second": effective_tps, "first_event_to_visible_gap_ms": first_event_to_visible_gap_ms,
    }


def _event(execution_index, event_index, arrival_ms, cumulative_chars):
    return {"execution_index": execution_index, "event_index": event_index, "relative_arrival_ms": arrival_ms, "cumulative_visible_character_count": cumulative_chars}


def _write_summary(run_dir, conditions=None, comparisons=None, records=None, events=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_run_id": run_dir.name,
        "conditions": conditions or [],
        "pair_comparisons": comparisons or [],
        "measurements": records or [],
        "stream_events": events or [],
    }
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestGenerateStreamingCharts:
    def test_full_run_produces_every_chart(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-1"
        conditions = [_condition("non_streaming", 5, 5, 5, 0.0005), _condition("streaming", 5, 5, 3, 0.0003)]
        comparisons = [_comparison(1, 500.0, 800.0, 1000.0, 500.0, 50.0, 200.0), _comparison(2, 600.0, 850.0, 700.0, 100.0, 14.3, -150.0)]
        records = [
            _record(1, 1, "non_streaming", effective_tps=300.0),
            _record(2, 1, "streaming", effective_tps=250.0, first_event_to_visible_gap_ms=300.0),
        ]
        events = [_event(2, 1, 10.0, 5), _event(2, 2, 50.0, 15)]
        _write_summary(run_dir, conditions, comparisons, records, events)

        written = generate_streaming_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_CHART_FILES
        for p in written:
            assert p.exists()

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-empty"
        run_dir.mkdir(parents=True)
        assert generate_streaming_charts(run_dir) == []

    def test_zero_successful_calls_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-zero"
        _write_summary(run_dir, [_condition("streaming", 0, 0, 0, None)], [], [], [])
        assert generate_streaming_charts(run_dir) == []

    def test_negative_perceived_gain_is_still_charted(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-negative"
        conditions = [_condition("non_streaming", 1, 1, 1, 0.0001), _condition("streaming", 1, 1, 0, None)]
        comparisons = [_comparison(1, 1200.0, 1300.0, 1000.0, -200.0, -20.0, -300.0)]
        _write_summary(run_dir, conditions, comparisons, [], [])
        written = generate_streaming_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_missing_stream_events_renders_honest_no_data_panel(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-no-events"
        conditions = [_condition("streaming", 1, 1, 0, None)]
        _write_summary(run_dir, conditions, [], [], [])
        written = generate_streaming_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_never_modifies_input_summary_file(self, tmp_path):
        run_dir = tmp_path / "lab06_streaming" / "run-readonly"
        conditions = [_condition("streaming", 1, 1, 0, None)]
        _write_summary(run_dir, conditions, [], [], [])
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_streaming_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _legacy_record(self, experiment_id, client_elapsed_ms, ttft_ms=None, first_visible_content_ms=None, output_tps=None):
        return {
            "metadata": {"experiment_id": experiment_id, "error": None},
            "performance": {
                "client_elapsed_ms": client_elapsed_ms, "ttft_ms": ttft_ms,
                "first_visible_content_ms": first_visible_content_ms, "output_tokens_per_second": output_tps,
            },
        }

    def test_legacy_file_produces_three_charts(self, tmp_path):
        records = [
            self._legacy_record("lab06_non_streaming", 1123.3, output_tps=326.7),
            self._legacy_record("lab06_streaming", 1205.1, ttft_ms=777.0, first_visible_content_ms=1086.6),
        ]
        (tmp_path / "lab06_streaming.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_streaming_charts(tmp_path)
        assert len(written) == 3
        names = {p.name for p in written}
        assert "perceived_latency_comparison.png" in names

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_streaming_charts(tmp_path) == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        (tmp_path / "lab06_streaming.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_streaming_charts(tmp_path) == []

    def test_legacy_backfill_never_modifies_source_json(self, tmp_path):
        records = [self._legacy_record("lab06_non_streaming", 1123.3, output_tps=326.7)]
        path = tmp_path / "lab06_streaming.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        generate_legacy_streaming_charts(tmp_path)
        assert path.read_text(encoding="utf-8") == before
