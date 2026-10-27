from __future__ import annotations

import json

from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab03_prompt_caching.charts import generate_cache_charts
from labs.lab03_prompt_caching.legacy_charts import generate_legacy_cache_charts

EXPECTED_CHART_FILES = {
    "cached_vs_uncached_tokens.png",
    "cache_status_by_request.png",
    "cache_ratio_by_request.png",
    "cache_friendly_vs_unfriendly_comparison.png",
    "estimated_cost_by_condition.png",
    "estimated_cost_savings.png",
    "latency_by_condition.png",
    "cached_tokens_vs_latency.png",
    "summary_dashboard.png",
}
EXPECTED_PROBE_CHART_FILES = EXPECTED_CHART_FILES | {"probe_sequence.png"}


def _stats(values) -> dict:
    return vars(compute_descriptive_stats(values))


def _condition(condition_id, prompt_tokens, cached_tokens, latency, cost, savings, hit_rate) -> dict:
    return {
        "condition_id": condition_id,
        "successful_calls": len(prompt_tokens),
        "prompt_tokens": _stats(prompt_tokens),
        "cached_tokens": _stats(cached_tokens),
        "uncached_prompt_tokens": _stats([(p - c) if c is not None else None for p, c in zip(prompt_tokens, cached_tokens)]),
        "latency_ms": _stats(latency),
        "mean_estimated_cost_usd": sum(cost) / len(cost) if cost else None,
        "total_estimated_cache_savings_usd": sum(savings) if savings else None,
        "observed_request_cache_hit_rate": hit_rate,
        "token_cache_ratio": hit_rate,
        "correct_count": len(prompt_tokens),
        "incorrect_count": 0,
        "format_invalid_count": 0,
        "missing_response_count": 0,
    }


def _measurement_record(execution_index, condition_id, prompt_tokens, cached_tokens, latency) -> dict:
    return {
        "execution_index": execution_index,
        "condition_id": condition_id,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "total_latency_ms": latency,
        "request_cache_ratio": (cached_tokens / prompt_tokens) if prompt_tokens else None,
    }


def _write_summary(run_dir, conditions, measurements, experiment_mode=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_run_id": run_dir.name, "conditions": conditions, "measurements": measurements}
    if experiment_mode is not None:
        payload["experiment_mode"] = experiment_mode
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestGenerateCacheCharts:
    def test_full_run_produces_every_chart(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-1"
        conditions = [
            _condition("repeated_prefix", [2000, 2000, 2000], [1500, 1600, 1700], [100, 110, 120], [0.001] * 3, [0.0001] * 3, 1.0),
            _condition("cache_unfriendly", [2000, 2000, 2000], [0, 0, 0], [200, 210, 220], [0.001] * 3, [], 0.0),
        ]
        records = [
            _measurement_record(1, "repeated_prefix", 2000, 1500, 100),
            _measurement_record(2, "cache_unfriendly", 2000, 0, 200),
        ]
        _write_summary(run_dir, conditions, records)

        written = generate_cache_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_CHART_FILES
        for p in written:
            assert p.exists()

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-empty"
        run_dir.mkdir(parents=True)
        assert generate_cache_charts(run_dir) == []

    def test_zero_successful_calls_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-zero"
        conditions = [_condition("repeated_prefix", [], [], [], [], [], None)]
        _write_summary(run_dir, conditions, [])
        assert generate_cache_charts(run_dir) == []

    def test_conditions_never_averaged_together(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-distinct"
        conditions = [
            _condition("repeated_prefix", [1000], [800], [50], [0.0001], [0.00005], 1.0),
            _condition("cache_unfriendly", [5000], [0], [500], [0.0005], [], 0.0),
        ]
        _write_summary(run_dir, conditions, [])
        written = generate_cache_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_missing_cache_usage_never_rendered_as_zero(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-missing-cache"
        conditions = [_condition("repeated_prefix", [1000, 1000], [None, None], [50, 60], [0.0001, 0.0001], [], None)]
        conditions[0]["cached_tokens"] = _stats([])  # no usable cached_tokens data at all
        _write_summary(run_dir, conditions, [])
        written = generate_cache_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)  # still renders (with "no data" panels), never raises

    def test_single_repetition_does_not_crash(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-n1"
        conditions = [_condition("repeated_prefix", [1000], [800], [50], [0.0001], [0.00005], 1.0)]
        _write_summary(run_dir, conditions, [])
        written = generate_cache_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_probe_run_produces_an_extra_probe_sequence_chart(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-probe"
        conditions = [_condition("repeated_prefix", [1454, 1454, 1454], [None, None, 0], [400, 300, 350], [0.0001] * 3, [], None)]
        records = [
            {"execution_index": 1, "condition_id": "repeated_prefix", "cached_tokens": None, "cache_status": "CACHE_USAGE_UNAVAILABLE"},
            {"execution_index": 2, "condition_id": "repeated_prefix", "cached_tokens": None, "cache_status": "CACHE_USAGE_UNAVAILABLE"},
            {"execution_index": 3, "condition_id": "repeated_prefix", "cached_tokens": 0, "cache_status": "CACHE_MISS"},
        ]
        _write_summary(run_dir, conditions, records, experiment_mode="cache_probe")
        written = generate_cache_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_PROBE_CHART_FILES

    def test_non_probe_run_never_produces_a_probe_sequence_chart(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-nonprobe"
        conditions = [_condition("repeated_prefix", [1454], [0], [400], [0.0001], [], 0.0)]
        _write_summary(run_dir, conditions, [], experiment_mode="all")
        written = generate_cache_charts(run_dir)
        assert "probe_sequence.png" not in {p.name for p in written}

    def test_never_modifies_input_summary_file(self, tmp_path):
        run_dir = tmp_path / "lab03_prompt_caching" / "run-readonly"
        conditions = [_condition("repeated_prefix", [1000], [800], [50], [0.0001], [0.00005], 1.0)]
        _write_summary(run_dir, conditions, [])
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_cache_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _legacy_record(self, experiment_id, prompt_tokens, cached_tokens, latency, cost):
        return {
            "metadata": {"experiment_id": experiment_id},
            "tokens_and_cost": {"prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens, "estimated_cost_usd": cost},
            "performance": {"client_elapsed_ms": latency},
        }

    def test_legacy_file_produces_seven_charts(self, tmp_path):
        records = [
            self._legacy_record("lab03_repeated_prefix", 800, None, 400, 0.0001),
            self._legacy_record("lab03_cache_unfriendly", 790, None, 230, 0.0001),
        ]
        (tmp_path / "lab03_prompt_caching.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_cache_charts(tmp_path)
        assert len(written) == 7
        assert all("answer_correctness" not in p.name for p in written)
        assert any(p.name == "cache_status_by_request.png" for p in written)
        assert any(p.name == "cache_friendly_vs_unfriendly_comparison.png" for p in written)

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_cache_charts(tmp_path) == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        (tmp_path / "lab03_prompt_caching.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_cache_charts(tmp_path) == []

    def test_legacy_backfill_never_modifies_source_json(self, tmp_path):
        records = [self._legacy_record("lab03_repeated_prefix", 800, None, 400, 0.0001)]
        path = tmp_path / "lab03_prompt_caching.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        generate_legacy_cache_charts(tmp_path)
        assert path.read_text(encoding="utf-8") == before

    def test_legacy_backfill_never_produces_a_correctness_chart(self, tmp_path):
        records = [
            self._legacy_record("lab03_repeated_prefix", 800, 600, 400, 0.0001),
            self._legacy_record("lab03_cache_unfriendly", 790, 0, 230, 0.0001),
        ]
        (tmp_path / "lab03_prompt_caching.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_cache_charts(tmp_path)
        assert not any("correctness" in p.name or "dashboard" in p.name for p in written)
