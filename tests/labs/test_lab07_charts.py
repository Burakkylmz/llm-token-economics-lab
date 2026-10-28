from __future__ import annotations

import json

from labs.lab07_rate_limiting.charts import generate_retry_charts
from labs.lab07_rate_limiting.legacy_charts import generate_legacy_retry_charts

EXPECTED_CHART_FILES = {
    "retry_timeline.png",
    "exponential_backoff_schedule.png",
    "backoff_vs_jitter_distribution.png",
    "success_rate_by_strategy.png",
    "total_429_responses.png",
    "virtual_recovery_time.png",
    "retry_collision_analysis.png",
    "retry_after_compliance.png",
    "resilience_tradeoff_dashboard.png",
}


def _summary(scenario_id, strategy_id, success_rate=1.0, total_attempts=3, total_429=2, completion=3.0, collisions=0, total_clients=1, successful_clients=1):
    return {
        "scenario_id": scenario_id, "strategy_id": strategy_id, "total_clients": total_clients,
        "successful_clients": successful_clients, "failed_clients": total_clients - successful_clients,
        "success_rate": success_rate, "total_attempts": total_attempts, "total_retry_count": total_attempts - 1,
        "total_429_responses": total_429, "mean_attempts_per_client": float(total_attempts),
        "total_scheduled_wait_seconds": 3.0, "mean_wait_seconds": 3.0, "max_wait_seconds": 3.0,
        "mean_recovery_time_seconds": completion, "completion_time_seconds": completion,
        "retry_collision_count": collisions,
    }


def _attempt(scenario_id, strategy_id, client_id=0, attempt_number=1, virtual_time=0.0, http_status=200, delay_reason="none", scheduled_delay=None):
    return {
        "scenario_id": scenario_id, "strategy_id": strategy_id, "simulation_run_id": "run", "client_id": client_id,
        "attempt_number": attempt_number, "retry_number": attempt_number - 1, "virtual_timestamp_seconds": virtual_time,
        "http_status": http_status, "retry_after_header": None, "scheduled_delay_seconds": scheduled_delay,
        "delay_reason": delay_reason, "final_outcome": "SUCCEEDED" if http_status == 200 else None, "collision": False,
    }


def _write_summary(run_dir, summaries=None, records=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_run_id": run_dir.name,
        "strategy_summaries": summaries or [],
        "attempt_records": records or [],
    }
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestGenerateRetryCharts:
    def test_full_run_produces_every_chart(self, tmp_path):
        run_dir = tmp_path / "lab07_rate_limiting" / "run-1"
        summaries = [
            _summary("transient_failures", "no_backoff"),
            _summary("transient_failures", "exponential_backoff"),
            _summary("transient_failures", "exponential_backoff_with_jitter"),
            _summary("transient_failures", "rate_limit_aware"),
            _summary("server_recovery_window", "no_backoff", success_rate=0.0, completion=None),
            _summary("shared_contention", "no_backoff", collisions=3, total_clients=8, successful_clients=5),
        ]
        records = [
            _attempt("transient_failures", "no_backoff", http_status=429, delay_reason="no_backoff", scheduled_delay=0.0),
            _attempt("transient_failures", "no_backoff", attempt_number=2, http_status=200),
            _attempt("transient_failures", "exponential_backoff_with_jitter", http_status=429, scheduled_delay=0.7, delay_reason="exponential_backoff_jitter"),
            _attempt("retry_after_variations", "rate_limit_aware", attempt_number=1, http_status=429, delay_reason="retry_after_header"),
        ]
        _write_summary(run_dir, summaries, records)

        written = generate_retry_charts(run_dir)
        names = {p.name for p in written}
        assert names == EXPECTED_CHART_FILES
        for p in written:
            assert p.exists()

    def test_missing_summary_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab07_rate_limiting" / "run-empty"
        run_dir.mkdir(parents=True)
        assert generate_retry_charts(run_dir) == []

    def test_no_strategy_summaries_returns_empty(self, tmp_path):
        run_dir = tmp_path / "lab07_rate_limiting" / "run-zero"
        _write_summary(run_dir, [], [])
        assert generate_retry_charts(run_dir) == []

    def test_missing_retry_after_variations_renders_honest_no_data_panel(self, tmp_path):
        run_dir = tmp_path / "lab07_rate_limiting" / "run-no-variants"
        summaries = [_summary("transient_failures", "no_backoff")]
        _write_summary(run_dir, summaries, [])
        written = generate_retry_charts(run_dir)
        assert len(written) == len(EXPECTED_CHART_FILES)

    def test_never_modifies_input_summary_file(self, tmp_path):
        run_dir = tmp_path / "lab07_rate_limiting" / "run-readonly"
        summaries = [_summary("transient_failures", "no_backoff")]
        _write_summary(run_dir, summaries, [])
        before = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        generate_retry_charts(run_dir)
        after = (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
        assert before == after


class TestLegacyBackfill:
    def _legacy_record(self, strategy, succeeded, total_wait_seconds, retry_count):
        return {
            "metadata": {
                "experiment_id": f"lab07_{strategy}",
                "model_parameters": {"fail_count": 2, "succeeded": succeeded, "total_wait_seconds": total_wait_seconds},
                "retry_count": retry_count,
            },
        }

    def test_legacy_file_produces_three_charts(self, tmp_path):
        records = [
            self._legacy_record("no_backoff", True, 0.0, 2),
            self._legacy_record("exponential_backoff", True, 3.0, 2),
        ]
        (tmp_path / "lab07_rate_limiting.json").write_text(json.dumps(records), encoding="utf-8")
        written = generate_legacy_retry_charts(tmp_path)
        assert len(written) == 3

    def test_missing_legacy_file_returns_empty(self, tmp_path):
        assert generate_legacy_retry_charts(tmp_path) == []

    def test_empty_legacy_file_returns_empty(self, tmp_path):
        (tmp_path / "lab07_rate_limiting.json").write_text("[]", encoding="utf-8")
        assert generate_legacy_retry_charts(tmp_path) == []

    def test_legacy_backfill_never_modifies_source_json(self, tmp_path):
        records = [self._legacy_record("no_backoff", True, 0.0, 2)]
        path = tmp_path / "lab07_rate_limiting.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        generate_legacy_retry_charts(tmp_path)
        assert path.read_text(encoding="utf-8") == before
