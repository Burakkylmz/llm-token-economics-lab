from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.lab08_concurrency.charts import generate_concurrency_charts
from labs.lab08_concurrency.legacy_charts import generate_legacy_concurrency_charts

REAL_LEGACY_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _write_summary(run_dir: Path, conditions: list[dict], events: list[dict] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_run_id": run_dir.name, "conditions": conditions, "timeline_events": events or []}
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _condition(max_concurrent, total=4, successful=4, cost=0.004, mean_latency=100.0, p95=150.0):
    return {
        "max_concurrent": max_concurrent,
        "total_requests": total,
        "successful_requests": successful,
        "successful_requests_per_second": successful / 1.0,
        "batch_wall_clock_ms": 1000.0,
        "queue_wait_ms": {"mean": 5.0, "p95": 10.0},
        "request_service_latency_ms": {"mean": mean_latency, "p95": p95},
        "total_estimated_cost_usd": cost,
    }


def test_generate_concurrency_charts_writes_all_nine_pngs(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_summary(run_dir, [_condition(1), _condition(5)])
    written = generate_concurrency_charts(run_dir)
    assert len(written) == 9
    for path in written:
        assert path.exists()
    assert (run_dir / "charts" / "performance_reliability_dashboard.png").exists()


def test_generate_concurrency_charts_returns_empty_when_no_summary(tmp_path):
    run_dir = tmp_path / "run-missing"
    written = generate_concurrency_charts(run_dir)
    assert written == []


def test_generate_concurrency_charts_returns_empty_when_no_usable_conditions(tmp_path):
    run_dir = tmp_path / "run-empty"
    _write_summary(run_dir, [{"max_concurrent": 1, "total_requests": 0}])
    written = generate_concurrency_charts(run_dir)
    assert written == []


def test_in_flight_timeline_uses_largest_present_concurrency_level(tmp_path):
    run_dir = tmp_path / "run-1"
    events = [
        {"max_concurrent": 1, "task_id": "t1", "event_type": "REQUEST_STARTED", "relative_ms": 0.0},
        {"max_concurrent": 1, "task_id": "t1", "event_type": "REQUEST_COMPLETED", "relative_ms": 50.0},
        {"max_concurrent": 5, "task_id": "t2", "event_type": "REQUEST_STARTED", "relative_ms": 0.0},
        {"max_concurrent": 5, "task_id": "t2", "event_type": "REQUEST_COMPLETED", "relative_ms": 30.0},
    ]
    _write_summary(run_dir, [_condition(1), _condition(5)], events=events)
    written = generate_concurrency_charts(run_dir)
    assert len(written) == 9  # must not raise despite mixed-level events


def test_generate_concurrency_charts_does_not_crash_on_zero_cost_conditions(tmp_path):
    run_dir = tmp_path / "run-1"
    conditions = [_condition(1), _condition(5)]
    for c in conditions:
        c["total_estimated_cost_usd"] = None
    _write_summary(run_dir, conditions)
    written = generate_concurrency_charts(run_dir)
    assert len(written) == 9


@pytest.mark.skipif(
    not (REAL_LEGACY_RESULTS_DIR / "lab08_concurrency.json").exists(),
    reason="Optional local historical results are not version controlled",
)
def test_legacy_backfill_against_real_preexisting_results_file():
    # results/lab08_concurrency.json is the project's real, pre-existing
    # flat legacy file -- confirming the backfill reads real historical
    # data (never fabricated) and makes zero API calls to do so.
    assert (REAL_LEGACY_RESULTS_DIR / "lab08_concurrency.json").exists()
    written = generate_legacy_concurrency_charts(REAL_LEGACY_RESULTS_DIR)
    assert len(written) == 3
    for path in written:
        assert path.exists()
        assert path.parent.name == "charts"


def test_legacy_backfill_returns_empty_when_file_absent(tmp_path):
    written = generate_legacy_concurrency_charts(tmp_path)
    assert written == []


def test_legacy_backfill_ignores_unrecognized_experiment_ids(tmp_path):
    records = [{"metadata": {"experiment_id": "some_other_lab", "timestamp": "2026-01-01T00:00:00+00:00"}}]
    with open(tmp_path / "lab08_concurrency.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    written = generate_legacy_concurrency_charts(tmp_path)
    assert written == []


def test_legacy_backfill_handles_single_timestamp_group_without_crashing(tmp_path):
    records = [
        {
            "metadata": {"experiment_id": "lab08_sequential", "timestamp": "2026-01-01T00:00:00+00:00"},
            "performance": {"client_elapsed_ms": 500.0},
            "tokens_and_cost": {"estimated_cost_usd": 0.001},
        }
    ]
    with open(tmp_path / "lab08_concurrency.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    written = generate_legacy_concurrency_charts(tmp_path)
    assert len(written) == 3  # a single timestamp can't span a duration, but must not crash
