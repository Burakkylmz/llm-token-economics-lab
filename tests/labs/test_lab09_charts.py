from __future__ import annotations

import json

from labs.lab09_batch_processing.charts import generate_batch_charts


def _write_summary(run_dir, payload):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_generate_batch_charts_returns_empty_for_missing_summary(tmp_path):
    assert generate_batch_charts(tmp_path / "nope") == []


def test_generate_batch_charts_returns_empty_when_no_realtime_data(tmp_path):
    run_dir = tmp_path / "run1"
    _write_summary(run_dir, {"experiment_run_id": "run1", "sequential_summary": None, "concurrent_summary": None})
    assert generate_batch_charts(run_dir) == []


def test_generate_batch_charts_writes_all_eight_pngs_with_realtime_only(tmp_path):
    run_dir = tmp_path / "run1"
    _write_summary(run_dir, {
        "experiment_run_id": "run1",
        "sequential_summary": {
            "condition": "sequential", "max_concurrent": 1, "total_requests": 2, "successful_requests": 2,
            "batch_wall_clock_ms": 100.0, "successful_requests_per_second": 20.0,
            "request_service_latency_ms": {"mean": 50.0, "p95": 60.0}, "total_estimated_cost_usd": 0.001,
        },
        "concurrent_summary": {
            "condition": "concurrent", "max_concurrent": 5, "total_requests": 2, "successful_requests": 2,
            "batch_wall_clock_ms": 40.0, "successful_requests_per_second": 50.0,
            "request_service_latency_ms": {"mean": 55.0, "p95": 65.0}, "total_estimated_cost_usd": 0.001,
        },
        "pricing_comparison": None,
        "batch_counts": None,
        "quality_summaries": [{"condition": "sequential", "total_evaluated": 2, "correct_count": 2, "accuracy": 1.0}],
        "token_totals": {"sequential": {"prompt_tokens": 20, "completion_tokens": 4}},
    })
    written = generate_batch_charts(run_dir)
    assert len(written) == 8
    for path in written:
        assert path.exists()
    names = {p.name for p in written}
    assert "batch_processing_dashboard.png" in names
    assert "standard_vs_batch_cost.png" in names


def test_generate_batch_charts_renders_batch_pending_without_crashing(tmp_path):
    run_dir = tmp_path / "run1"
    _write_summary(run_dir, {
        "experiment_run_id": "run1",
        "sequential_summary": {
            "condition": "sequential", "max_concurrent": 1, "total_requests": 1, "successful_requests": 1,
            "batch_wall_clock_ms": 10.0, "successful_requests_per_second": 100.0,
            "request_service_latency_ms": {"mean": 10.0, "p95": 10.0}, "total_estimated_cost_usd": 0.0001,
        },
        "concurrent_summary": None,
        "pricing_comparison": None,
        "batch_counts": None,
        "quality_summaries": [],
        "token_totals": {},
    })
    written = generate_batch_charts(run_dir)
    assert len(written) == 8  # renders BATCH_PENDING markers, never crashes or fabricates
