from __future__ import annotations

from labs.lab09_batch_processing.analysis import compute_and_persist_summary
from labs.lab09_batch_processing.models import (
    RECON_COMPLETED,
    RECON_MISSING,
    STATUS_SUCCESS,
    BatchJobMetadata,
    BatchRequestMeasurement,
    RealtimeMeasurement,
)
from labs.lab09_batch_processing.persistence import (
    write_batch_job,
    write_batch_measurements,
    write_manifest,
    write_realtime_measurements,
)


def _seq_measurement(task_id, correct):
    return RealtimeMeasurement(
        experiment_run_id="run1", execution_index=1, task_id=task_id, category="c", condition="sequential",
        max_concurrent=1, timestamp="t", model="m", reasoning_effort="low", max_completion_tokens=100,
        request_status=STATUS_SUCCESS, prompt_tokens=10, completion_tokens=2, answer_correct=correct,
        evaluation_status="CORRECT" if correct else "INCORRECT",
        estimated_standard_cost_usd=0.001,
    )


def test_compute_and_persist_summary_realtime_only(tmp_path, config_with_pricing):
    run_dir = tmp_path / "run1"
    write_manifest(run_dir, {"experiment_run_id": "run1", "profile": "pilot"})
    write_realtime_measurements(run_dir, "sequential", [_seq_measurement("task_001", True), _seq_measurement("task_002", False)])

    lab09_config = config_with_pricing.lab09
    pricing = config_with_pricing.pricing.get("openai/gpt-oss-20b")
    summary = compute_and_persist_summary(run_dir, lab09_config, "openai/gpt-oss-20b", pricing)

    assert summary["sequential_summary"]["successful_requests"] == 2
    assert summary["sequential_summary"]["correct_count"] == 1
    assert summary["batch_job"] is None
    assert summary["pricing_comparison"] is None
    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "analysis_report.md").exists()
    report_text = (run_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "BATCH_PENDING" in report_text


def test_compute_and_persist_summary_includes_batch_once_evaluated(tmp_path, config_with_pricing):
    run_dir = tmp_path / "run1"
    write_manifest(run_dir, {"experiment_run_id": "run1", "profile": "pilot"})
    write_batch_job(run_dir, BatchJobMetadata(experiment_run_id="run1", workload_version="v1", model="openai/gpt-oss-20b", local_lifecycle_state="EVALUATED", batch_id="b1"))
    write_batch_measurements(
        run_dir,
        [
            BatchRequestMeasurement(
                experiment_run_id="run1", task_id="task_001", category="c", custom_id="task_001",
                reconciliation_status=RECON_COMPLETED, prompt_tokens=10, completion_tokens=2,
                estimated_standard_cost_usd=0.002, estimated_batch_cost_usd=0.001, answer_correct=True,
            ),
            BatchRequestMeasurement(
                experiment_run_id="run1", task_id="task_002", category="c", custom_id="task_002",
                reconciliation_status=RECON_MISSING,
            ),
        ],
    )

    lab09_config = config_with_pricing.lab09
    pricing = config_with_pricing.pricing.get("openai/gpt-oss-20b")
    summary = compute_and_persist_summary(run_dir, lab09_config, "openai/gpt-oss-20b", pricing)

    assert summary["pricing_comparison"]["estimated_standard_cost_usd"] == 0.002
    assert summary["pricing_comparison"]["estimated_batch_cost_usd"] == 0.001
    assert summary["batch_counts"]["completed_requests"] == 1
    assert summary["batch_counts"]["missing_requests"] == 1
    quality_batch = next(q for q in summary["quality_summaries"] if q["condition"] == "batch")
    assert quality_batch["correct_count"] == 1
    assert quality_batch["total_evaluated"] == 1
    assert (run_dir / "cost_comparison.csv").exists()
    assert (run_dir / "quality_comparison.csv").exists()


def test_compute_and_persist_summary_never_crashes_on_empty_run(tmp_path, config_with_pricing):
    run_dir = tmp_path / "run1"
    write_manifest(run_dir, {"experiment_run_id": "run1", "profile": "pilot"})
    lab09_config = config_with_pricing.lab09
    summary = compute_and_persist_summary(run_dir, lab09_config, "openai/gpt-oss-20b", None)
    assert summary["sequential_summary"] is None
    assert summary["concurrent_summary"] is None
    assert summary["quality_summaries"] == []
