"""The fully offline educational demonstration (spec Part 15): some
learners cannot reach Groq's Batch API (plan restrictions, or simply not
wanting to wait out a real completion window in class). This proves the
ENTIRE post-submission pipeline -- reconciliation, evaluation, batch
pricing, comparison persistence, and chart generation -- works end to end
against the synthetic fixtures in tests/fixtures/lab09/, with no
GROQ_API_KEY and no network access at all.

Every fixture file is explicitly labeled SYNTHETIC FIXTURE -- NOT LIVE
GROQ TELEMETRY (see sample_batch_job.json's own "_label" field); this test
module never presents their contents as real provider measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

from lab_core.config import ModelPricing
from labs.lab09_batch_processing.analysis import compute_and_persist_summary
from labs.lab09_batch_processing.batch import parse_batch_output_jsonl
from labs.lab09_batch_processing.datasets import load_dataset, task_ids_for_profile, tasks_by_id
from labs.lab09_batch_processing.models import RECON_COMPLETED, RECON_MISSING, RECON_MALFORMED
from labs.lab09_batch_processing.persistence import write_batch_job, write_batch_measurements, write_manifest
from labs.lab09_batch_processing.reconciliation import reconcile_batch_results

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "lab09"

PRICING = ModelPricing(input_per_1m_usd=0.075, output_per_1m_usd=0.30, cached_input_per_1m_usd=0.0375, price_verified_date="2026-10-21")


def _pilot_tasks():
    tasks = load_dataset("v1")
    ids = task_ids_for_profile(tasks, "pilot")
    by_id = tasks_by_id(tasks)
    return [by_id[tid] for tid in ids]


def test_fixture_job_metadata_is_explicitly_labeled_synthetic():
    with open(FIXTURES_DIR / "sample_batch_job.json", encoding="utf-8") as f:
        job = json.load(f)
    assert job["_label"] == "SYNTHETIC FIXTURE -- NOT LIVE GROQ TELEMETRY"


def test_offline_reconciliation_against_sample_output_all_correct():
    planned_tasks = _pilot_tasks()
    output_rows = parse_batch_output_jsonl((FIXTURES_DIR / "sample_batch_output.jsonl").read_text(encoding="utf-8"))
    error_rows = parse_batch_output_jsonl((FIXTURES_DIR / "sample_batch_errors.jsonl").read_text(encoding="utf-8"))

    measurements, counts = reconcile_batch_results("offline-demo", planned_tasks, output_rows, error_rows, "openai/gpt-oss-20b", PRICING)

    assert counts.planned_requests == 4
    assert counts.completed_requests == 4
    assert counts.missing_requests == 0
    assert counts.correct_answers == 4
    assert all(m.reconciliation_status == RECON_COMPLETED for m in measurements)


def test_offline_reconciliation_against_partial_output_reports_missing():
    planned_tasks = _pilot_tasks()
    output_rows = parse_batch_output_jsonl((FIXTURES_DIR / "partial_batch_output.jsonl").read_text(encoding="utf-8"))

    measurements, counts = reconcile_batch_results("offline-demo", planned_tasks, output_rows, [], "openai/gpt-oss-20b", PRICING)

    assert counts.completed_requests == 2
    assert counts.missing_requests == 2
    missing_ids = {m.task_id for m in measurements if m.reconciliation_status == RECON_MISSING}
    assert missing_ids == {"task_003", "task_004"}


def test_offline_reconciliation_against_malformed_output_reports_every_anomaly():
    planned_tasks = _pilot_tasks()
    output_rows = parse_batch_output_jsonl((FIXTURES_DIR / "malformed_batch_output.jsonl").read_text(encoding="utf-8"))

    measurements, counts = reconcile_batch_results("offline-demo", planned_tasks, output_rows, [], "openai/gpt-oss-20b", PRICING)

    assert counts.duplicate_result_ids == 1  # task_001 appears twice
    assert counts.unexpected_result_ids == 1  # task_999 was never planned
    assert any(m.reconciliation_status == RECON_MALFORMED for m in measurements)
    # task_002/003/004 never arrived in this fixture at all.
    assert counts.missing_requests == 3


def test_full_offline_pipeline_produces_summary_report_and_charts_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    planned_tasks = _pilot_tasks()
    output_rows = parse_batch_output_jsonl((FIXTURES_DIR / "sample_batch_output.jsonl").read_text(encoding="utf-8"))
    error_rows = parse_batch_output_jsonl((FIXTURES_DIR / "sample_batch_errors.jsonl").read_text(encoding="utf-8"))
    measurements, counts = reconcile_batch_results("offline-demo", planned_tasks, output_rows, error_rows, "openai/gpt-oss-20b", PRICING)

    run_dir = tmp_path / "offline-demo"
    write_manifest(run_dir, {"experiment_run_id": "offline-demo", "profile": "pilot"})

    with open(FIXTURES_DIR / "sample_batch_job.json", encoding="utf-8") as f:
        job_fields = json.load(f)
    from labs.lab09_batch_processing.models import BatchJobMetadata

    job = BatchJobMetadata(**{k: v for k, v in job_fields.items() if not k.startswith("_") and k in BatchJobMetadata.__dataclass_fields__})
    write_batch_job(run_dir, job)
    write_batch_measurements(run_dir, measurements)

    from lab_core.config import load_config

    lab09_config = load_config(None).lab09
    summary = compute_and_persist_summary(run_dir, lab09_config, "openai/gpt-oss-20b", PRICING)

    assert summary["batch_counts"]["completed_requests"] == 4
    assert summary["pricing_comparison"]["estimated_batch_cost_usd"] is not None
    assert (run_dir / "analysis_report.md").exists()
    assert (run_dir / "experiment_summary.json").exists()

    from labs.lab09_batch_processing.charts import generate_batch_charts

    # No sequential/concurrent data in this offline-only demo -- charts
    # generation correctly produces nothing rather than fabricating
    # real-time panels that were never measured.
    written = generate_batch_charts(run_dir)
    assert written == []
