from __future__ import annotations

import csv
import json

from lab_core.metrics import ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from lab_core.results_writer import append_csv, write_json


def _make_result(run_id: str) -> ExperimentResult:
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id="exp-1",
            run_id=run_id,
            model_id="llama-3.1-8b-instant",
            execution_mode="sync",
            timestamp="2026-10-21T00:00:00Z",
        ),
        tokens_and_cost=TokenCostMetrics(prompt_tokens=10, completion_tokens=20, total_tokens=30, api_call_count=1),
        performance=PerformanceMetrics(client_elapsed_ms=123.4),
    )


def test_write_json_round_trip(results_dir):
    output_path = results_dir / "run.json"
    write_json([_make_result("run-1")], output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert loaded[0]["metadata"]["run_id"] == "run-1"
    assert loaded[0]["tokens_and_cost"]["prompt_tokens"] == 10


def test_append_csv_writes_header_once(results_dir):
    output_path = results_dir / "run.csv"
    append_csv([_make_result("run-1")], output_path)
    append_csv([_make_result("run-2")], output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["metadata.run_id"] == "run-1"
    assert rows[1]["metadata.run_id"] == "run-2"


def test_append_csv_noop_on_empty_list(results_dir):
    output_path = results_dir / "run.csv"
    append_csv([], output_path)
    assert not output_path.exists()
