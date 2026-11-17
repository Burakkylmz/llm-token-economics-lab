"""Offline documentation-to-implementation checks for benchmark contracts.

When a normalized field, data source, or report metric changes, the reference
documentation must be updated in the same pull request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lab_core.benchmark_report import GROUP_KEYS, NUMERIC_FIELDS, build_benchmark
from lab_core.dashboard_data import EXPERIMENT_SOURCES, EXPORT_COLUMNS, _normalize

ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = ROOT / "docs" / "data-dictionary.md"
METHODOLOGY = ROOT / "docs" / "benchmark-methodology.md"


def _contract_table(filename: Path, contract_name: str) -> list[tuple[str, str]]:
    doc = filename.read_text(encoding="utf-8")
    begin = f"<!-- {contract_name}:start -->"
    end = f"<!-- {contract_name}:end -->"
    assert doc.count(begin) == doc.count(end) == 1, f"Missing/duplicated contract: {contract_name}"
    body = doc.split(begin, 1)[1].split(end, 1)[0]
    tick = chr(96)
    pattern = r"\| " + tick + r"([^" + tick + r"]+)" + tick + r" \| (.*?) \|"
    pairs = []
    for line in body.splitlines():
        match = re.fullmatch(pattern, line.strip())
        if match is not None:
            pairs.append((match.group(1), match.group(2)))
    assert pairs, f"No documented rows for {contract_name}"
    return pairs


def test_normalized_dictionary_matches_approved_export_contract():
    rows = _contract_table(DICTIONARY, "data-dictionary:normalized-fields")
    documented = [name for name, _ in rows]
    assert documented == list(EXPORT_COLUMNS)
    normalized = _normalize(
        {}, lab="lab01", experiment="lab01_effort",
        run_id="run", source="raw_measurements.json", manifest={},
    )
    assert set(normalized) == set(EXPORT_COLUMNS)


def test_allowlisted_sources_match_dictionary_without_unapproved_files():
    rows = _contract_table(DICTIONARY, "data-dictionary:experiment-sources")
    tick = chr(96)
    documented = {
        experiment: tuple(re.findall(tick + "([^" + tick + "]+)" + tick, filenames))
        for experiment, filenames in rows
    }
    assert len(rows) == len(documented)
    assert documented == EXPERIMENT_SOURCES


def test_benchmark_group_keys_and_metric_names_are_documented_exactly():
    keys = [name for name, _ in _contract_table(DICTIONARY, "data-dictionary:group-keys")]
    metrics = [name for name, _ in _contract_table(DICTIONARY, "data-dictionary:report-metrics")]
    formula_metrics = [
        name for name, _ in _contract_table(METHODOLOGY, "benchmark-methodology:metric-fields")
    ]
    assert keys == list(GROUP_KEYS)
    assert metrics == list(NUMERIC_FIELDS)
    assert formula_metrics == list(NUMERIC_FIELDS)


def test_published_worked_example_matches_actual_statistics():
    """Check denominators, nearest-rank percentiles and boolean rates."""
    values = [
        (10, 0.001, 50, True, True),
        (None, None, 60, False, None),
        (30, 0.003, 70, None, False),
        (50, None, 80, False, None),
    ]
    records = [{
        "lab": "lab01",
        "experiment": "lab01_effort",
        "run_id": "example",
        "source": "raw_measurements.json",
        "scenario": "Unspecified",
        "model": "model-1",
        "condition": "control",
        "cost_basis": "Estimated",
        "simulated": False,
        "total_tokens": tokens,
        "estimated_cost_usd": usd,
        "latency_ms": latency,
        "succeeded": succeeded,
        "answer_correct": correct,
    } for tokens, usd, latency, succeeded, correct in values]
    report = build_benchmark(records, lab="lab01", run_id="example")
    assert report["measurement_count"] == 4
    group = report["groups"][0]
    assert group["requests"] == 4
    assert group["metrics"]["total_tokens"]["observed"] == 3
    assert group["metrics"]["total_tokens"]["mean"] == 30
    assert group["metrics"]["estimated_cost_usd"]["observed"] == 2
    assert group["metrics"]["estimated_cost_usd"]["mean"] == pytest.approx(0.002)
    assert group["metrics"]["latency_ms"] == {
        "observed": 4, "mean": 65, "p50": 60, "p95": 80,
    }
    assert group["success"]["evaluated"] == 3
    assert group["success"]["rate"] == pytest.approx(1 / 3)
    assert group["answer_accuracy"] == {"evaluated": 2, "rate": 0.5}


def test_normalized_alias_precedence_and_missing_differ_from_zero():
    raw = {
        "selected_model": "chosen-model",
        "model": "ignored-model",
        "total_latency_ms": 120,
        "performance.client_elapsed_ms": 400,
        "estimated_batch_cost_usd": "0",
        "estimated_standard_cost_usd": "10.00",
        "answer_correct": None,
        "prompt_tokens": "0",
        "completion_tokens": "2.5",
        "succeeded": False,
        "request_status": "ERROR",
        "prompt": "private content not for reports",
    }
    data = _normalize(
        raw, lab="lab05", experiment="lab05_model_routing",
        run_id="test", source="routing_measurements.json", manifest={},
    )
    assert data["model"] == "chosen-model"
    assert data["latency_ms"] == 120
    assert data["estimated_cost_usd"] == 0
    assert data["cost_basis"] == "Batch discounted estimate"
    assert data["prompt_tokens"] == 0
    assert data["completion_tokens"] is None
    assert data["answer_correct"] is None
    assert data["succeeded"] is False
    assert "prompt" not in data


def test_report_format_version_is_not_misrepresented_as_raw_schema_version():
    dictionary = DICTIONARY.read_text(encoding="utf-8")
    methodology = METHODOLOGY.read_text(encoding="utf-8")
    assert "schema_version = 1" in dictionary
    assert "report output" in methodology
    assert "not a" in dictionary.lower() and "raw" in dictionary.lower()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in ("docs/data-dictionary.md", "docs/benchmark-methodology.md"):
        assert target in readme
