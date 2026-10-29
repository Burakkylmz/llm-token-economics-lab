from __future__ import annotations

import json

import pytest

from lab_core.config import Lab10Config
from labs.lab10_conversation_summarization.datasets import load_conversation
from labs.lab10_conversation_summarization.models import FollowupMeasurement, PlannedCall, SummaryMeasurement
from labs.lab10_conversation_summarization.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab10_run_dir,
    read_followup_measurements,
    read_generated_summaries,
    read_manifest,
    read_summary_measurements,
    refuse_if_already_has_measurements,
    write_experiment_plan_csv,
    write_followup_measurements,
    write_generated_summaries,
    write_manifest,
    write_original_conversations,
    write_summary_measurements,
    write_traces,
)


def _config() -> Lab10Config:
    return Lab10Config(
        enabled=True, model="openai/gpt-oss-20b", default_profile="default", dataset_version="v1",
        summarization_max_completion_tokens=1536, answering_reasoning_effort="low", answering_max_completion_tokens=256,
        persist_synthetic_io=True, show_all_io_in_terminal=False, max_api_calls=28, max_estimated_cost_usd=0.10,
    )


def _summary_measurement(dataset_id="customer_support_001", condition_id="summarization_low") -> SummaryMeasurement:
    return SummaryMeasurement(
        experiment_run_id="run-1", execution_index=1, dataset_id=dataset_id, condition_id=condition_id,
        reasoning_effort="low", timestamp="t", model="m", request_status="SUCCESS",
        summary_generation_prompt_tokens=1800, summary_generation_completion_tokens=400,
        summary_generation_estimated_cost_usd=0.001, json_valid=True, schema_errors=(),
    )


def _followup_measurement(question_id="q1_order_id", condition_id="full_history_baseline") -> FollowupMeasurement:
    return FollowupMeasurement(
        experiment_run_id="run-1", execution_index=2, dataset_id="customer_support_001", condition_id=condition_id,
        question_id=question_id, timestamp="t", model="m", reasoning_effort="low", request_status="SUCCESS",
        followup_prompt_tokens=150, followup_completion_tokens=10, followup_estimated_cost_usd=0.0001,
        evaluation_status="CORRECT", answer_correct=True,
    )


def test_run_dir_is_isolated_per_run_id(tmp_path):
    a = lab10_run_dir(tmp_path, "run-a")
    b = lab10_run_dir(tmp_path, "run-b")
    assert a != b
    assert a.parent == b.parent
    assert a.parent.name == "lab10_conversation_summarization"


def test_manifest_round_trip(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", _config(), "default", 14, "v1", {"conversations": "abc"}, {"summarization": "def"})
    write_manifest(run_dir, manifest)
    loaded = read_manifest(run_dir)
    assert loaded["experiment_run_id"] == "run-1"
    assert loaded["planned_calls"] == 14
    assert loaded["dataset_hashes"] == {"conversations": "abc"}


def test_read_manifest_returns_none_when_absent(tmp_path):
    assert read_manifest(lab10_run_dir(tmp_path, "missing")) is None


def test_refuse_if_already_has_measurements_blocks_reuse(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    write_summary_measurements(run_dir, [_summary_measurement()])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_if_already_has_measurements_blocks_on_followup_only_too(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    write_followup_measurements(run_dir, [_followup_measurement()])
    with pytest.raises(RunIdAlreadyUsed):
        refuse_if_already_has_measurements(run_dir)


def test_refuse_if_already_has_measurements_allows_fresh_run(tmp_path):
    refuse_if_already_has_measurements(lab10_run_dir(tmp_path, "fresh"))  # must not raise


def test_summary_measurements_csv_round_trip(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    original = [_summary_measurement(condition_id="summarization_low"), _summary_measurement(condition_id="summarization_medium")]
    write_summary_measurements(run_dir, original)
    loaded = read_summary_measurements(run_dir)
    assert len(loaded) == 2
    assert loaded[0].summary_generation_prompt_tokens == 1800
    assert loaded[0].summary_generation_estimated_cost_usd == 0.001
    assert loaded[0].json_valid is True


def test_followup_measurements_csv_round_trip(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    original = [_followup_measurement("q1_order_id"), _followup_measurement("q2_latest_preference")]
    write_followup_measurements(run_dir, original)
    loaded = read_followup_measurements(run_dir)
    assert len(loaded) == 2
    assert loaded[0].followup_prompt_tokens == 150
    assert loaded[0].answer_correct is True


def test_read_measurements_return_empty_list_when_absent(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "missing")
    assert read_summary_measurements(run_dir) == []
    assert read_followup_measurements(run_dir) == []


def test_generated_summaries_round_trip_preserves_dataset_condition_keys(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    summaries = {
        ("customer_support_001", "summarization_low"): {"overview": "o", "facts": [], "decisions": [], "open_issues": [], "timeline": []},
    }
    write_generated_summaries(run_dir, summaries)
    loaded = read_generated_summaries(run_dir)
    assert loaded == summaries


def test_read_generated_summaries_empty_when_absent(tmp_path):
    assert read_generated_summaries(lab10_run_dir(tmp_path, "missing")) == {}


def test_experiment_plan_csv_written(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    plan = [PlannedCall(execution_index=1, dataset_id="customer_support_001", condition_id="summarization_low", call_purpose="summary_generation", reasoning_effort="low")]
    write_experiment_plan_csv(run_dir, plan)
    content = (run_dir / "experiment_plan.csv").read_text(encoding="utf-8")
    assert "summary_generation" in content


def test_original_conversations_written_as_json(tmp_path):
    run_dir = lab10_run_dir(tmp_path, "run-1")
    conversation = load_conversation("customer_support_001")
    write_original_conversations(run_dir, {"customer_support_001": conversation})
    with open(run_dir / "original_conversation.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert "ORD-74291" in payload["customer_support_001"]["messages"][0]["content"] or any(
        "ORD-74291" in m["content"] for m in payload["customer_support_001"]["messages"]
    )


def test_write_traces_splits_summarization_only_subset(tmp_path):
    from labs.lab10_conversation_summarization.models import TraceRecord

    run_dir = lab10_run_dir(tmp_path, "run-1")
    traces = [
        TraceRecord(experiment_run_id="run-1", dataset_id="d", condition_id="summarization_low", call_id="c1", call_purpose="summary_generation", question_id=None, request_model="m", request_parameters={}, request_messages=[], response_visible_content=None, response_finish_reason=None, response_usage={}, response_latency_ms=None, evaluation_status=None),
        TraceRecord(experiment_run_id="run-1", dataset_id="d", condition_id="full_history_baseline", call_id="c2", call_purpose="followup_answer", question_id="q1", request_model="m", request_parameters={}, request_messages=[], response_visible_content=None, response_finish_reason=None, response_usage={}, response_latency_ms=None, evaluation_status="CORRECT"),
    ]
    write_traces(run_dir, traces)
    with open(run_dir / "request_response_inspection.json", encoding="utf-8") as f:
        all_traces = json.load(f)
    with open(run_dir / "summarization_requests.json", encoding="utf-8") as f:
        summarization_only = json.load(f)
    assert len(all_traces) == 2
    assert len(summarization_only) == 1
    assert summarization_only[0]["call_purpose"] == "summary_generation"
