from __future__ import annotations

from labs.lab10_conversation_summarization.analysis import break_even_summary_rows, compute_reports, token_comparison_rows
from labs.lab10_conversation_summarization.models import FollowupMeasurement, SummaryMeasurement

DATASET_ID = "customer_support_001"

VALID_SUMMARY = {
    "overview": "o",
    "facts": [
        {"name": "order_id", "value": "ORD-74291", "status": "confirmed"},
        {"name": "support_ticket", "value": "TKT-516", "status": "confirmed"},
        {"name": "delivery_city", "value": "Izmir", "status": "confirmed"},
    ],
    "decisions": [
        {"topic": "delivery date", "current_value": "2026-11-05", "superseded_value": "2026-11-03"},
        {"topic": "delivery time preference", "current_value": "afternoon", "superseded_value": "morning"},
    ],
    "open_issues": [{"issue": "return request status", "status": "pending"}],
    "timeline": [],
}


def _summary(condition_id, prompt_tokens=1800, cost=0.01):
    return SummaryMeasurement(
        experiment_run_id="run-1", execution_index=1, dataset_id=DATASET_ID, condition_id=condition_id,
        reasoning_effort="low" if condition_id == "summarization_low" else "medium", timestamp="t", model="m",
        request_status="SUCCESS", summary_generation_prompt_tokens=prompt_tokens, summary_generation_completion_tokens=300,
        summary_generation_reasoning_tokens=100, summary_generation_non_reasoning_completion_tokens=200,
        summary_generation_estimated_cost_usd=cost,
    )


def _followup(condition_id, question_id, status="CORRECT", prompt_tokens=200, cost=0.0005, execution_index=10):
    return FollowupMeasurement(
        experiment_run_id="run-1", execution_index=execution_index, dataset_id=DATASET_ID, condition_id=condition_id,
        question_id=question_id, timestamp="t", model="m", reasoning_effort="low", request_status="SUCCESS",
        followup_prompt_tokens=prompt_tokens, followup_completion_tokens=15, followup_estimated_cost_usd=cost,
        evaluation_status=status, answer_correct=(status == "CORRECT"),
    )


def _build_full_dataset_measurements():
    summaries = [_summary("summarization_low"), _summary("summarization_medium")]
    followups = []
    question_ids = ["q1", "q2"]
    for i, qid in enumerate(question_ids):
        followups.append(_followup("full_history_baseline", qid, prompt_tokens=4000, cost=0.001, execution_index=i))
        followups.append(_followup("summarization_low", qid, prompt_tokens=500, cost=0.0003, execution_index=i + 10))
        followups.append(_followup("summarization_medium", qid, prompt_tokens=550, cost=0.0003, execution_index=i + 20))
    generated = {(DATASET_ID, "summarization_low"): VALID_SUMMARY, (DATASET_ID, "summarization_medium"): VALID_SUMMARY}
    return summaries, followups, generated


def test_compute_reports_produces_one_condition_row_per_condition():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    assert {c["condition_id"] for c in computed["conditions"]} == {"full_history_baseline", "summarization_low", "summarization_medium"}
    assert computed["datasets"] == [DATASET_ID]


def test_compute_reports_summary_generation_cost_included_once_per_condition():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    low = next(c for c in computed["conditions"] if c["condition_id"] == "summarization_low")
    assert low["summary_generation_cost_usd"] == 0.01
    assert low["followup_cost_usd"] == 0.0006  # 2 questions x 0.0003
    assert abs(low["total_workflow_cost_usd"] - 0.0106) < 1e-9


def test_compute_reports_baseline_has_no_summary_cost_or_fact_retention():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    baseline = next(c for c in computed["conditions"] if c["condition_id"] == "full_history_baseline")
    assert baseline["summary_generation_cost_usd"] is None
    assert baseline["fact_retention_rate"] is None


def test_compute_reports_context_reduction_measured_per_question_and_averaged():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    assert len(computed["context_comparison"]) == 2  # 2 questions
    for row in computed["context_comparison"]:
        assert row.low_context_reduction_tokens == 3500  # 4000 - 500
    low = next(c for c in computed["conditions"] if c["condition_id"] == "summarization_low")
    assert low["mean_context_reduction_percent"] == 87.5


def test_compute_reports_net_savings_computed_relative_to_baseline():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    baseline = next(c for c in computed["conditions"] if c["condition_id"] == "full_history_baseline")
    low = next(c for c in computed["conditions"] if c["condition_id"] == "summarization_low")
    assert low["net_savings_usd"] == baseline["total_workflow_cost_usd"] - low["total_workflow_cost_usd"]
    assert low["net_savings_usd"] < 0  # the one-time summary cost (0.01) dwarfs 2 questions' savings here


def test_compute_reports_fact_retention_and_zero_known_violations_for_a_correct_summary():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    low = next(c for c in computed["conditions"] if c["condition_id"] == "summarization_low")
    assert low["fact_retention_rate"] == 1.0
    assert low["known_fact_violation_count"] == 0


def test_compute_reports_cumulative_tokens_start_at_summary_overhead_for_summary_conditions():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    low_cumulative = sorted((r for r in computed["cumulative"] if r["condition_id"] == "summarization_low"), key=lambda r: r["call_index"])
    assert low_cumulative[0]["cumulative_input_tokens"] == 1800 + 500  # summary overhead + first followup
    baseline_cumulative = sorted((r for r in computed["cumulative"] if r["condition_id"] == "full_history_baseline"), key=lambda r: r["call_index"])
    assert baseline_cumulative[0]["cumulative_input_tokens"] == 4000  # no summary overhead


def test_compute_reports_break_even_rows_cover_low_and_medium_only():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    conditions_seen = {r["condition_id"] for r in computed["break_even"]}
    assert conditions_seen == {"summarization_low", "summarization_medium"}


def test_compute_reports_missing_generated_summary_marks_fact_retention_unavailable():
    summaries, followups, _ = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated_summaries={})
    low = next(c for c in computed["conditions"] if c["condition_id"] == "summarization_low")
    assert low["fact_retention_rate"] is None


def test_compute_reports_summarize_only_measurements_produce_no_followup_stats():
    summaries = [_summary("summarization_low"), _summary("summarization_medium")]
    generated = {(DATASET_ID, "summarization_low"): VALID_SUMMARY, (DATASET_ID, "summarization_medium"): VALID_SUMMARY}
    computed = compute_reports({}, summaries, [], generated)
    for c in computed["conditions"]:
        assert c["evaluated_answers"] == 0
        assert c["answer_accuracy"] is None


def test_token_comparison_rows_extracts_expected_keys():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    rows = token_comparison_rows(computed["conditions"])
    assert all("summary_generation_prompt_tokens" in r for r in rows)
    assert all("dataset_id" in r for r in rows)


def test_break_even_summary_rows_deduplicates_to_one_per_pair():
    summaries, followups, generated = _build_full_dataset_measurements()
    computed = compute_reports({}, summaries, followups, generated)
    summary_rows = break_even_summary_rows(computed["break_even"])
    assert len(summary_rows) == 2  # one for low, one for medium
