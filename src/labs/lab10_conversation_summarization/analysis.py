"""Derives every downstream report/chart input (Part 9-11, 14.3) purely
from already-recorded measurements -- never from live state. This is the
ONE computation path shared by `lab10 all` (computed right after live
execution) and `lab10 backfill` (computed from persisted CSV/JSON files),
so a backfilled run always produces byte-for-byte the same derived
numbers as a freshly-executed one, and recomputation never needs an API
call (Part 14.3).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from labs.lab10_conversation_summarization.evaluation import evaluate_fact_retention
from labs.lab10_conversation_summarization.models import (
    CONDITION_FULL_HISTORY_BASELINE,
    CONDITION_SUMMARIZATION_LOW,
    CONDITION_SUMMARIZATION_MEDIUM,
    CONDITIONS,
    FollowupMeasurement,
    SummaryMeasurement,
)
from labs.lab10_conversation_summarization.stats import (
    ConditionWorkflowCost,
    compute_break_even,
    compute_context_comparison,
    compute_workflow_cost,
    cost_per_correct_answer,
    net_savings_percent,
    net_savings_usd,
    summarize_followups,
)


def _ordered_dataset_ids(summary_measurements: list[SummaryMeasurement], followup_measurements: list[FollowupMeasurement]) -> list[str]:
    seen: dict[str, int] = {}
    for m in summary_measurements:
        seen.setdefault(m.dataset_id, m.execution_index)
    for m in followup_measurements:
        seen.setdefault(m.dataset_id, m.execution_index)
    return [d for d, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def compute_reports(
    manifest: dict,
    summary_measurements: list[SummaryMeasurement],
    followup_measurements: list[FollowupMeasurement],
    generated_summaries: dict[tuple[str, str], dict],
    dataset_version: str = "v1",
) -> dict:
    dataset_ids = _ordered_dataset_ids(summary_measurements, followup_measurements)

    conditions: list[dict] = []
    context_rows: list[dict] = []
    cumulative: list[dict] = []
    break_even_rows: list[dict] = []
    quality: list[dict] = []

    for dataset_id in dataset_ids:
        followups_by_condition = {
            cid: sorted(
                (m for m in followup_measurements if m.dataset_id == dataset_id and m.condition_id == cid),
                key=lambda m: m.execution_index,
            )
            for cid in CONDITIONS
        }
        summary_by_condition = {
            cid: next((m for m in summary_measurements if m.dataset_id == dataset_id and m.condition_id == cid), None)
            for cid in (CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM)
        }

        # Per-question context comparison rows, in baseline execution order.
        question_ids = [m.question_id for m in followups_by_condition[CONDITION_FULL_HISTORY_BASELINE]]
        tokens_by_question: dict[str, dict[str, Optional[int]]] = {qid: {} for qid in question_ids}
        for cid in CONDITIONS:
            by_qid = {m.question_id: m for m in followups_by_condition[cid]}
            for qid in question_ids:
                m = by_qid.get(qid)
                tokens_by_question[qid][cid] = m.followup_prompt_tokens if m else None

        dataset_context_rows = []
        for qid in question_ids:
            row = compute_context_comparison(
                dataset_id, qid,
                tokens_by_question[qid].get(CONDITION_FULL_HISTORY_BASELINE),
                tokens_by_question[qid].get(CONDITION_SUMMARIZATION_LOW),
                tokens_by_question[qid].get(CONDITION_SUMMARIZATION_MEDIUM),
            )
            dataset_context_rows.append(row)
        context_rows.extend(dataset_context_rows)

        def _mean(values: list[Optional[float]]) -> Optional[float]:
            usable = [v for v in values if v is not None]
            return (sum(usable) / len(usable)) if usable else None

        baseline_workflow = compute_workflow_cost(dataset_id, CONDITION_FULL_HISTORY_BASELINE, None, followups_by_condition[CONDITION_FULL_HISTORY_BASELINE])

        for cid in CONDITIONS:
            followups = followups_by_condition[cid]
            followup_stats = summarize_followups(followups)
            summary_m = summary_by_condition.get(cid)
            workflow = compute_workflow_cost(dataset_id, cid, summary_m, followups)

            fact_result = None
            if cid in (CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM):
                summary_dict = generated_summaries.get((dataset_id, cid))
                fact_result = evaluate_fact_retention(dataset_id, summary_dict)
                quality.append({
                    "dataset_id": dataset_id, "condition_id": cid,
                    "evaluated_answers": followup_stats.evaluated_answers, "correct_answers": followup_stats.correct_answers,
                    "answer_accuracy": followup_stats.answer_accuracy, "invalid_format_count": followup_stats.invalid_format_count,
                    "missing_answer_count": followup_stats.missing_answer_count,
                    "fact_retention_rate": fact_result.fact_retention_rate,
                    "known_fact_violation_count": fact_result.known_fact_violation_count,
                    "fact_checks": [asdict(c) for c in fact_result.checks],
                })
            else:
                quality.append({
                    "dataset_id": dataset_id, "condition_id": cid,
                    "evaluated_answers": followup_stats.evaluated_answers, "correct_answers": followup_stats.correct_answers,
                    "answer_accuracy": followup_stats.answer_accuracy, "invalid_format_count": followup_stats.invalid_format_count,
                    "missing_answer_count": followup_stats.missing_answer_count,
                    "fact_retention_rate": None, "known_fact_violation_count": None, "fact_checks": [],
                })

            if cid == CONDITION_FULL_HISTORY_BASELINE:
                savings_usd = None
                savings_pct = None
                reduction_pct = None
                ratio = None
            else:
                savings_usd = net_savings_usd(baseline_workflow.total_workflow_cost_usd, workflow.total_workflow_cost_usd)
                savings_pct = net_savings_percent(baseline_workflow.total_workflow_cost_usd, workflow.total_workflow_cost_usd)
                reduction_key = "low_context_reduction_percent" if cid == CONDITION_SUMMARIZATION_LOW else "medium_context_reduction_percent"
                ratio_key = "low_compression_ratio" if cid == CONDITION_SUMMARIZATION_LOW else "medium_compression_ratio"
                reduction_pct = _mean([getattr(r, reduction_key) for r in dataset_context_rows])
                ratio = _mean([getattr(r, ratio_key) for r in dataset_context_rows])

            mean_prompt_tokens = _mean([m.followup_prompt_tokens for m in followups])

            conditions.append({
                "dataset_id": dataset_id, "condition_id": cid,
                "total_requests": len(followups) + (1 if summary_m is not None else 0),
                "successful_requests": followup_stats.evaluated_answers + (1 if (summary_m is not None and summary_m.request_status == "SUCCESS") else 0),
                "evaluated_answers": followup_stats.evaluated_answers, "correct_answers": followup_stats.correct_answers,
                "answer_accuracy": followup_stats.answer_accuracy, "invalid_format_count": followup_stats.invalid_format_count,
                "missing_answer_count": followup_stats.missing_answer_count,
                "summary_generation_prompt_tokens": summary_m.summary_generation_prompt_tokens if summary_m else None,
                "summary_generation_completion_tokens": summary_m.summary_generation_completion_tokens if summary_m else None,
                "summary_generation_reasoning_tokens": summary_m.summary_generation_reasoning_tokens if summary_m else None,
                "summary_generation_non_reasoning_completion_tokens": summary_m.summary_generation_non_reasoning_completion_tokens if summary_m else None,
                "followup_total_prompt_tokens": followup_stats.total_prompt_tokens,
                "followup_total_completion_tokens": followup_stats.total_completion_tokens,
                "followup_total_reasoning_tokens": followup_stats.total_reasoning_tokens,
                "summary_generation_cost_usd": workflow.summary_generation_cost_usd,
                "followup_cost_usd": workflow.followup_cost_usd,
                "total_workflow_cost_usd": workflow.total_workflow_cost_usd,
                "net_savings_usd": savings_usd, "net_savings_percent": savings_pct,
                "cost_per_correct_answer_usd": cost_per_correct_answer(workflow.total_workflow_cost_usd, followup_stats.correct_answers),
                "fact_retention_rate": fact_result.fact_retention_rate if fact_result else None,
                "known_fact_violation_count": fact_result.known_fact_violation_count if fact_result else None,
                "mean_context_reduction_percent": reduction_pct, "mean_compression_ratio": ratio,
                "mean_request_prompt_tokens": mean_prompt_tokens,
            })

            # Cumulative input tokens (chart 3): low/medium start at their
            # summary's own prompt tokens (the one-time overhead), baseline
            # starts at 0 -- then each follow-up's prompt tokens accumulate.
            running = (summary_m.summary_generation_prompt_tokens if summary_m else 0) or 0
            has_summary_tokens = summary_m is None or summary_m.summary_generation_prompt_tokens is not None
            for i, m in enumerate(followups, start=1):
                if not has_summary_tokens or m.followup_prompt_tokens is None:
                    cumulative.append({"dataset_id": dataset_id, "condition_id": cid, "call_index": i, "cumulative_input_tokens": None})
                    has_summary_tokens = False
                    continue
                running += m.followup_prompt_tokens
                cumulative.append({"dataset_id": dataset_id, "condition_id": cid, "call_index": i, "cumulative_input_tokens": running})

        # Break-even (chart 4, Part 10.6): low/medium vs baseline, in
        # matched question order.
        baseline_costs = [m.followup_estimated_cost_usd for m in followups_by_condition[CONDITION_FULL_HISTORY_BASELINE]]
        for cid in (CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM):
            summary_m = summary_by_condition.get(cid)
            strategy_costs = [m.followup_estimated_cost_usd for m in followups_by_condition[cid]]
            result = compute_break_even(
                summary_m.summary_generation_estimated_cost_usd if summary_m else None, baseline_costs, strategy_costs,
            )
            for i, (b, s) in enumerate(zip(result.baseline_cumulative_costs, result.summary_cumulative_costs), start=1):
                break_even_rows.append({
                    "dataset_id": dataset_id, "condition_id": cid, "call_index": i,
                    "baseline_cumulative_cost_usd": b, "summary_cumulative_cost_usd": s,
                    "break_even_call_index": result.break_even_call_index, "break_even_status": result.break_even_status,
                })

    workflow_costs = [
        ConditionWorkflowCost(
            dataset_id=c["dataset_id"], condition_id=c["condition_id"], summary_generation_cost_usd=c["summary_generation_cost_usd"],
            followup_cost_usd=c["followup_cost_usd"], total_workflow_cost_usd=c["total_workflow_cost_usd"],
        )
        for c in conditions
    ]
    cost_extra_fields = [
        {"net_savings_usd": c["net_savings_usd"], "net_savings_percent": c["net_savings_percent"], "cost_per_correct_answer_usd": c["cost_per_correct_answer_usd"]}
        for c in conditions
    ]

    return {
        "datasets": dataset_ids,
        "conditions": conditions,
        "context_comparison": context_rows,
        "cumulative": cumulative,
        "break_even": break_even_rows,
        "quality": quality,
        "workflow_costs": workflow_costs,
        "cost_extra_fields": cost_extra_fields,
    }


_TOKEN_COMPARISON_KEYS = (
    "dataset_id", "condition_id", "summary_generation_prompt_tokens", "summary_generation_completion_tokens",
    "summary_generation_reasoning_tokens", "summary_generation_non_reasoning_completion_tokens",
    "followup_total_prompt_tokens", "followup_total_completion_tokens", "followup_total_reasoning_tokens",
)


def token_comparison_rows(conditions: list[dict]) -> list[dict]:
    return [{key: c[key] for key in _TOKEN_COMPARISON_KEYS} for c in conditions]


def break_even_summary_rows(break_even_rows: list[dict]) -> list[dict]:
    """One row per (dataset_id, condition_id) pair -- break_even_call_index
    and break_even_status are repeated across every call_index row for
    that pair, so the first occurrence is representative."""
    seen: dict[tuple[str, str], dict] = {}
    for row in break_even_rows:
        key = (row["dataset_id"], row["condition_id"])
        seen.setdefault(key, row)
    return list(seen.values())


def persist_computed_reports(run_dir: Path, manifest: dict, computed: dict) -> dict:
    """Writes every derived file (context/token/cost comparison, break-even,
    quality evaluation, experiment summary, analysis report) from an
    already-computed report bundle. Shared verbatim by `lab10 all`'s live
    path and `lab10 backfill`'s offline path, so the two can never drift
    apart. Returns the `experiment_summary.json` payload it wrote."""
    from labs.lab10_conversation_summarization import persistence, report

    persistence.write_context_comparison_csv(run_dir, computed["context_comparison"])
    persistence.write_token_comparison_csv(run_dir, token_comparison_rows(computed["conditions"]))
    persistence.write_cost_comparison_csv(run_dir, computed["workflow_costs"], computed["cost_extra_fields"])
    persistence.write_quality_evaluation(run_dir, {"quality": computed["quality"]})
    persistence.write_break_even_csv(run_dir, computed["break_even"])

    summary_payload = {
        "experiment_run_id": manifest.get("experiment_run_id"),
        "profile": manifest.get("profile"),
        "planned_calls": manifest.get("planned_calls"),
        "datasets": computed["datasets"],
        "conditions": computed["conditions"],
        "context_comparison": [asdict(r) for r in computed["context_comparison"]],
        "cumulative": computed["cumulative"],
        "break_even": computed["break_even"],
        "quality": computed["quality"],
    }
    persistence.write_experiment_summary_json(run_dir, summary_payload)

    report_markdown = report.build_analysis_report_markdown(manifest, computed["conditions"], break_even_summary_rows(computed["break_even"]))
    persistence.write_analysis_report(run_dir, report_markdown)

    return summary_payload
