"""Markdown analysis report for Lab10 (Part 13.5/20.1's two core lessons:
a shorter summary is not necessarily a better summary, and context
reduction is not equivalent to total workflow cost reduction)."""

from __future__ import annotations

from typing import Optional

CONDITION_LABELS = {"full_history_baseline": "Full History Baseline", "summarization_low": "Summarization (Low)", "summarization_medium": "Summarization (Medium)"}


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value: Optional[float]) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def build_analysis_report_markdown(manifest: dict, conditions: list[dict], break_even_summaries: list[dict]) -> str:
    lines = ["# LAB10 -- Conversation Summarization & Context Economics: Analysis Report", ""]
    lines.append(f"Experiment Run ID: `{manifest.get('experiment_run_id')}`")
    lines.append(f"Profile: `{manifest.get('profile')}`")
    lines.append(f"Dataset Version: `{manifest.get('dataset_version')}`")
    lines.append(f"Planned Calls: {manifest.get('planned_calls')}")
    lines.append("")

    lines.append("## Condition Summary")
    lines.append("")
    lines.append("| Dataset | Condition | Evaluated | Correct | Accuracy | Fact Retention | Violations | Workflow Cost |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in conditions:
        lines.append(
            f"| {c['dataset_id']} | {CONDITION_LABELS.get(c['condition_id'], c['condition_id'])} | "
            f"{c.get('evaluated_answers', 0)} | {c.get('correct_answers', 0)} | {_fmt_pct(c.get('answer_accuracy'))} | "
            f"{_fmt_pct(c.get('fact_retention_rate'))} | {c.get('known_fact_violation_count', 'N/A')} | "
            f"{_fmt_cost(c.get('total_workflow_cost_usd'))} |"
        )
    lines.append("")

    lines.append("## Context Compression")
    lines.append("")
    for c in conditions:
        if c["condition_id"] == "full_history_baseline":
            continue
        reduction = c.get("mean_context_reduction_percent")
        ratio = c.get("mean_compression_ratio")
        lines.append(
            f"- {c['dataset_id']} / {CONDITION_LABELS.get(c['condition_id'])}: "
            f"mean input-token reduction {_fmt_pct(reduction and reduction / 100)}, compression ratio {_fmt(ratio)}x"
        )
    lines.append("")

    lines.append("## Break-Even Analysis")
    lines.append("")
    for b in break_even_summaries:
        status = b.get("break_even_status")
        index = b.get("break_even_call_index")
        lines.append(
            f"- {b['dataset_id']} / {CONDITION_LABELS.get(b['condition_id'])}: "
            f"{'break-even reached at follow-up #' + str(index) if status == 'REACHED' else 'break-even NOT reached within the measured follow-up questions'}"
        )
    lines.append("")

    lines.append("## Net Savings")
    lines.append("")
    for c in conditions:
        if c["condition_id"] == "full_history_baseline":
            continue
        savings = c.get("net_savings_usd")
        savings_pct = c.get("net_savings_percent")
        verdict = "saved money" if (savings is not None and savings > 0) else ("cost MORE than baseline" if (savings is not None and savings < 0) else "inconclusive (missing cost data)")
        lines.append(f"- {c['dataset_id']} / {CONDITION_LABELS.get(c['condition_id'])}: net savings {_fmt_cost(savings)} ({_fmt_pct(savings_pct and savings_pct / 100)}) -- {verdict}")
    lines.append("")

    lines.append("## Experimental Limitations")
    lines.append("")
    lines.append(
        "- This is an educational controlled experiment on two synthetic golden datasets with "
        "at most 4 follow-up questions each -- NOT a statistically robust production benchmark. "
        "No claim of universal superiority between Low and Medium reasoning effort is made."
    )
    lines.append(
        "- A shorter summary is not necessarily a better summary: context reduction is reported "
        "separately from fact retention and answer accuracy, and a high compression ratio with "
        "low fact retention is a FAILURE of the summarization strategy, not a success."
    )
    lines.append(
        "- Context reduction is not equivalent to total workflow cost reduction: summary "
        "generation overhead is included in every cumulative cost figure above, and break-even "
        "is only ever reported as OBSERVED over the measured follow-up questions, never "
        "extrapolated or projected beyond them."
    )
    lines.append(
        "- Known-fact-violation counts are a bounded, documented set of checks for this "
        "project's two specific golden datasets -- not a complete or general hallucination "
        "detector."
    )
    lines.append("")

    return "\n".join(lines)
