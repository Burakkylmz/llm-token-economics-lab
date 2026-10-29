"""Chart generation for Lab10 (Part 15). Reads only `experiment_summary.json`
from a completed run directory (same convention as every other redesigned
lab). Charts 3/4 (cumulative curves) use the FIRST dataset present as their
representative sequence -- documented in-chart, never averaged across
datasets with different question counts/content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import CATEGORICAL_DARK, MUTED_INK, add_footer_caption, new_figure, no_data_text, save_png

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"

CONDITION_LABELS = {"full_history_baseline": "Full History", "summarization_low": "Summary (Low)", "summarization_medium": "Summary (Medium)"}
CONDITION_COLOR = {"full_history_baseline": CATEGORICAL_DARK[0], "summarization_low": CATEGORICAL_DARK[2], "summarization_medium": CATEGORICAL_DARK[1]}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _by_condition(conditions: list[dict], dataset_id: Optional[str] = None) -> dict[str, dict]:
    rows = [c for c in conditions if dataset_id is None or c["dataset_id"] == dataset_id]
    result: dict[str, dict] = {}
    for c in rows:
        cid = c["condition_id"]
        result.setdefault(cid, []).append(c)
    # Aggregate across datasets when dataset_id is None: mean of per-dataset means.
    merged = {}
    for cid, rows_for_cid in result.items():
        merged[cid] = rows_for_cid[0] if len(rows_for_cid) == 1 else _mean_merge(rows_for_cid)
    return merged


def _mean_merge(rows: list[dict]) -> dict:
    keys = set().union(*(r.keys() for r in rows))
    merged = {}
    for key in keys:
        values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        merged[key] = (sum(values) / len(values)) if values else rows[0].get(key)
    return merged


def _draw_context_size(ax, conditions: list[dict]) -> list[int]:
    by_cond = _by_condition(conditions)
    order = ["full_history_baseline", "summarization_low", "summarization_medium"]
    usable = [(cid, by_cond[cid]) for cid in order if cid in by_cond and by_cond[cid].get("mean_request_prompt_tokens") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Context Size Before vs After", fontsize=10)
        return []
    xs = [CONDITION_LABELS[cid] for cid, _ in usable]
    ys = [row["mean_request_prompt_tokens"] for _, row in usable]
    colors = [CONDITION_COLOR[cid] for cid, _ in usable]
    ax.bar(xs, ys, color=colors, width=0.5)
    ax.set_ylabel("Mean Request Input Tokens")
    ax.set_title("Context Size Before vs After", fontsize=10)
    return [row.get("evaluated_answers", 0) for _, row in usable]


def _draw_summary_token_breakdown(ax, conditions: list[dict]) -> list[int]:
    by_cond = _by_condition(conditions)
    order = ["summarization_low", "summarization_medium"]
    usable = [(cid, by_cond[cid]) for cid in order if cid in by_cond and by_cond[cid].get("summary_generation_prompt_tokens") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Summary Generation Token Breakdown", fontsize=10)
        return []
    xs = [CONDITION_LABELS[cid] for cid, _ in usable]
    inputs = [row["summary_generation_prompt_tokens"] or 0 for _, row in usable]
    reasoning = [row.get("summary_generation_reasoning_tokens") or 0 for _, row in usable]
    non_reasoning = [row.get("summary_generation_non_reasoning_completion_tokens") or 0 for _, row in usable]
    ax.bar(xs, inputs, color=CATEGORICAL_DARK[0], width=0.5, label="Input Tokens")
    ax.bar(xs, reasoning, bottom=inputs, color=CATEGORICAL_DARK[4], width=0.5, label="Reasoning Tokens")
    bottoms2 = [i + r for i, r in zip(inputs, reasoning)]
    ax.bar(xs, non_reasoning, bottom=bottoms2, color=CATEGORICAL_DARK[2], width=0.5, label="Non-Reasoning Completion Tokens")
    ax.set_ylabel("Tokens")
    ax.set_title("Summary Generation Token Breakdown", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [1 for _ in usable]


def _draw_cumulative_tokens(ax, cumulative: list[dict], representative_dataset: Optional[str]) -> list[int]:
    if representative_dataset is None:
        no_data_text(ax)
        ax.set_title("Cumulative Input Token Consumption", fontsize=10)
        return []
    rows = [r for r in cumulative if r["dataset_id"] == representative_dataset]
    if not rows:
        no_data_text(ax)
        ax.set_title("Cumulative Input Token Consumption", fontsize=10)
        return []
    order = ["full_history_baseline", "summarization_low", "summarization_medium"]
    plotted = 0
    for cid in order:
        series = sorted((r for r in rows if r["condition_id"] == cid), key=lambda r: r["call_index"])
        series = [r for r in series if r.get("cumulative_input_tokens") is not None]
        if not series:
            continue
        xs = [r["call_index"] for r in series]
        ys = [r["cumulative_input_tokens"] for r in series]
        ax.plot(xs, ys, marker="o", color=CONDITION_COLOR[cid], label=CONDITION_LABELS[cid])
        plotted += 1
    if plotted == 0:
        no_data_text(ax)
    ax.set_xlabel(f"Follow-Up Question # (dataset: {representative_dataset})")
    ax.set_ylabel("Cumulative Input Tokens")
    ax.set_title("Cumulative Input Token Consumption", fontsize=10)
    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    return [plotted]


def _draw_cumulative_cost_break_even(ax, break_even: list[dict], representative_dataset: Optional[str]) -> list[int]:
    if representative_dataset is None:
        no_data_text(ax)
        ax.set_title("Cumulative Cost & Break-Even", fontsize=10)
        return []
    rows = [r for r in break_even if r["dataset_id"] == representative_dataset]
    if not rows:
        no_data_text(ax)
        ax.set_title("Cumulative Cost & Break-Even", fontsize=10)
        return []

    plotted = 0
    baseline_plotted = False
    for cid in ("summarization_low", "summarization_medium"):
        series = sorted((r for r in rows if r["condition_id"] == cid), key=lambda r: r["call_index"])
        if not series:
            continue
        if not baseline_plotted:
            baseline_xs = [r["call_index"] for r in series if r.get("baseline_cumulative_cost_usd") is not None]
            baseline_ys = [r["baseline_cumulative_cost_usd"] for r in series if r.get("baseline_cumulative_cost_usd") is not None]
            if baseline_xs:
                ax.plot(baseline_xs, baseline_ys, marker="o", color=CONDITION_COLOR["full_history_baseline"], label=CONDITION_LABELS["full_history_baseline"])
                plotted += 1
            baseline_plotted = True
        xs = [r["call_index"] for r in series if r.get("summary_cumulative_cost_usd") is not None]
        ys = [r["summary_cumulative_cost_usd"] for r in series if r.get("summary_cumulative_cost_usd") is not None]
        if xs:
            ax.plot(xs, ys, marker="o", color=CONDITION_COLOR[cid], label=CONDITION_LABELS[cid])
            plotted += 1
        be_index = series[0].get("break_even_call_index")
        if be_index:
            ax.axvline(be_index, color=MUTED_INK, linestyle="--", linewidth=1)

    if plotted == 0:
        no_data_text(ax)
    ax.set_xlabel(f"Follow-Up Question # (dataset: {representative_dataset})")
    ax.set_ylabel("Cumulative Estimated Cost (USD)")
    ax.set_title("Cumulative Cost & Break-Even", fontsize=10)
    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    return [plotted]


def _draw_retention_and_accuracy(ax, conditions: list[dict]) -> list[int]:
    by_cond = _by_condition(conditions)
    order = ["full_history_baseline", "summarization_low", "summarization_medium"]
    usable = [(cid, by_cond[cid]) for cid in order if cid in by_cond]
    if not usable:
        no_data_text(ax)
        ax.set_title("Information Retention & Accuracy", fontsize=10)
        return []

    xs = range(len(usable))
    width = 0.35
    accuracy = [row.get("answer_accuracy") or 0 for _, row in usable]
    retention = [row.get("fact_retention_rate") or 0 for _, row in usable]
    ax.bar([x - width / 2 for x in xs], accuracy, width=width, color=CATEGORICAL_DARK[0], label="Answer Accuracy")
    ax.bar([x + width / 2 for x in xs], retention, width=width, color=CATEGORICAL_DARK[2], label="Fact Retention Rate")
    for x, (cid, row) in zip(xs, usable):
        violations = row.get("known_fact_violation_count")
        if violations:
            ax.annotate(f"violations: {violations}", (x, max(accuracy[x], retention[x]) + 0.05), ha="center", fontsize=7, color=MUTED_INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([CONDITION_LABELS[cid] for cid, _ in usable], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("Information Retention & Accuracy", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [row.get("evaluated_answers", 0) for _, row in usable]


def _draw_quality_vs_cost(ax, conditions: list[dict]) -> list[int]:
    by_cond = _by_condition(conditions)
    order = ["full_history_baseline", "summarization_low", "summarization_medium"]
    usable = [(cid, by_cond[cid]) for cid in order if cid in by_cond and by_cond[cid].get("total_workflow_cost_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Quality vs Cost Trade-off", fontsize=10)
        return []

    for cid, row in usable:
        cost = row["total_workflow_cost_usd"]
        accuracy = row.get("answer_accuracy") or 0
        ax.scatter([cost], [accuracy], color=CONDITION_COLOR[cid], s=80, zorder=3)
        ax.annotate(CONDITION_LABELS[cid], (cost, accuracy), textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED_INK)
    ax.set_xlabel("Total Workflow Cost (USD)")
    ax.set_ylabel("Answer Accuracy")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Quality vs Cost Trade-off", fontsize=10)
    return [row.get("evaluated_answers", 0) for _, row in usable]


def generate_summarization_charts(run_dir: Path) -> list[Path]:
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    conditions = summary.get("conditions", [])
    if not conditions:
        return []
    cumulative = summary.get("cumulative", [])
    break_even = summary.get("break_even", [])
    representative_dataset = summary.get("datasets", [None])[0] if summary.get("datasets") else None

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    fig, ax = new_figure(figsize=(7, 5))
    from lab_core.charts import sample_size_caption

    counts = _draw_context_size(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "context_size_before_vs_after.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_summary_token_breakdown(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "summary_token_breakdown.png"))

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cumulative_tokens(ax, cumulative, representative_dataset)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cumulative_input_tokens.png"))

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cumulative_cost_break_even(ax, break_even, representative_dataset)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cumulative_cost_break_even.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_retention_and_accuracy(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "retention_and_accuracy.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_quality_vs_cost(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "quality_vs_cost.png"))

    return written
