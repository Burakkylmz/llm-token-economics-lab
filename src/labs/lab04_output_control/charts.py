"""Chart generation for Lab04's output-control and invocation-optimization
experiment.

Reads only `experiment_summary.json` from a completed run directory (same
convention as labs.lab02_context_growth.charts/labs.lab03_prompt_caching.charts),
and renders PNGs into a `charts/` subdirectory next to it. Same data-integrity
rules as those labs: a condition with 0 successful calls is never drawn as a
zero-height bar ("Veri yok" instead); budget values are never confused with
observed consumption; a missing invocation_comparison section (e.g. a
output_limit_sweep-only run) simply skips those charts rather than fabricating
empty ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import (
    CATEGORICAL_DARK,
    MUTED_INK,
    SECONDARY_INK,
    add_footer_caption,
    new_figure,
    no_data_text,
    sample_size_caption,
    save_png,
)

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"

FINISH_REASON_COLOR = {
    "stop": CATEGORICAL_DARK[2],
    "length": CATEGORICAL_DARK[7],
    "unavailable": MUTED_INK,
}

STRATEGY_COLOR = {"single": CATEGORICAL_DARK[0], "multiple": CATEGORICAL_DARK[1]}
STRATEGY_LABEL = {"single": "Single Invocation", "multiple": "Multiple Invocations"}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_conditions(conditions: list[dict]) -> list[dict]:
    """Output-limit conditions sorted by their token budget, ascending --
    the natural reading order for every "vs budget" chart, regardless of
    the order calls actually executed in."""
    return sorted(conditions, key=lambda c: c.get("max_completion_tokens", 0))


def _usable_conditions(conditions: list[dict]) -> list[dict]:
    return [c for c in conditions if c.get("successful_calls", 0) > 0]


def _draw_completion_tokens_vs_budget(ax, conditions: list[dict]) -> list[int]:
    usable = _usable_conditions(conditions)
    if not usable:
        no_data_text(ax)
        ax.set_title("Completion Tokens vs Token Budget", fontsize=10)
        return []
    budgets = [c["max_completion_tokens"] for c in usable]
    means = [c["completion_tokens"].get("mean") for c in usable]
    ax.plot(budgets, budgets, color=MUTED_INK, linestyle="--", linewidth=1, label="Budget (max_completion_tokens)")
    ax.scatter(budgets, means, color=CATEGORICAL_DARK[0], s=60, zorder=3, label="Observed (mean)")
    ax.plot(budgets, means, color=CATEGORICAL_DARK[0], linewidth=1, zorder=2)
    ax.set_xlabel("max_completion_tokens")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Completion Tokens vs Token Budget", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [c.get("successful_calls", 0) for c in usable]


def _draw_reasoning_vs_non_reasoning(ax, conditions: list[dict]) -> list[int]:
    usable = _usable_conditions(conditions)
    if not usable:
        no_data_text(ax)
        ax.set_title("Reasoning vs Non-Reasoning Tokens", fontsize=10)
        return []
    x = list(range(len(usable)))
    reasoning = [c["reasoning_tokens"].get("mean") or 0 for c in usable]
    non_reasoning = [c["non_reasoning_completion_tokens"].get("mean") or 0 for c in usable]
    ax.bar(x, reasoning, width=0.5, label="Reasoning", color=CATEGORICAL_DARK[4])
    ax.bar(x, non_reasoning, width=0.5, bottom=reasoning, label="Non-Reasoning", color=CATEGORICAL_DARK[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['condition_id']}\n({c['max_completion_tokens']})" for c in usable], fontsize=8)
    ax.set_ylabel("Mean Completion Tokens")
    ax.set_title("Reasoning vs Non-Reasoning Tokens", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [c.get("successful_calls", 0) for c in usable]


def _finish_reason_tally(records: list[dict], condition_id: str) -> dict[str, int]:
    tally: dict[str, int] = {}
    for r in records:
        if r.get("condition_id") != condition_id or not r.get("succeeded"):
            continue
        reason = r.get("finish_reason") or "unavailable"
        tally[reason] = tally.get(reason, 0) + 1
    return tally


def _draw_finish_reason_distribution(ax, conditions: list[dict], records: list[dict]) -> list[int]:
    usable = _usable_conditions(conditions)
    if not usable:
        no_data_text(ax)
        ax.set_title("Finish Reason Distribution", fontsize=10)
        return []
    x = list(range(len(usable)))
    reasons = sorted({reason for c in usable for reason in _finish_reason_tally(records, c["condition_id"])})
    bottom = [0] * len(usable)
    for reason in reasons:
        values = [_finish_reason_tally(records, c["condition_id"]).get(reason, 0) for c in usable]
        ax.bar(x, values, width=0.5, bottom=bottom, label=reason, color=FINISH_REASON_COLOR.get(reason, SECONDARY_INK))
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(x)
    ax.set_xticklabels([c["condition_id"] for c in usable], fontsize=8)
    ax.set_ylabel("Call Count")
    ax.set_title("Finish Reason Distribution", fontsize=10)
    if reasons:
        ax.legend(fontsize=7, loc="upper right")
    return [c.get("successful_calls", 0) for c in usable]


def _draw_completeness_vs_budget(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable_conditions(conditions) if c.get("mean_field_completeness_ratio") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Answer Completeness vs Token Budget", fontsize=10)
        return []
    budgets = [c["max_completion_tokens"] for c in usable]
    completeness = [c["mean_field_completeness_ratio"] * 100 for c in usable]
    correctness = [
        (c.get("mean_field_correctness_ratio") or 0) * 100 for c in usable
    ]
    ax.plot(budgets, completeness, color=CATEGORICAL_DARK[2], marker="o", label="Completeness %")
    ax.plot(budgets, correctness, color=CATEGORICAL_DARK[0], marker="s", label="Correctness %")
    ax.set_ylim(0, 105)
    ax.set_xlabel("max_completion_tokens")
    ax.set_ylabel("%")
    ax.set_title("Answer Completeness vs Token Budget", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    return [c.get("successful_calls", 0) for c in usable]


def _draw_cost_vs_budget(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable_conditions(conditions) if c.get("mean_estimated_cost_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Output Limit vs Estimated Cost", fontsize=10)
        return []
    budgets = [c["max_completion_tokens"] for c in usable]
    costs = [c["mean_estimated_cost_usd"] for c in usable]
    ax.plot(budgets, costs, color=CATEGORICAL_DARK[3], marker="o")
    ax.set_xlabel("max_completion_tokens (budget)")
    ax.set_ylabel("Mean Estimated Cost (USD)")
    ax.set_title("Output Limit vs Estimated Cost", fontsize=10)
    return [c.get("successful_calls", 0) for c in usable]


def _invocation_usable(comparison: Optional[dict]) -> bool:
    return bool(comparison) and comparison.get("single", {}).get("api_call_count", 0) > 0 and comparison.get("multiple", {}).get("api_call_count", 0) > 0


def _draw_invocation_token_breakdown(ax, comparison: Optional[dict]) -> list[int]:
    if not _invocation_usable(comparison):
        no_data_text(ax)
        ax.set_title("Single vs Multiple Invocation Token Breakdown", fontsize=10)
        return []
    strategies = ["single", "multiple"]
    x = list(range(len(strategies)))
    prompt = [comparison[s]["total_prompt_tokens"] for s in strategies]
    reasoning = [comparison[s]["total_reasoning_tokens"] for s in strategies]
    non_reasoning = [comparison[s]["total_completion_tokens"] - comparison[s]["total_reasoning_tokens"] for s in strategies]
    ax.bar(x, prompt, width=0.5, label="Prompt", color=CATEGORICAL_DARK[0])
    ax.bar(x, reasoning, width=0.5, bottom=prompt, label="Reasoning", color=CATEGORICAL_DARK[4])
    bottom2 = [p + r for p, r in zip(prompt, reasoning)]
    ax.bar(x, non_reasoning, width=0.5, bottom=bottom2, label="Non-Reasoning Completion", color=CATEGORICAL_DARK[2])
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in strategies])
    ax.set_ylabel("Token Count")
    ax.set_title("Single vs Multiple Invocation Token Breakdown", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [comparison[s]["api_call_count"] for s in strategies]


def _draw_invocation_cost_comparison(ax, comparison: Optional[dict]) -> list[int]:
    if not _invocation_usable(comparison) or comparison["single"].get("total_estimated_cost_usd") is None:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Invocation Cost Comparison", fontsize=10)
        return []
    strategies = ["single", "multiple"]
    values = [comparison[s]["total_estimated_cost_usd"] or 0 for s in strategies]
    colors = [STRATEGY_COLOR[s] for s in strategies]
    ax.bar(range(len(strategies)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in strategies])
    ax.set_ylabel("Total Estimated Cost (USD)")
    ax.set_title("Invocation Cost Comparison", fontsize=10)
    return [comparison[s]["api_call_count"] for s in strategies]


def _draw_workflow_latency(ax, comparison: Optional[dict]) -> list[int]:
    if not _invocation_usable(comparison) or comparison["single"].get("workflow_wall_clock_ms") is None:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Workflow Wall-Clock Latency", fontsize=10)
        return []
    strategies = ["single", "multiple"]
    values = [comparison[s]["workflow_wall_clock_ms"] or 0 for s in strategies]
    colors = [STRATEGY_COLOR[s] for s in strategies]
    ax.bar(range(len(strategies)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in strategies])
    ax.set_ylabel("Workflow Wall-Clock (ms)")
    ax.set_title("Workflow Wall-Clock Latency", fontsize=10)
    return [comparison[s]["api_call_count"] for s in strategies]


def _draw_quality_optimization_panel(ax, conditions: list[dict], comparison: Optional[dict]) -> list[int]:
    """Required chart #9's core panel: answer acceptance, cost, and
    truncation together per output-limit condition, in one compact view --
    the full multi-signal dashboard is the figure this panel is embedded
    in (summary_dashboard.png), per this lab's design."""
    usable = _usable_conditions(conditions)
    if not usable:
        no_data_text(ax)
        ax.set_title("Quality-Constrained Optimization", fontsize=10)
        return []
    x = list(range(len(usable)))
    accepted_rate = [
        (c["accepted_count"] / c["successful_calls"]) * 100 if c.get("successful_calls") else 0 for c in usable
    ]
    truncation_rate = [(c.get("truncation_rate") or 0) * 100 for c in usable]
    ax.bar([i - 0.2 for i in x], accepted_rate, width=0.4, label="Accepted %", color=CATEGORICAL_DARK[2])
    ax.bar([i + 0.2 for i in x], truncation_rate, width=0.4, label="Truncated %", color=CATEGORICAL_DARK[7])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['condition_id']}\n({c['max_completion_tokens']})" for c in usable], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Quality-Constrained Optimization", fontsize=10)
    ax.legend(fontsize=7, loc="upper center")
    return [c.get("successful_calls", 0) for c in usable]


def generate_output_control_charts(run_dir: Path) -> list[Path]:
    """Render every Lab04 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    conditions = _ordered_conditions(summary.get("output_limit_conditions", []))
    records = summary.get("output_limit_measurements", [])
    comparison = summary.get("invocation_comparison")

    has_output_limit_data = bool(_usable_conditions(conditions))
    has_invocation_data = _invocation_usable(comparison)
    if not has_output_limit_data and not has_invocation_data:
        return []

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. Completion Tokens vs Token Budget.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_completion_tokens_vs_budget(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "completion_tokens_vs_budget.png"))

    # 2. Reasoning vs Non-Reasoning Tokens.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_reasoning_vs_non_reasoning(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "reasoning_vs_non_reasoning_tokens.png"))

    # 3. Finish Reason Distribution.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_finish_reason_distribution(ax, conditions, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "finish_reason_distribution.png"))

    # 4. Answer Completeness vs Token Budget.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_completeness_vs_budget(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "answer_completeness_vs_budget.png"))

    # 5. Output Limit vs Estimated Cost.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cost_vs_budget(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "output_limit_vs_estimated_cost.png"))

    # 6. Single vs Multiple Invocation Token Breakdown.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_invocation_token_breakdown(ax, comparison)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "invocation_token_breakdown.png"))

    # 7. Invocation Cost Comparison.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_invocation_cost_comparison(ax, comparison)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "invocation_cost_comparison.png"))

    # 8. Workflow Wall-Clock Latency.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_workflow_latency(ax, comparison)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "workflow_wall_clock_latency.png"))

    # 9. Quality-Constrained Optimization Dashboard.
    fig, axes = new_figure(figsize=(20, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_quality_optimization_panel(axes[0][0], conditions, comparison))
    all_counts.extend(_draw_completion_tokens_vs_budget(axes[0][1], conditions))
    all_counts.extend(_draw_cost_vs_budget(axes[0][2], conditions))
    all_counts.extend(_draw_finish_reason_distribution(axes[0][3], conditions, records))
    all_counts.extend(_draw_reasoning_vs_non_reasoning(axes[1][0], conditions))
    all_counts.extend(_draw_invocation_token_breakdown(axes[1][1], comparison))
    all_counts.extend(_draw_invocation_cost_comparison(axes[1][2], comparison))
    all_counts.extend(_draw_workflow_latency(axes[1][3], comparison))
    fig.suptitle(
        f"LAB04 Output Control Summary -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "quality_constrained_optimization_dashboard.png"))

    return written
