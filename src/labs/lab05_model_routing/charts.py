"""Chart generation for Lab05's model-routing experiment.

Reads only `experiment_summary.json` from a completed run directory (same
convention as every other redesigned lab), and renders PNGs into a
`charts/` subdirectory next to it. Same data-integrity rules as those
labs: a strategy with 0 successful calls is never drawn as a zero-height
bar ("Veri yok" instead); a routing-rule distribution that's empty (fixed
baselines never match a rule) is shown honestly, never fabricated.
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

STRATEGY_COLOR = {
    "always_small": CATEGORICAL_DARK[0],
    "always_large": CATEGORICAL_DARK[1],
    "rule_based_routing": CATEGORICAL_DARK[2],
}
STRATEGY_LABEL = {
    "always_small": "Always-Small",
    "always_large": "Always-Large",
    "rule_based_routing": "Rule-Based Routing",
}
STRATEGY_ORDER = ("always_small", "always_large", "rule_based_routing")

QUALITY_OUTCOME_COLOR = {
    "QUALITY_PRESERVED": CATEGORICAL_DARK[2],
    "ROUTING_QUALITY_IMPROVEMENT": CATEGORICAL_DARK[5],
    "ROUTING_QUALITY_REGRESSION": CATEGORICAL_DARK[7],
    "BOTH_FAILED": MUTED_INK,
    "INSUFFICIENT_DATA": SECONDARY_INK,
}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_strategies(strategies: list[dict]) -> list[str]:
    present = {s["strategy"] for s in strategies}
    ordered = [s for s in STRATEGY_ORDER if s in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _usable(strategies_by_id: dict[str, dict], order: list[str]) -> list[str]:
    return [s for s in order if strategies_by_id.get(s, {}).get("total_tasks", 0) > 0]


def _draw_model_selection_distribution(ax, strategies_by_id: dict, order: list[str]) -> list[int]:
    usable = _usable(strategies_by_id, order)
    if not usable:
        no_data_text(ax)
        ax.set_title("Model Selection Distribution", fontsize=10)
        return []
    x = list(range(len(usable)))
    small = [strategies_by_id[s]["small_model_calls"] for s in usable]
    large = [strategies_by_id[s]["large_model_calls"] for s in usable]
    ax.bar(x, small, width=0.5, label="Small Model", color=CATEGORICAL_DARK[0])
    ax.bar(x, large, width=0.5, bottom=small, label="Large Model", color=CATEGORICAL_DARK[1])
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in usable], fontsize=8)
    ax.set_ylabel("Call Count")
    ax.set_title("Model Selection Distribution", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [strategies_by_id[s]["total_tasks"] for s in usable]


def _draw_accuracy_by_strategy(ax, strategies_by_id: dict, order: list[str]) -> list[int]:
    usable = [s for s in _usable(strategies_by_id, order) if strategies_by_id[s].get("accuracy") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Accuracy by Strategy", fontsize=10)
        return []
    values = [strategies_by_id[s]["accuracy"] * 100 for s in usable]
    colors = [STRATEGY_COLOR.get(s, SECONDARY_INK) for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([
        f"{STRATEGY_LABEL.get(s, s)}\n({strategies_by_id[s]['correct_tasks']}/{strategies_by_id[s]['total_tasks']})"
        for s in usable
    ], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Strategy", fontsize=10)
    return [strategies_by_id[s]["total_tasks"] for s in usable]


def _draw_cost_by_strategy(ax, strategies_by_id: dict, order: list[str]) -> list[int]:
    usable = [s for s in _usable(strategies_by_id, order) if strategies_by_id[s].get("total_estimated_cost_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Cost by Strategy", fontsize=10)
        return []
    values = [strategies_by_id[s]["total_estimated_cost_usd"] for s in usable]
    colors = [STRATEGY_COLOR.get(s, SECONDARY_INK) for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in usable], fontsize=8)
    ax.set_ylabel("Total Estimated Cost (USD)")
    ax.set_title("Cost by Strategy", fontsize=10)
    return [strategies_by_id[s]["total_tasks"] for s in usable]


def _draw_latency_by_strategy(ax, strategies_by_id: dict, order: list[str]) -> list[int]:
    usable = [s for s in _usable(strategies_by_id, order) if strategies_by_id[s].get("latency_ms", {}).get("mean") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Latency by Strategy", fontsize=10)
        return []
    means = [strategies_by_id[s]["latency_ms"]["mean"] for s in usable]
    errors = [strategies_by_id[s]["latency_ms"].get("sample_standard_deviation") or 0.0 for s in usable]
    colors = [STRATEGY_COLOR.get(s, SECONDARY_INK) for s in usable]
    ax.bar(range(len(usable)), means, yerr=errors, capsize=4, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in usable], fontsize=8)
    ax.set_ylabel("Mean Client Latency (ms)")
    ax.set_title("Latency by Strategy", fontsize=10)
    return [strategies_by_id[s]["total_tasks"] for s in usable]


def _draw_cost_per_task(ax, comparisons: list[dict]) -> list[int]:
    usable = [c for c in comparisons if c.get("baseline_cost") is not None and c.get("routed_cost") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Cost per Task", fontsize=10)
        return []
    x = list(range(len(usable)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [c["baseline_cost"] for c in usable], width=width, label="Baseline", color=CATEGORICAL_DARK[0])
    ax.bar([i + width / 2 for i in x], [c["routed_cost"] for c in usable], width=width, label="Routed", color=CATEGORICAL_DARK[2])
    ax.set_xticks(x)
    ax.set_xticklabels([c["task_id"] for c in usable], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("Cost per Task", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [1 for _ in usable]


def _draw_quality_regression_matrix(ax, comparisons: list[dict]) -> list[int]:
    if not comparisons:
        no_data_text(ax)
        ax.set_title("Quality Regression Matrix", fontsize=10)
        return []
    outcomes_present = sorted({c["quality_outcome"] for c in comparisons})
    for outcome in outcomes_present:
        points = [c for c in comparisons if c["quality_outcome"] == outcome]
        xs = [comparisons.index(p) for p in points]
        ys = [1] * len(points)
        ax.scatter(xs, ys, c=QUALITY_OUTCOME_COLOR.get(outcome, SECONDARY_INK), s=120, label=outcome, marker="s")
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels([c["task_id"] for c in comparisons], fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("Quality Regression Matrix", fontsize=10)
    ax.legend(fontsize=6, loc="upper center", ncols=2)
    return [1 for _ in comparisons]


def _draw_token_consumption_breakdown(ax, strategies_by_id: dict, order: list[str], records: list[dict]) -> list[int]:
    usable = _usable(strategies_by_id, order)
    if not usable:
        no_data_text(ax)
        ax.set_title("Token Consumption Breakdown", fontsize=10)
        return []
    x = list(range(len(usable)))
    prompt_means, reasoning_means, non_reasoning_means = [], [], []
    for s in usable:
        strategy_records = [r for r in records if r.get("strategy") == s and r.get("succeeded")]
        prompt_vals = [r["prompt_tokens"] for r in strategy_records if r.get("prompt_tokens") is not None]
        reasoning_vals = [r["reasoning_tokens"] for r in strategy_records if r.get("reasoning_tokens") is not None]
        non_reasoning_vals = [r["non_reasoning_completion_tokens"] for r in strategy_records if r.get("non_reasoning_completion_tokens") is not None]
        prompt_means.append(sum(prompt_vals) / len(prompt_vals) if prompt_vals else 0)
        reasoning_means.append(sum(reasoning_vals) / len(reasoning_vals) if reasoning_vals else 0)
        non_reasoning_means.append(sum(non_reasoning_vals) / len(non_reasoning_vals) if non_reasoning_vals else 0)
    ax.bar(x, prompt_means, width=0.5, label="Prompt", color=CATEGORICAL_DARK[0])
    ax.bar(x, reasoning_means, width=0.5, bottom=prompt_means, label="Reasoning", color=CATEGORICAL_DARK[4])
    bottom2 = [p + r for p, r in zip(prompt_means, reasoning_means)]
    ax.bar(x, non_reasoning_means, width=0.5, bottom=bottom2, label="Non-Reasoning Completion", color=CATEGORICAL_DARK[2])
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in usable], fontsize=8)
    ax.set_ylabel("Mean Token Count")
    ax.set_title("Token Consumption Breakdown", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [strategies_by_id[s]["total_tasks"] for s in usable]


def _draw_routing_decision_analysis(ax, strategies_by_id: dict) -> list[int]:
    router = strategies_by_id.get("rule_based_routing")
    rule_distribution = (router or {}).get("routing_rule_distribution", {})
    if not router or not rule_distribution:
        no_data_text(ax, "Veri yok: router hiçbir kuralı eşlemedi (tüm görevler small modele gitti)" if router else "Veri yok")
        ax.set_title("Routing Decision Analysis", fontsize=10)
        return [router["total_tasks"]] if router else []
    rules = sorted(rule_distribution.keys())
    values = [rule_distribution[r] for r in rules]
    ax.barh(range(len(rules)), values, color=CATEGORICAL_DARK[2], height=0.5)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=7)
    ax.set_xlabel("Tasks Routed to Large Model via This Rule")
    ax.set_title("Routing Decision Analysis", fontsize=10)
    return [router["total_tasks"]]


def generate_routing_charts(run_dir: Path) -> list[Path]:
    """Render every Lab05 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    strategies = summary.get("strategies", [])
    strategies_with_data = [s for s in strategies if s.get("total_tasks", 0) > 0]
    if not strategies_with_data:
        return []

    order = _ordered_strategies(strategies)
    strategies_by_id = {s["strategy"]: s for s in strategies}
    comparisons = summary.get("task_comparisons", [])
    records = summary.get("measurements", [])
    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. Model Selection Distribution.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_model_selection_distribution(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "model_selection_distribution.png"))

    # 2. Accuracy by Strategy.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_accuracy_by_strategy(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "accuracy_by_strategy.png"))

    # 3. Cost by Strategy.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cost_by_strategy(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cost_by_strategy.png"))

    # 4. Latency by Strategy.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_latency_by_strategy(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "latency_by_strategy.png"))

    # 5. Cost per Task.
    fig, ax = new_figure(figsize=(9, 5))
    counts = _draw_cost_per_task(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cost_per_task.png"))

    # 6. Quality Regression Matrix.
    fig, ax = new_figure(figsize=(9, 3.5))
    counts = _draw_quality_regression_matrix(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "quality_regression_matrix.png"))

    # 7. Token Consumption Breakdown.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_token_consumption_breakdown(ax, strategies_by_id, order, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "token_consumption_breakdown.png"))

    # 8. Routing Decision Analysis.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_routing_decision_analysis(ax, strategies_by_id)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "routing_decision_analysis.png"))

    # 9. Quality-Cost Trade-off Dashboard.
    fig, axes = new_figure(figsize=(20, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_accuracy_by_strategy(axes[0][0], strategies_by_id, order))
    all_counts.extend(_draw_cost_by_strategy(axes[0][1], strategies_by_id, order))
    all_counts.extend(_draw_latency_by_strategy(axes[0][2], strategies_by_id, order))
    all_counts.extend(_draw_model_selection_distribution(axes[0][3], strategies_by_id, order))
    all_counts.extend(_draw_cost_per_task(axes[1][0], comparisons))
    all_counts.extend(_draw_quality_regression_matrix(axes[1][1], comparisons))
    all_counts.extend(_draw_token_consumption_breakdown(axes[1][2], strategies_by_id, order, records))
    all_counts.extend(_draw_routing_decision_analysis(axes[1][3], strategies_by_id))
    fig.suptitle(
        f"LAB05 Model Routing Summary -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "quality_cost_tradeoff_dashboard.png"))

    return written
