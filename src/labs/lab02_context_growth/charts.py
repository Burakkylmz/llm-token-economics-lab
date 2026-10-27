"""Chart generation for Lab02's context-growth/context-bloat experiment.

Reads only already-computed, measurement-derived files from a completed
run directory -- `experiment_summary.json` (condition-level descriptive
statistics) and `multi_turn_measurements.json` (the turn-by-turn growth
sequence) -- and renders PNGs into a `charts/` subdirectory next to them.
Never reads or writes `context_measurements.*`, `experiment_plan.csv`,
`experiment_manifest.json`, or `analysis_report.md`.

Same data-integrity rules as labs.lab01_token_anatomy.effort_charts: a
condition/turn with 0 successful calls is never drawn as a zero-height bar
(shown as "Veri yok" instead); n=1 groups never get an error bar and the
figure's footer explicitly disclaims statistical significance; the three
context conditions are never averaged together.
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
from labs.lab02_context_growth.experiments import CONTEXT_CONDITION_ORDER

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
MULTI_TURN_JSON_FILENAME = "multi_turn_measurements.json"
CHARTS_DIRNAME = "charts"

CONTEXT_CONDITION_COLOR = {
    "minimal_context": CATEGORICAL_DARK[0],
    "expanded_context": CATEGORICAL_DARK[1],
    "irrelevant_context": CATEGORICAL_DARK[2],
}
CONTEXT_CONDITION_LABEL = {
    "minimal_context": "Minimal",
    "expanded_context": "Expanded",
    "irrelevant_context": "Irrelevant",
}


def _load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_conditions_present(conditions: list[dict]) -> list[str]:
    present = {c["condition_id"] for c in conditions}
    ordered = [c for c in CONTEXT_CONDITION_ORDER if c in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _usable(conditions_by_id: dict[str, dict], order: list[str], field: str) -> list[str]:
    usable = []
    for cid in order:
        cond = conditions_by_id.get(cid)
        if cond is None or cond.get("successful_calls", 0) == 0:
            continue
        value = cond.get(field)
        mean = value.get("mean") if isinstance(value, dict) else value
        if mean is not None:
            usable.append(cid)
    return usable


def _mean_and_err(cond: dict, field: str) -> tuple[Optional[float], Optional[float]]:
    value = cond.get(field)
    if isinstance(value, dict):
        return value.get("mean"), value.get("sample_standard_deviation")
    return value, None


def _draw_single_metric_bar(ax, conditions_by_id, order, field, title, ylabel) -> list[int]:
    usable = _usable(conditions_by_id, order, field)
    if not usable:
        no_data_text(ax)
        ax.set_title(title, fontsize=10)
        return []
    values, errors, colors, labels = [], [], [], []
    for cid in usable:
        mean, err = _mean_and_err(conditions_by_id[cid], field)
        values.append(mean)
        errors.append(err if err is not None else 0.0)
        colors.append(CONTEXT_CONDITION_COLOR.get(cid, SECONDARY_INK))
        labels.append(CONTEXT_CONDITION_LABEL.get(cid, cid))
    x = range(len(usable))
    ax.bar(x, values, yerr=errors, color=colors, capsize=4, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    missing = [cid for cid in order if cid not in usable]
    if missing:
        ax.text(
            0.5, -0.22, f"Veri yok: {', '.join(missing)}", transform=ax.transAxes,
            ha="center", va="top", color=SECONDARY_INK, fontsize=7, style="italic",
        )
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_context_size_vs_latency(ax_tokens, ax_latency, conditions_by_id, order) -> list[int]:
    counts: list[int] = []
    counts.extend(_draw_single_metric_bar(ax_tokens, conditions_by_id, order, "prompt_tokens", "Context Size (Prompt Tokens)", "Token Sayısı"))
    counts.extend(_draw_single_metric_bar(ax_latency, conditions_by_id, order, "latency_ms", "Mean Latency", "Gecikme (ms)"))
    return counts


def _draw_token_breakdown(ax, conditions_by_id, order) -> list[int]:
    usable = [
        cid for cid in order
        if cid in _usable(conditions_by_id, order, "prompt_tokens")
        and cid in _usable(conditions_by_id, order, "completion_tokens")
    ]
    if not usable:
        no_data_text(ax)
        ax.set_title("Token Breakdown by Context Condition", fontsize=10)
        return []
    width = 0.25
    x = list(range(len(usable)))
    fields_and_offsets = [("prompt_tokens", -width, "Prompt"), ("reasoning_tokens", 0, "Reasoning"), ("non_reasoning_completion_tokens", width, "Non-Reasoning")]
    for field, offset, label in fields_and_offsets:
        values = [(_mean_and_err(conditions_by_id[cid], field)[0] or 0) for cid in usable]
        ax.bar([i + offset for i in x], values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([CONTEXT_CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Token Sayısı")
    ax.set_title("Token Breakdown by Context Condition", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_answer_correctness(ax, conditions_by_id, order) -> list[int]:
    usable = [cid for cid in order if conditions_by_id.get(cid, {}).get("successful_calls", 0) > 0]
    if not usable:
        no_data_text(ax)
        ax.set_title("Answer Correctness by Context Condition", fontsize=10)
        return []
    statuses = [("correct_count", "Correct"), ("incorrect_count", "Incorrect"), ("format_invalid_count", "Format Invalid"), ("missing_response_count", "Missing")]
    width = 0.6
    x = list(range(len(usable)))
    bottom = [0.0] * len(usable)
    for field, label in statuses:
        values = [conditions_by_id[cid].get(field, 0) or 0 for cid in usable]
        if all(v == 0 for v in values):
            continue
        ax.bar(x, values, width=width, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(x)
    ax.set_xticklabels([CONTEXT_CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Çağrı Sayısı")
    ax.set_title("Answer Correctness by Context Condition", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_multi_turn_growth(ax, turns: list[dict]) -> list[int]:
    usable = [t for t in turns if t.get("prompt_tokens") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Multi-Turn Growth: Prompt Tokens per Turn", fontsize=10)
        return []
    xs = [t["turn_index"] for t in usable]
    ys = [t["prompt_tokens"] for t in usable]
    ax.plot(xs, ys, marker="o", color=CATEGORICAL_DARK[0])
    ax.set_xticks(xs)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Prompt Tokens")
    ax.set_title("Multi-Turn Growth: Prompt Tokens per Turn", fontsize=10)
    return [1 for _ in usable]


def _draw_cumulative_consumption(ax, turns: list[dict]) -> list[int]:
    usable = [t for t in turns if t.get("cumulative_prompt_tokens") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Cumulative Input Token Consumption", fontsize=10)
        return []
    xs = [t["turn_index"] for t in usable]
    ys = [t["cumulative_prompt_tokens"] for t in usable]
    ax.plot(xs, ys, marker="o", color=CATEGORICAL_DARK[3])
    ax.set_xticks(xs)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative Prompt Tokens")
    ax.set_title("Cumulative Input Token Consumption", fontsize=10)
    return [1 for _ in usable]


def generate_context_growth_charts(run_dir: Path) -> list[Path]:
    """Render every Lab02 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    turn_records = _load_json(run_dir / MULTI_TURN_JSON_FILENAME) or []
    turns_successful = [t for t in turn_records if t.get("succeeded")]

    conditions = (summary or {}).get("conditions", []) if summary else []
    conditions_with_data = [c for c in conditions if c.get("successful_calls", 0) > 0]

    if not conditions_with_data and not turns_successful:
        return []

    order = _ordered_conditions_present(conditions) if conditions else list(CONTEXT_CONDITION_ORDER)
    conditions_by_id = {c["condition_id"]: c for c in conditions}
    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. Context size vs latency (two subplots: tokens, latency).
    fig, axes = new_figure(figsize=(11, 4.8), ncols=2)
    counts = _draw_context_size_vs_latency(axes[0], axes[1], conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "context_size_vs_latency.png"))

    # 2. Token breakdown.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_token_breakdown(ax, conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "token_breakdown.png"))

    # 3. Reasoning tokens by context.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "reasoning_tokens", "Reasoning Tokens by Context", "Token Sayısı")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "reasoning_tokens_by_context.png"))

    # 4. Estimated cost by context.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "mean_estimated_cost_usd", "Estimated Cost by Context", "Tahmini Maliyet (USD)")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "estimated_cost_by_context.png"))

    # 5. Multi-turn growth.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_multi_turn_growth(ax, turn_records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "multi_turn_growth.png"))

    # 6. Cumulative token consumption.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cumulative_consumption(ax, turn_records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cumulative_token_consumption.png"))

    # 7. Answer correctness.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_answer_correctness(ax, conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "answer_correctness.png"))

    # 8. Summary dashboard.
    fig, axes = new_figure(figsize=(22, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_single_metric_bar(axes[0][0], conditions_by_id, order, "prompt_tokens", "Context Size", "Token Sayısı"))
    all_counts.extend(_draw_single_metric_bar(axes[0][1], conditions_by_id, order, "latency_ms", "Latency", "Gecikme (ms)"))
    all_counts.extend(_draw_token_breakdown(axes[0][2], conditions_by_id, order))
    all_counts.extend(_draw_single_metric_bar(axes[0][3], conditions_by_id, order, "mean_estimated_cost_usd", "Estimated Cost", "USD"))
    all_counts.extend(_draw_answer_correctness(axes[1][0], conditions_by_id, order))
    all_counts.extend(_draw_multi_turn_growth(axes[1][1], turn_records))
    all_counts.extend(_draw_cumulative_consumption(axes[1][2], turn_records))
    no_data_text(axes[1][3], "")
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB02 Context Growth Summary -- {(summary or {}).get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "summary_dashboard.png"))

    return written
