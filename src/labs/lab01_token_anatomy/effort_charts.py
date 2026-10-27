"""Chart generation for Lab01's `effort` mode (reasoning_effort comparison).

Reads only `effort_summary.json` (already-computed, measurement-derived
descriptive statistics -- see labs.lab01_token_anatomy.effort_stats) from a
completed run directory and renders PNGs into a `charts/` subdirectory next
to it. Never reads or writes `raw_measurements.*`, `experiment_plan.csv`,
`experiment_manifest.json`, or `analysis_report.md` -- those stay exactly as
the experiment runner wrote them.

Design rules enforced throughout (per this project's data-integrity policy):
  - A condition with 0 successful calls is never drawn as a zero-height bar;
    it is omitted from the bars and named in a "no data" note instead.
  - A condition with exactly 1 successful call never gets an error bar, and
    the figure's footer explicitly disclaims statistical significance.
  - Multiple prompt_types are never averaged together into one bar; each
    gets its own facet (column) in every chart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from lab_core.charts import (
    PROMPT_TYPE_COLOR,
    REASONING_EFFORT_COLOR,
    REASONING_EFFORT_ORDER,
    SECONDARY_INK,
    new_figure,
    no_data_text,
    sample_size_caption,
    save_png,
)

EFFORT_SUMMARY_FILENAME = "effort_summary.json"
CHARTS_DIRNAME = "charts"


def load_effort_summary(run_dir: Path) -> Optional[dict]:
    path = run_dir / EFFORT_SUMMARY_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_efforts_present(conditions: list[dict]) -> list[str]:
    present = {c["reasoning_effort"] for c in conditions}
    ordered = [e for e in REASONING_EFFORT_ORDER if e in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _ordered_prompt_types_present(conditions: list[dict]) -> list[str]:
    present = {c["prompt_type"] for c in conditions}
    ordered = [p for p in ("short_prompt", "long_prompt") if p in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _conditions_by_prompt_and_effort(conditions: list[dict]) -> dict[str, dict[str, dict]]:
    by_prompt: dict[str, dict[str, dict]] = {}
    for c in conditions:
        by_prompt.setdefault(c["prompt_type"], {})[c["reasoning_effort"]] = c
    return by_prompt


def _metric_mean_and_err(cond: dict, field: str) -> tuple[Optional[float], Optional[float]]:
    value = cond.get(field)
    if isinstance(value, dict):
        return value.get("mean"), value.get("sample_standard_deviation")
    return value, None


def _usable_efforts(cond_by_effort: dict[str, dict], efforts: list[str], field: str) -> list[str]:
    """Efforts that have at least one successful call AND a non-missing value
    for `field` -- never plotted as a fabricated zero."""
    usable = []
    for effort in efforts:
        cond = cond_by_effort.get(effort)
        if cond is None or cond.get("successful_calls", 0) == 0:
            continue
        value, _ = _metric_mean_and_err(cond, field)
        if value is not None:
            usable.append(effort)
    return usable


def _draw_bar_panel(
    ax,
    cond_by_effort: dict[str, dict],
    efforts: list[str],
    field: str,
    title: str,
    ylabel: str,
    value_transform: Optional[Callable[[float], float]] = None,
) -> list[int]:
    usable = _usable_efforts(cond_by_effort, efforts, field)
    if not usable:
        no_data_text(ax)
        ax.set_title(title, fontsize=10)
        return []

    values, errors, colors = [], [], []
    for effort in usable:
        mean, err = _metric_mean_and_err(cond_by_effort[effort], field)
        if value_transform is not None:
            mean = value_transform(mean)
            err = value_transform(err) if err is not None else None
        values.append(mean)
        errors.append(err if err is not None else 0.0)
        colors.append(REASONING_EFFORT_COLOR.get(effort, SECONDARY_INK))

    x = range(len(usable))
    ax.bar(x, values, yerr=errors, color=colors, capsize=4, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(usable)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)

    missing = [e for e in efforts if e not in usable]
    if missing:
        ax.text(
            0.5, -0.22, f"Veri yok: {', '.join(missing)}", transform=ax.transAxes,
            ha="center", va="top", color=SECONDARY_INK, fontsize=7, style="italic",
        )
    return [cond_by_effort[e].get("successful_calls", 0) for e in usable]


def _draw_reasoning_vs_non_reasoning_panel(ax, cond_by_effort: dict[str, dict], efforts: list[str]) -> list[int]:
    usable_r = _usable_efforts(cond_by_effort, efforts, "reasoning_tokens")
    usable_n = _usable_efforts(cond_by_effort, efforts, "non_reasoning_completion_tokens")
    usable = [e for e in efforts if e in usable_r and e in usable_n]
    if not usable:
        no_data_text(ax)
        ax.set_title("Reasoning vs Non-Reasoning Tokens", fontsize=10)
        return []

    reasoning_vals, reasoning_err, non_reasoning_vals, non_reasoning_err = [], [], [], []
    for effort in usable:
        r_mean, r_err = _metric_mean_and_err(cond_by_effort[effort], "reasoning_tokens")
        n_mean, n_err = _metric_mean_and_err(cond_by_effort[effort], "non_reasoning_completion_tokens")
        reasoning_vals.append(r_mean)
        reasoning_err.append(r_err or 0.0)
        non_reasoning_vals.append(n_mean)
        non_reasoning_err.append(n_err or 0.0)

    x = list(range(len(usable)))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], reasoning_vals, width=width, yerr=reasoning_err, capsize=3,
        color=[REASONING_EFFORT_COLOR.get(e) for e in usable], label="Reasoning",
    )
    ax.bar(
        [i + width / 2 for i in x], non_reasoning_vals, width=width, yerr=non_reasoning_err, capsize=3,
        color=[REASONING_EFFORT_COLOR.get(e) for e in usable], alpha=0.45, label="Non-reasoning",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(usable)
    ax.set_ylabel("Token Sayısı")
    ax.set_title("Reasoning vs Non-Reasoning Completion Tokens", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [cond_by_effort[e].get("successful_calls", 0) for e in usable]


# (title, ylabel, field, value_transform) for the four single-metric charts.
_METRIC_CHARTS: list[tuple[str, str, str, str, Optional[Callable[[float], float]]]] = [
    ("completion_tokens_by_effort.png", "Completion Tokens by Reasoning Effort", "Token Sayısı", "completion_tokens", None),
    ("latency_by_effort.png", "Latency by Reasoning Effort", "Gecikme (ms)", "latency_ms", None),
    ("estimated_cost_by_effort.png", "Estimated Cost by Reasoning Effort", "Tahmini Maliyet (USD)", "mean_estimated_cost_usd", None),
    ("reasoning_ratio_by_effort.png", "Reasoning Ratio by Effort", "Reasoning Oranı (%)", "mean_reasoning_ratio", lambda v: v * 100),
]


def _facet_figure(prompt_types: list[str], draw_one_panel) -> tuple:
    fig, axes = new_figure(figsize=(5.5 * len(prompt_types), 4.8), ncols=len(prompt_types))
    if len(prompt_types) == 1:
        axes = [axes]
    all_counts: list[int] = []
    for ax, prompt_type in zip(axes, prompt_types):
        counts = draw_one_panel(ax, prompt_type)
        all_counts.extend(counts)
        ax.set_xlabel(prompt_type, color=PROMPT_TYPE_COLOR.get(prompt_type, SECONDARY_INK), fontsize=9)
    return fig, all_counts


def generate_effort_charts(run_dir: Path) -> list[Path]:
    """Render every effort-mode chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data -- never raises on sparse/empty results)."""
    summary = load_effort_summary(run_dir)
    if not summary:
        return []
    conditions = [c for c in summary.get("conditions", []) if c.get("successful_calls", 0) > 0]
    if not conditions:
        return []

    by_prompt = _conditions_by_prompt_and_effort(summary.get("conditions", []))
    prompt_types = _ordered_prompt_types_present(conditions)
    efforts = _ordered_efforts_present(conditions)
    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. Reasoning vs non-reasoning tokens.
    fig, counts = _facet_figure(
        prompt_types,
        lambda ax, p: _draw_reasoning_vs_non_reasoning_panel(ax, by_prompt.get(p, {}), efforts),
    )
    from lab_core.charts import add_footer_caption

    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "reasoning_vs_non_reasoning_tokens.png"))

    # 2-5. The four single-metric comparison charts.
    for filename, title, ylabel, field, transform in _METRIC_CHARTS:
        fig, counts = _facet_figure(
            prompt_types,
            lambda ax, p, field=field, title=title, ylabel=ylabel, transform=transform: _draw_bar_panel(
                ax, by_prompt.get(p, {}), efforts, field, title, ylabel, transform
            ),
        )
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / filename))

    # 6. Summary dashboard: every panel above, one row per prompt_type.
    nrows = len(prompt_types)
    fig, axes = new_figure(figsize=(26, 4.6 * nrows), nrows=nrows, ncols=5)
    if nrows == 1:
        axes = [axes]
    all_counts: list[int] = []
    panel_specs = [("Reasoning vs Non-Reasoning", None)] + [(title, (field, transform)) for _, title, _, field, transform in _METRIC_CHARTS]
    for row, prompt_type in zip(axes, prompt_types):
        for col_ax, (title, spec) in zip(row, panel_specs):
            if spec is None:
                counts = _draw_reasoning_vs_non_reasoning_panel(col_ax, by_prompt.get(prompt_type, {}), efforts)
            else:
                field, transform = spec
                ylabel = next(yl for _, t, yl, f, _ in _METRIC_CHARTS if f == field)
                counts = _draw_bar_panel(col_ax, by_prompt.get(prompt_type, {}), efforts, field, title, ylabel, transform)
            all_counts.extend(counts)
    fig.suptitle(
        f"LAB01 Effort Summary -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "summary_dashboard.png"))

    return written
