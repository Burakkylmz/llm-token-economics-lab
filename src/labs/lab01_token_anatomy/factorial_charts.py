"""Chart generation for Lab01's retired `factorial` mode's historical results.

Purely a backfill/visualization reader: `lab01 factorial` itself always
fails closed now (see labs.lab01_token_anatomy.cli._run_legacy_factorial_stub),
so this module is reached only via `lab01 backfill`, never from live
execution. It reads `factorial_summary.json` from an existing
`results/lab01_factorial/<run_id>/` directory and renders heatmaps
(temperature x top_p, one facet column per reasoning_effort) plus a
parameter-interaction line chart, into a `charts/` subdirectory next to it.

The factorial schema's nucleus-sampling field is named `top_p` (0 < top_p
<= 1) -- distinct from `top_k`, which this project's model/API never uses.
Every value drawn here is read directly from that field name; nothing is
assumed about which temperature/top_p grid cells exist -- only cells with at
least one successful measurement are plotted, everything else renders as an
explicit "no data" cell, never a fabricated zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from lab_core.charts import (
    CATEGORICAL_DARK,
    MUTED_INK,
    add_footer_caption,
    new_figure,
    no_data_text,
    sample_size_caption,
    save_png,
    sequential_blue_cmap,
)

FACTORIAL_SUMMARY_FILENAME = "factorial_summary.json"
CHARTS_DIRNAME = "charts"

REASONING_EFFORT_ORDER = ("low", "medium", "high")


def load_factorial_summary(run_dir: Path) -> Optional[dict]:
    path = run_dir / FACTORIAL_SUMMARY_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_present(values: set, fixed_order: tuple) -> list:
    ordered = [v for v in fixed_order if v in values]
    extra = sorted(values - set(ordered))
    return ordered + extra


def _metric_mean(cond: dict, field: str) -> Optional[float]:
    value = cond.get(field)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _heatmap_grid(conditions: list[dict], temps: list[float], top_ps: list[float], field: str) -> np.ma.MaskedArray:
    grid = np.full((len(temps), len(top_ps)), np.nan)
    for cond in conditions:
        value = _metric_mean(cond, field)
        if value is None:
            continue
        try:
            row = temps.index(cond["temperature"])
            col = top_ps.index(cond["top_p"])
        except ValueError:
            continue
        grid[row, col] = value
    return np.ma.masked_invalid(grid)


def _draw_heatmap(
    ax, grid: np.ma.MaskedArray, temps: list[float], top_ps: list[float], title: str, fmt: str, show_ylabel: bool = True
) -> None:
    if grid.mask.all():
        no_data_text(ax)
        ax.set_title(title, fontsize=9)
        return
    cmap = sequential_blue_cmap().with_extremes(bad="#2c2c2a")
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(top_ps)))
    ax.set_xticklabels([f"{v:.2f}" for v in top_ps], fontsize=8)
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([f"{v:.2f}" for v in temps], fontsize=8)
    ax.set_xlabel("top_p", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("temperature", fontsize=8)
    ax.set_title(title, fontsize=9)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid.mask[row, col]:
                ax.text(col, row, "--", ha="center", va="center", color=MUTED_INK, fontsize=7)
            else:
                ax.text(col, row, format(grid[row, col], fmt), ha="center", va="center", color="#0b0b0b", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _heatmap_figure(
    conditions: list[dict], efforts: list[str], temps: list[float], top_ps: list[float], field: str, title_prefix: str, fmt: str
) -> tuple:
    fig, axes = new_figure(figsize=(5.0 * len(efforts), 4.5), ncols=len(efforts))
    if len(efforts) == 1:
        axes = [axes]
    successful_counts = []
    for i, (ax, effort) in enumerate(zip(axes, efforts)):
        effort_conditions = [c for c in conditions if c["reasoning_effort"] == effort]
        successful_counts.extend(c.get("successful_calls", 0) for c in effort_conditions)
        grid = _heatmap_grid(effort_conditions, temps, top_ps, field)
        _draw_heatmap(ax, grid, temps, top_ps, f"{title_prefix} (effort={effort})", fmt, show_ylabel=(i == 0))
    return fig, successful_counts


def _interaction_lines_figure(conditions: list[dict], efforts: list[str], top_ps: list[float]) -> tuple:
    """Reasoning tokens vs temperature, one line per top_p, faceted by
    reasoning_effort -- directly visualizes the temperature x top_p x
    reasoning_effort interaction this experiment was designed to study."""
    fig, axes = new_figure(figsize=(5.0 * len(efforts), 4.5), ncols=len(efforts))
    if len(efforts) == 1:
        axes = [axes]
    successful_counts = []
    for ax, effort in zip(axes, efforts):
        effort_conditions = [c for c in conditions if c["reasoning_effort"] == effort]
        plotted_any = False
        for i, top_p in enumerate(top_ps):
            points = sorted(
                (
                    (c["temperature"], _metric_mean(c, "reasoning_tokens"))
                    for c in effort_conditions
                    if c["top_p"] == top_p and _metric_mean(c, "reasoning_tokens") is not None
                ),
                key=lambda pair: pair[0],
            )
            successful_counts.extend(
                c.get("successful_calls", 0) for c in effort_conditions if c["top_p"] == top_p
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", color=CATEGORICAL_DARK[i % len(CATEGORICAL_DARK)], label=f"top_p={top_p:.2f}")
            plotted_any = True
        if not plotted_any:
            no_data_text(ax)
        else:
            ax.set_xlabel("temperature", fontsize=8)
            ax.set_ylabel("reasoning_tokens (mean)", fontsize=8)
            ax.legend(fontsize=7)
        ax.set_title(f"effort={effort}", fontsize=9)
    return fig, successful_counts


_METRIC_HEATMAPS = [
    ("reasoning_tokens_heatmap", "reasoning_tokens", "Reasoning Tokens", ".0f"),
    ("completion_tokens_heatmap", "completion_tokens", "Completion Tokens", ".0f"),
    ("latency_heatmap", "latency_ms", "Latency (ms)", ".0f"),
    ("cost_heatmap", "mean_estimated_cost_usd", "Estimated Cost (USD)", ".6f"),
]


def generate_factorial_charts(run_dir: Path) -> list[Path]:
    """Render every legacy factorial-mode chart for one completed run, into
    `<run_dir>/charts/`. Returns the PNG paths written (empty if there is no
    usable data)."""
    summary = load_factorial_summary(run_dir)
    if not summary:
        return []
    conditions = [c for c in summary.get("conditions", []) if c.get("successful_calls", 0) > 0]
    if not conditions:
        return []

    prompt_types = _ordered_present({c["prompt_type"] for c in conditions}, ("short_prompt", "long_prompt"))
    efforts = _ordered_present({c["reasoning_effort"] for c in conditions}, REASONING_EFFORT_ORDER)
    temps = sorted({c["temperature"] for c in conditions})
    top_ps = sorted({c["top_p"] for c in conditions})

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    for prompt_type in prompt_types:
        prompt_conditions = [c for c in conditions if c["prompt_type"] == prompt_type]

        for file_stub, field, title_prefix, fmt in _METRIC_HEATMAPS:
            fig, counts = _heatmap_figure(prompt_conditions, efforts, temps, top_ps, field, title_prefix, fmt)
            fig.suptitle(f"{title_prefix} -- {prompt_type}", color="#f0efec", fontsize=12, y=1.04)
            add_footer_caption(fig, sample_size_caption(counts))
            written.append(save_png(fig, charts_dir / f"{file_stub}_{prompt_type}.png"))

        fig, counts = _interaction_lines_figure(prompt_conditions, efforts, top_ps)
        fig.suptitle(
            f"Temperature x Top-p x Reasoning-Effort Interaction (Reasoning Tokens) -- {prompt_type}",
            color="#f0efec", fontsize=11, y=1.04,
        )
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / f"parameter_interaction_{prompt_type}.png"))

    return written
