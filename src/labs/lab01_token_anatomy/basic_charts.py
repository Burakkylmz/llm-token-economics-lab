"""Automatic chart generation for Lab01's basic execution modes
(`lab01 short_prompt` / `long_prompt` / `all`).

Called automatically at the end of a successful `--live` run, right after
the existing JSON/CSV/summary files are written (see labs.lab01_token_
anatomy.cli._run) -- never during a dry run, and never making any extra API
call. Reads only the in-memory `ExperimentResult` list that run just
produced; never re-reads or rewrites `<output_name>.json` / `.csv` /
`_summary.json`. Writes PNGs into `results/<output_name>/charts/`, which
cannot collide with those sibling files: `<output_name>.json` is a file,
`<output_name>/` is a directory with no extension -- two distinct
filesystem entries even though they share a stem.

Each experiment (prompt_type) gets its own chart set, exactly like effort
mode's per-prompt_type faceting: short_prompt and long_prompt results are
never averaged together, even when both ran as part of `lab01 all`.
"""

from __future__ import annotations

from pathlib import Path

from lab_core.charts import CATEGORICAL_DARK, add_footer_caption, new_figure, no_data_text, sample_size_caption, save_png
from lab_core.metrics import ExperimentResult

CHARTS_DIRNAME = "charts"


def _draw_completion_tokens_panel(ax, results: list[ExperimentResult]) -> int:
    points = [(i, r.tokens_and_cost.completion_tokens) for i, r in enumerate(results, start=1)]
    usable = [(i, v) for i, v in points if v is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Completion Tokens per Run", fontsize=10)
        return 0
    xs, ys = zip(*usable)
    ax.plot(xs, ys, marker="o", color=CATEGORICAL_DARK[0])
    ax.set_xlabel("Run")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Completion Tokens per Run", fontsize=10)
    ax.set_xticks(list(xs))
    return len(usable)


def _draw_reasoning_vs_non_reasoning_panel(ax, results: list[ExperimentResult]) -> int:
    rows = []
    for i, r in enumerate(results, start=1):
        reasoning = r.tokens_and_cost.reasoning_tokens
        non_reasoning = r.derived.non_reasoning_completion_tokens
        if reasoning is None or non_reasoning is None:
            continue
        rows.append((i, reasoning, non_reasoning))
    if not rows:
        no_data_text(ax)
        ax.set_title("Reasoning vs Non-Reasoning Tokens", fontsize=10)
        return 0
    xs = [row[0] for row in rows]
    reasoning_vals = [row[1] for row in rows]
    non_reasoning_vals = [row[2] for row in rows]
    width = 0.35
    ax.bar([x - width / 2 for x in xs], reasoning_vals, width=width, color=CATEGORICAL_DARK[0], label="Reasoning")
    ax.bar([x + width / 2 for x in xs], non_reasoning_vals, width=width, color=CATEGORICAL_DARK[1], label="Non-reasoning")
    ax.set_xlabel("Run")
    ax.set_ylabel("Token Sayısı")
    ax.set_title("Reasoning vs Non-Reasoning Tokens", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(fontsize=7)
    return len(rows)


def _draw_finish_reason_panel(ax, results: list[ExperimentResult]) -> int:
    tally: dict[str, int] = {}
    for r in results:
        reason = r.tokens_and_cost.finish_reason or "unavailable"
        tally[reason] = tally.get(reason, 0) + 1
    if not tally:
        no_data_text(ax)
        ax.set_title("Finish Reasons", fontsize=10)
        return 0
    reasons = sorted(tally.keys())
    counts = [tally[r] for r in reasons]
    colors = [CATEGORICAL_DARK[i % len(CATEGORICAL_DARK)] for i in range(len(reasons))]
    ax.bar(range(len(reasons)), counts, color=colors)
    ax.set_xticks(range(len(reasons)))
    ax.set_xticklabels(reasons, fontsize=8)
    ax.set_ylabel("Call Sayısı")
    ax.set_title("Finish Reasons", fontsize=10)
    return sum(counts)


_PANEL_BUILDERS = [
    ("completion_tokens.png", "Completion Tokens per Run", _draw_completion_tokens_panel),
    ("reasoning_vs_non_reasoning_tokens.png", "Reasoning vs Non-Reasoning Tokens", _draw_reasoning_vs_non_reasoning_panel),
    ("finish_reasons.png", "Finish Reasons", _draw_finish_reason_panel),
]


def generate_basic_mode_charts(
    output_dir: Path, output_name: str, results_by_experiment: dict[str, list[ExperimentResult]]
) -> list[Path]:
    """Render charts for one basic-mode run (short_prompt/long_prompt/all)
    into `<output_dir>/<output_name>/charts/`. Returns the PNG paths written
    (empty if there is no usable data -- never raises on sparse/empty
    results; callers should also wrap this in try/except so a charting
    failure can never cost an already-written measurement)."""
    charts_dir = Path(output_dir) / output_name / CHARTS_DIRNAME
    written: list[Path] = []

    for experiment, results in results_by_experiment.items():
        if not results:
            continue

        caption = sample_size_caption([len(results)])

        # Three standalone single-panel charts.
        for filename, title, builder in _PANEL_BUILDERS:
            fig, ax = new_figure(figsize=(6.5, 4.8))
            builder(ax, results)
            fig.suptitle(f"LAB01 {experiment} -- {title}", color="#f0efec", fontsize=11, y=1.03)
            add_footer_caption(fig, caption)
            written.append(save_png(fig, charts_dir / f"{experiment}_{filename}"))

        # Combined per-experiment dashboard.
        fig, axes = new_figure(figsize=(16.5, 4.8), ncols=3)
        for ax, (_, title, builder) in zip(axes, _PANEL_BUILDERS):
            builder(ax, results)
        fig.suptitle(f"LAB01 -- {experiment} Dashboard", color="#f0efec", fontsize=13, y=1.04)
        add_footer_caption(fig, caption)
        written.append(save_png(fig, charts_dir / f"{experiment}_dashboard.png"))

    return written
