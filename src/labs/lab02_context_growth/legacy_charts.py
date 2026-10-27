"""Chart generation from Lab02's pre-existing flat result files
(`results/lab02_context_growth.json`), for charts the task's "Legacy
Backfill" requirement asks for.

Only the flat `.json` file is read, never the accompanying `.csv`:
`write_json` (lab_core.results_writer) OVERWRITES that file on every run,
so it always reflects exactly one coherent experiment, while `append_csv`
accumulates rows across every run ever executed -- mixing them would
merge unrelated run histories into one fake "experiment", which this
project's data-integrity policy forbids (see this lab's design task, Part
I "Legacy Backfill").

The legacy schema never stored the model's final answer text, so answer
correctness cannot be reconstructed for it at all -- this module simply
never produces an answer-correctness chart, rather than rendering a
misleading all-zero one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import add_footer_caption, new_figure, sample_size_caption, save_png
from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab02_context_growth.charts import (
    _draw_context_size_vs_latency,
    _draw_cumulative_consumption,
    _draw_multi_turn_growth,
    _draw_single_metric_bar,
    _draw_token_breakdown,
    _ordered_conditions_present,
)
from labs.lab02_context_growth.experiments import CONTEXT_CONDITION_ORDER

LEGACY_JSON_FILENAME = "lab02_context_growth.json"
LEGACY_CHARTS_SUBDIR = Path("lab02_context_growth") / "_legacy_backfill" / "charts"


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def _condition_id_from_experiment_id(experiment_id: str) -> Optional[str]:
    candidate = experiment_id.removeprefix("lab02_")
    return candidate if candidate in CONTEXT_CONDITION_ORDER else None


def _build_legacy_condition_summaries(records: list[dict]) -> list[dict]:
    by_condition: dict[str, list[dict]] = {}
    for r in records:
        condition_id = _condition_id_from_experiment_id(r.get("metadata", {}).get("experiment_id", ""))
        if condition_id is not None:
            by_condition.setdefault(condition_id, []).append(r)

    summaries = []
    for condition_id, group in by_condition.items():
        tokens = [g.get("tokens_and_cost", {}) for g in group]
        performance = [g.get("performance", {}) for g in group]
        summaries.append(
            {
                "condition_id": condition_id,
                "successful_calls": len(group),
                "prompt_tokens": vars(compute_descriptive_stats([t.get("prompt_tokens") for t in tokens])),
                "completion_tokens": vars(compute_descriptive_stats([t.get("completion_tokens") for t in tokens])),
                "reasoning_tokens": vars(compute_descriptive_stats([t.get("reasoning_tokens") for t in tokens])),
                "non_reasoning_completion_tokens": vars(
                    compute_descriptive_stats([g.get("derived", {}).get("non_reasoning_completion_tokens") for g in group])
                ),
                "latency_ms": vars(compute_descriptive_stats([p.get("client_elapsed_ms") for p in performance])),
                "mean_estimated_cost_usd": (
                    lambda costs: (sum(costs) / len(costs)) if costs else None
                )([t.get("estimated_cost_usd") for t in tokens if t.get("estimated_cost_usd") is not None]),
            }
        )
    return summaries


def _build_legacy_turn_records(records: list[dict]) -> list[dict]:
    turns = []
    for r in records:
        if r.get("metadata", {}).get("experiment_id") != "lab02_multi_turn_growth":
            continue
        turn_index = r.get("metadata", {}).get("model_parameters", {}).get("turn")
        if turn_index is None:
            continue
        turns.append(
            {
                "turn_index": turn_index,
                "prompt_tokens": r.get("tokens_and_cost", {}).get("prompt_tokens"),
                "succeeded": True,
            }
        )
    turns.sort(key=lambda t: t["turn_index"])
    # cumulative_prompt_tokens is a derived running sum of already-measured
    # prompt_tokens values -- never a fabricated number -- computed here
    # because the legacy schema never stored it.
    running_total = 0
    for t in turns:
        if t["prompt_tokens"] is not None:
            running_total += t["prompt_tokens"]
            t["cumulative_prompt_tokens"] = running_total
        else:
            t["cumulative_prompt_tokens"] = None
    return turns


def generate_legacy_context_growth_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab02_context_growth.json` file, into
    `results/lab02_context_growth/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files themselves. Returns an empty
    list if the legacy file is absent/empty, or has no recognizable Lab02
    condition/multi-turn records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    condition_summaries = _build_legacy_condition_summaries(records)
    turn_records = _build_legacy_turn_records(records)
    if not condition_summaries and not turn_records:
        return []

    order = _ordered_conditions_present(condition_summaries) if condition_summaries else list(CONTEXT_CONDITION_ORDER)
    conditions_by_id = {c["condition_id"]: c for c in condition_summaries}
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []

    if condition_summaries:
        fig, axes = new_figure(figsize=(11, 4.8), ncols=2)
        counts = _draw_context_size_vs_latency(axes[0], axes[1], conditions_by_id, order)
        add_footer_caption(fig, sample_size_caption(counts) or "Legacy backfill: pre-dates per-run isolation and answer-correctness evaluation.")
        written.append(save_png(fig, charts_dir / "context_size_vs_latency.png"))

        fig, ax = new_figure(figsize=(8, 5))
        counts = _draw_token_breakdown(ax, conditions_by_id, order)
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / "token_breakdown.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_single_metric_bar(ax, conditions_by_id, order, "reasoning_tokens", "Reasoning Tokens by Context", "Token Sayısı")
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / "reasoning_tokens_by_context.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_single_metric_bar(ax, conditions_by_id, order, "mean_estimated_cost_usd", "Estimated Cost by Context", "Tahmini Maliyet (USD)")
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / "estimated_cost_by_context.png"))

    if turn_records:
        fig, ax = new_figure(figsize=(7, 5))
        _draw_multi_turn_growth(ax, turn_records)
        written.append(save_png(fig, charts_dir / "multi_turn_growth.png"))

        fig, ax = new_figure(figsize=(7, 5))
        _draw_cumulative_consumption(ax, turn_records)
        written.append(save_png(fig, charts_dir / "cumulative_token_consumption.png"))

    return written
