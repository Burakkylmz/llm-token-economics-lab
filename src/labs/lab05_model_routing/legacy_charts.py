"""Chart generation from Lab05's pre-existing flat result file
(`results/lab05_model_routing.json`), for the "Legacy Backfill"
requirement.

Only the flat `.json` file is read, never the accompanying `.csv` -- same
rule as every other redesigned lab's legacy_charts module.

The legacy schema's `passed` boolean came from substring-only checks (Part
E's "Critical Problems" #4/#5) -- it is NOT the same evaluation this
redesign performs, and every legacy chart/caption says so explicitly
rather than silently presenting it as equivalent `answer_correct` data.
The legacy schema also never recorded `model_tier`, `routing_reason`, or
`task_category`, so model-selection-distribution, routing-decision, cost-
per-task, and quality-regression charts cannot be honestly reconstructed
from it -- this module does not attempt them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import CATEGORICAL_DARK, add_footer_caption, new_figure, sample_size_caption, save_png
from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab05_model_routing.charts import _draw_cost_by_strategy, _draw_latency_by_strategy, _draw_token_consumption_breakdown

LEGACY_JSON_FILENAME = "lab05_model_routing.json"
LEGACY_CHARTS_SUBDIR = Path("lab05_model_routing") / "_legacy_backfill" / "charts"

_EXPERIMENT_ID_TO_STRATEGY = {
    "lab05_fixed_model_baseline": "fixed_model_baseline",
    "lab05_rule_based_routing": "rule_based_routing",
}

LEGACY_CAPTION = (
    "Legacy backfill: pre-dates per-run isolation, deterministic multi-type evaluation, and "
    "routing-decision/task-category tracking. 'passed' used substring-only checks -- not "
    "equivalent to this redesign's answer_correct."
)


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def _build_legacy_strategy_summaries_and_records(records: list[dict]) -> tuple[dict, list[dict]]:
    by_strategy: dict[str, list[dict]] = {}
    flat_records: list[dict] = []

    for r in records:
        strategy = _EXPERIMENT_ID_TO_STRATEGY.get(r.get("metadata", {}).get("experiment_id", ""))
        if strategy is None:
            continue
        by_strategy.setdefault(strategy, []).append(r)
        tokens = r.get("tokens_and_cost", {})
        prompt_tokens = tokens.get("prompt_tokens")
        reasoning_tokens = tokens.get("reasoning_tokens")
        completion_tokens = tokens.get("completion_tokens")
        non_reasoning = (
            completion_tokens - reasoning_tokens
            if completion_tokens is not None and reasoning_tokens is not None and completion_tokens >= reasoning_tokens
            else None
        )
        flat_records.append(
            {
                "strategy": strategy,
                "succeeded": r.get("metadata", {}).get("error") is None,
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": reasoning_tokens,
                "non_reasoning_completion_tokens": non_reasoning,
            }
        )

    strategies_by_id: dict[str, dict] = {}
    for strategy, group in by_strategy.items():
        tokens_list = [g.get("tokens_and_cost", {}) for g in group]
        performance_list = [g.get("performance", {}) for g in group]
        costs = [t.get("estimated_cost_usd") for t in tokens_list if t.get("estimated_cost_usd") is not None]
        passed = sum(1 for g in group if g.get("metadata", {}).get("model_parameters", {}).get("passed") is True)
        strategies_by_id[strategy] = {
            "strategy": strategy,
            "total_tasks": len(group),
            "correct_tasks": passed,
            "accuracy": (passed / len(group)) if group else None,
            "total_estimated_cost_usd": sum(costs) if costs else None,
            "latency_ms": vars(compute_descriptive_stats([p.get("client_elapsed_ms") for p in performance_list])),
            "small_model_calls": 0,  # legacy schema never recorded model_tier -- never fabricated
            "large_model_calls": 0,
            "routing_rule_distribution": {},
        }
    return strategies_by_id, flat_records


def generate_legacy_routing_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab05_model_routing.json` file, into
    `results/lab05_model_routing/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab05 records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    strategies_by_id, flat_records = _build_legacy_strategy_summaries_and_records(records)
    if not strategies_by_id:
        return []

    order = sorted(strategies_by_id.keys())
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cost_by_strategy(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts) or LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "cost_by_strategy.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_latency_by_strategy(ax, strategies_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts) or LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "latency_by_strategy.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_token_consumption_breakdown(ax, strategies_by_id, order, flat_records)
    add_footer_caption(fig, sample_size_caption(counts) or LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "token_consumption_breakdown.png"))

    fig, ax = new_figure(figsize=(7, 5))
    usable = [s for s in order if strategies_by_id[s].get("accuracy") is not None]
    if usable:
        values = [strategies_by_id[s]["accuracy"] * 100 for s in usable]
        ax.bar(range(len(usable)), values, color=CATEGORICAL_DARK[0], width=0.5)
        ax.set_xticks(range(len(usable)))
        ax.set_xticklabels(usable, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Legacy 'passed' Rate (%)")
        ax.set_title("Legacy Pass Rate by Strategy (substring-only, NOT answer_correct)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "legacy_pass_rate_by_strategy.png"))

    return written
