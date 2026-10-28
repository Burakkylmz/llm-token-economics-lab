"""Chart generation from Lab04's pre-existing flat result file
(`results/lab04_output_control.json`), for the "Legacy Backfill"
requirement.

Only the flat `.json` file is read, never the accompanying `.csv` -- same
rule as labs.lab02_context_growth.legacy_charts/labs.lab03_prompt_caching.legacy_charts
(`write_json` overwrites that file every run; `append_csv` accumulates rows
across every run ever executed, so mixing them would merge unrelated run
histories into one fake "experiment").

The legacy schema never stored the model's final answer text, so answer
completeness/correctness cannot be reconstructed (mean_field_completeness_
ratio/mean_field_correctness_ratio/accepted_count stay None/0 here, never
fabricated) -- but prompt/completion/reasoning tokens, finish_reason, cost,
and per-call latency WERE already recorded by the pre-redesign pipeline
(lab_core.groq_client.call_sync has always normalized these), so those
charts are faithfully reconstructed. Workflow wall-clock latency was never
measured by the legacy pipeline, so it stays None rather than being
approximated from per-call latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import add_footer_caption, new_figure, sample_size_caption, save_png
from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab04_output_control.charts import (
    _draw_cost_vs_budget,
    _draw_completion_tokens_vs_budget,
    _draw_finish_reason_distribution,
    _draw_invocation_cost_comparison,
    _draw_invocation_token_breakdown,
    _draw_reasoning_vs_non_reasoning,
    _ordered_conditions,
)
from labs.lab04_output_control.experiments import condition_label_for

LEGACY_JSON_FILENAME = "lab04_output_control.json"
LEGACY_CHARTS_SUBDIR = Path("lab04_output_control") / "_legacy_backfill" / "charts"

OUTPUT_LIMIT_EXPERIMENT_ID = "lab04_output_limit_sweep"
SINGLE_INVOCATION_EXPERIMENT_ID = "lab04_single_invocation"
MULTIPLE_INVOCATION_EXPERIMENT_ID = "lab04_multiple_invocations"


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def _build_legacy_output_limit_conditions(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_limit: dict[int, list[dict]] = {}
    flat_records: list[dict] = []

    sweep_records = [r for r in records if r.get("metadata", {}).get("experiment_id") == OUTPUT_LIMIT_EXPERIMENT_ID]
    for index, r in enumerate(sweep_records, start=1):
        max_tokens = r.get("metadata", {}).get("model_parameters", {}).get("max_tokens")
        if max_tokens is None:
            continue
        by_limit.setdefault(max_tokens, []).append(r)
        tokens = r.get("tokens_and_cost", {})
        condition_id = condition_label_for(max_tokens)
        flat_records.append(
            {
                "execution_index": index,
                "condition_id": condition_id,
                "succeeded": r.get("metadata", {}).get("error") is None,
                "finish_reason": tokens.get("finish_reason"),
            }
        )

    conditions = []
    for max_tokens, group in by_limit.items():
        tokens_list = [g.get("tokens_and_cost", {}) for g in group]
        performance_list = [g.get("performance", {}) for g in group]
        costs = [t.get("estimated_cost_usd") for t in tokens_list if t.get("estimated_cost_usd") is not None]
        truncated = [t for t in tokens_list if t.get("finish_reason") == "length"]
        conditions.append(
            {
                "condition_id": condition_label_for(max_tokens),
                "max_completion_tokens": max_tokens,
                "successful_calls": len(group),
                "truncated_calls": len(truncated),
                "truncation_rate": (len(truncated) / len(group)) if group else None,
                "completion_tokens": vars(compute_descriptive_stats([t.get("completion_tokens") for t in tokens_list])),
                "reasoning_tokens": vars(compute_descriptive_stats([t.get("reasoning_tokens") for t in tokens_list])),
                "non_reasoning_completion_tokens": vars(
                    compute_descriptive_stats([g.get("derived", {}).get("non_reasoning_completion_tokens") for g in group])
                ),
                "mean_estimated_cost_usd": (sum(costs) / len(costs)) if costs else None,
                "accepted_count": 0,  # legacy data stored no answer text -- never fabricated
                "mean_field_completeness_ratio": None,
                "mean_field_correctness_ratio": None,
            }
        )
    return conditions, flat_records


def _strategy_totals(records: list[dict]) -> dict:
    tokens_list = [r.get("tokens_and_cost", {}) for r in records]
    costs = [t.get("estimated_cost_usd") for t in tokens_list if t.get("estimated_cost_usd") is not None]
    return {
        "api_call_count": len(records),
        "total_prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in tokens_list),
        "total_completion_tokens": sum(t.get("completion_tokens") or 0 for t in tokens_list),
        "total_reasoning_tokens": sum(t.get("reasoning_tokens") or 0 for t in tokens_list),
        "total_estimated_cost_usd": sum(costs) if costs else None,
        # Never reconstructed: the legacy pipeline did not measure workflow
        # wall-clock latency at all (only added in this redesign).
        "workflow_wall_clock_ms": None,
        "individual_request_latencies_ms": [r.get("performance", {}).get("client_elapsed_ms") for r in records],
        "correct_answers": 0,
        "missing_answers": 0,
        "invalid_answers": 0,
        "all_answers_accepted": False,
    }


def _build_legacy_invocation_comparison(records: list[dict]) -> Optional[dict]:
    single = [r for r in records if r.get("metadata", {}).get("experiment_id") == SINGLE_INVOCATION_EXPERIMENT_ID]
    multiple = [r for r in records if r.get("metadata", {}).get("experiment_id") == MULTIPLE_INVOCATION_EXPERIMENT_ID]
    if not single or not multiple:
        return None
    return {"single": _strategy_totals(single), "multiple": _strategy_totals(multiple)}


def generate_legacy_output_control_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab04_output_control.json` file, into
    `results/lab04_output_control/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab04 records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    conditions, flat_records = _build_legacy_output_limit_conditions(records)
    comparison = _build_legacy_invocation_comparison(records)
    if not conditions and comparison is None:
        return []

    order = _ordered_conditions(conditions)
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []
    caption = "Legacy backfill: pre-dates per-run isolation, answer evaluation, and workflow-latency tracking."

    if conditions:
        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_completion_tokens_vs_budget(ax, order)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "completion_tokens_vs_budget.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_reasoning_vs_non_reasoning(ax, order)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "reasoning_vs_non_reasoning_tokens.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_finish_reason_distribution(ax, order, flat_records)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "finish_reason_distribution.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_cost_vs_budget(ax, order)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "output_limit_vs_estimated_cost.png"))

    if comparison is not None:
        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_invocation_token_breakdown(ax, comparison)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "invocation_token_breakdown.png"))

        fig, ax = new_figure(figsize=(7, 5))
        counts = _draw_invocation_cost_comparison(ax, comparison)
        add_footer_caption(fig, sample_size_caption(counts) or caption)
        written.append(save_png(fig, charts_dir / "invocation_cost_comparison.png"))

    return written
