"""Chart generation from Lab03's pre-existing flat result file
(`results/lab03_prompt_caching.json`), for the "Legacy Backfill"
requirement.

Only the flat `.json` file is read, never the accompanying `.csv`:
`write_json` (lab_core.results_writer) OVERWRITES that file on every run,
so it always reflects exactly one coherent experiment, while `append_csv`
accumulates rows across every run ever executed -- mixing them would
merge unrelated run histories into one fake "experiment" (same rule as
labs.lab02_context_growth.legacy_charts).

The legacy schema never stored the model's final answer text, so answer
correctness cannot be reconstructed -- this module never produces an
answer-correctness chart or a full summary dashboard, rather than
rendering a misleading all-zero/partial one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import add_footer_caption, new_figure, sample_size_caption, save_png
from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats
from labs.lab03_prompt_caching.cache_metrics import compute_request_cache_hit, compute_request_cache_ratio
from labs.lab03_prompt_caching.cache_status import classify_cache_status
from labs.lab03_prompt_caching.charts import (
    _draw_cache_friendly_vs_unfriendly,
    _draw_cache_ratio_by_request,
    _draw_cache_status_by_request,
    _draw_cached_tokens_vs_latency,
    _draw_cached_vs_uncached_tokens,
    _draw_single_metric_bar,
    _ordered_conditions_present,
)
from labs.lab03_prompt_caching.experiments import CONDITION_ORDER

LEGACY_JSON_FILENAME = "lab03_prompt_caching.json"
LEGACY_CHARTS_SUBDIR = Path("lab03_prompt_caching") / "_legacy_backfill" / "charts"

_EXPERIMENT_ID_TO_CONDITION = {
    "lab03_repeated_prefix": "repeated_prefix",
    "lab03_cache_unfriendly": "cache_unfriendly",
}


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def _build_legacy_condition_summaries_and_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_condition: dict[str, list[dict]] = {}
    flat_records: list[dict] = []

    for index, r in enumerate(records, start=1):
        condition_id = _EXPERIMENT_ID_TO_CONDITION.get(r.get("metadata", {}).get("experiment_id", ""))
        if condition_id is None:
            continue
        by_condition.setdefault(condition_id, []).append(r)
        tokens = r.get("tokens_and_cost", {})
        performance = r.get("performance", {})
        prompt_tokens = tokens.get("prompt_tokens")
        cached_tokens = tokens.get("cached_tokens")
        flat_records.append(
            {
                "execution_index": index,
                "condition_id": condition_id,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "uncached_prompt_tokens": (
                    prompt_tokens - cached_tokens
                    if prompt_tokens is not None and cached_tokens is not None and prompt_tokens >= cached_tokens
                    else None
                ),
                "total_latency_ms": performance.get("client_elapsed_ms"),
                "request_cache_hit": compute_request_cache_hit(cached_tokens),
                "request_cache_ratio": compute_request_cache_ratio(cached_tokens, prompt_tokens),
                "cache_status": classify_cache_status(cached_tokens, prompt_tokens),
            }
        )

    summaries = []
    for condition_id, group in by_condition.items():
        tokens_list = [g.get("tokens_and_cost", {}) for g in group]
        performance_list = [g.get("performance", {}) for g in group]
        prompt_list = [t.get("prompt_tokens") for t in tokens_list]
        cached_list = [t.get("cached_tokens") for t in tokens_list]
        uncached_list = [
            (p - c) if (p is not None and c is not None and p >= c) else None for p, c in zip(prompt_list, cached_list)
        ]
        costs = [t.get("estimated_cost_usd") for t in tokens_list if t.get("estimated_cost_usd") is not None]
        reported = [c for c in cached_list if c is not None]
        summaries.append(
            {
                "condition_id": condition_id,
                "successful_calls": len(group),
                "prompt_tokens": vars(compute_descriptive_stats(prompt_list)),
                "cached_tokens": vars(compute_descriptive_stats(cached_list)),
                "uncached_prompt_tokens": vars(compute_descriptive_stats(uncached_list)),
                "latency_ms": vars(compute_descriptive_stats([p.get("client_elapsed_ms") for p in performance_list])),
                "mean_estimated_cost_usd": (sum(costs) / len(costs)) if costs else None,
                "total_estimated_cache_savings_usd": None,  # legacy pricing lacked cache-aware savings tracking
                "observed_request_cache_hit_rate": (
                    (sum(1 for c in reported if c > 0) / len(reported)) if reported else None
                ),
                "correct_count": 0,
                "incorrect_count": 0,
                "format_invalid_count": 0,
                "missing_response_count": 0,
            }
        )
    return summaries, flat_records


def generate_legacy_cache_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab03_prompt_caching.json` file, into
    `results/lab03_prompt_caching/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab03 records.
    Never produces an answer-correctness chart (legacy data cannot support
    one)."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    summaries, flat_records = _build_legacy_condition_summaries_and_records(records)
    if not summaries:
        return []

    order = _ordered_conditions_present(summaries)
    conditions_by_id = {c["condition_id"]: c for c in summaries}
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cached_vs_uncached_tokens(ax, conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts) or "Legacy backfill: pre-dates per-run isolation and cache-savings tracking.")
    written.append(save_png(fig, charts_dir / "cached_vs_uncached_tokens.png"))

    fig, ax = new_figure(figsize=(9, 3.5))
    counts = _draw_cache_status_by_request(ax, flat_records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_status_by_request.png"))

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cache_ratio_by_request(ax, flat_records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_ratio_by_request.png"))

    fig, axes = new_figure(figsize=(12, 5), ncols=2)
    counts = _draw_cache_friendly_vs_unfriendly(axes[0], axes[1], conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_friendly_vs_unfriendly_comparison.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "latency_ms", "Latency by Condition", "Gecikme (ms)")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "latency_by_condition.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "mean_estimated_cost_usd", "Estimated Cost by Condition", "Tahmini Maliyet (USD)")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "estimated_cost_by_condition.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cached_tokens_vs_latency(ax, flat_records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cached_tokens_vs_latency.png"))

    return written
