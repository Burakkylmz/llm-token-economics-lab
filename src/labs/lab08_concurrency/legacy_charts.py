"""Chart generation from Lab08's pre-existing flat result file
(`results/lab08_concurrency.json`), for the "Legacy Backfill" requirement.

Only the flat `.json` file is read, never the accompanying `.csv`. The
legacy schema never recorded a batch wall-clock duration directly, so
this module reconstructs an APPROXIMATE one from each group's own
min/max ISO timestamps (`max(timestamp) - min(timestamp)`) -- an honest,
documented approximation, captioned as such, never presented as this
redesign's directly-measured `batch_wall_clock_ms`. The legacy schema also
mislabeled `execution_mode` as `"sync"` for both conditions even though
both used the async client -- this module does not repeat that label.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from lab_core.charts import CATEGORICAL_DARK, add_footer_caption, new_figure, no_data_text, save_png
from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats

LEGACY_JSON_FILENAME = "lab08_concurrency.json"
LEGACY_CHARTS_SUBDIR = Path("lab08_concurrency") / "_legacy_backfill" / "charts"

LEGACY_CAPTION = (
    "Legacy backfill: pre-dates per-run isolation, queue-wait instrumentation, and correctness "
    "evaluation. Batch wall-clock here is an APPROXIMATION reconstructed from each group's own "
    "min/max call timestamps, not this redesign's directly-measured value."
)

_EXPERIMENT_ID_TO_CONDITION = {"lab08_sequential": "sequential", "lab08_concurrent": "concurrent"}


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def generate_legacy_concurrency_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab08_concurrency.json` file, into
    `results/lab08_concurrency/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab08 records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    by_condition: dict[str, list[dict]] = {}
    for r in records:
        condition = _EXPERIMENT_ID_TO_CONDITION.get(r.get("metadata", {}).get("experiment_id", ""))
        if condition is None:
            continue
        by_condition.setdefault(condition, []).append(r)

    if not by_condition:
        return []

    conditions = sorted(by_condition.keys())
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []

    batch_durations_ms = {}
    for condition, group in by_condition.items():
        timestamps = [t for t in (_parse_timestamp(g.get("metadata", {}).get("timestamp", "")) for g in group) if t is not None]
        if len(timestamps) >= 2:
            batch_durations_ms[condition] = (max(timestamps) - min(timestamps)).total_seconds() * 1000

    fig, ax = new_figure(figsize=(7, 5))
    usable = [c for c in conditions if c in batch_durations_ms]
    if usable:
        ax.bar(usable, [batch_durations_ms[c] for c in usable], color=CATEGORICAL_DARK[0], width=0.5)
        ax.set_ylabel("Approx. Batch Duration (ms)")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Batch Completion Time (legacy, approximate)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "batch_completion_time.png"))

    fig, ax = new_figure(figsize=(7, 5))
    means, p95s = [], []
    for condition in conditions:
        latencies = [g.get("performance", {}).get("client_elapsed_ms") for g in by_condition[condition]]
        stats = compute_descriptive_stats(latencies)
        means.append(stats.mean or 0)
        p95s.append(stats.p95)
    ax.bar(range(len(conditions)), means, color=CATEGORICAL_DARK[4], width=0.5, label="Mean")
    ax.scatter(range(len(conditions)), p95s, color=CATEGORICAL_DARK[7], zorder=3, label="P95", marker="D")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Client Elapsed Latency (ms)")
    ax.set_title("Service Latency Distribution (legacy)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "service_latency_distribution.png"))

    fig, ax = new_figure(figsize=(7, 5))
    costs = []
    for condition in conditions:
        condition_costs = [g.get("tokens_and_cost", {}).get("estimated_cost_usd") for g in by_condition[condition]]
        condition_costs = [c for c in condition_costs if c is not None]
        costs.append(sum(condition_costs) if condition_costs else None)
    usable_cost = [(c, v) for c, v in zip(conditions, costs) if v is not None]
    if usable_cost:
        ax.bar([c for c, _ in usable_cost], [v for _, v in usable_cost], color=CATEGORICAL_DARK[3], width=0.5)
        ax.set_ylabel("Total Estimated Cost (USD)")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Token & Cost Comparison (legacy)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "token_cost_comparison.png"))

    return written
