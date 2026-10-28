"""Chart generation from Lab07's pre-existing flat result file
(`results/lab07_rate_limiting.json`), for the "Legacy Backfill"
requirement.

Only the flat `.json` file is read, never the accompanying `.csv`. The
legacy schema only ever ran ONE fixed scenario (a configurable
`fail_count` then success) with no virtual clock, no multi-client
contention, and real `time.sleep()` -- every legacy chart/caption says so
explicitly rather than presenting it as equivalent to this redesign's
scenario matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import CATEGORICAL_DARK, add_footer_caption, new_figure, no_data_text, save_png

LEGACY_JSON_FILENAME = "lab07_rate_limiting.json"
LEGACY_CHARTS_SUBDIR = Path("lab07_rate_limiting") / "_legacy_backfill" / "charts"

LEGACY_CAPTION = (
    "Legacy backfill: pre-dates the virtual clock, multi-scenario matrix, and multi-client "
    "contention simulation -- this is a single fixed-failure-count run per strategy, with real "
    "time.sleep() wait times, not virtual time."
)

_STRATEGY_PREFIX = "lab07_"


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def generate_legacy_retry_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab07_rate_limiting.json` file, into
    `results/lab07_rate_limiting/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab07 records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    by_strategy: dict[str, dict] = {}
    for r in records:
        experiment_id = r.get("metadata", {}).get("experiment_id", "")
        if not experiment_id.startswith(_STRATEGY_PREFIX):
            continue
        strategy_id = experiment_id[len(_STRATEGY_PREFIX):]
        params = r.get("metadata", {}).get("model_parameters", {})
        by_strategy[strategy_id] = {
            "succeeded": params.get("succeeded"),
            "total_wait_seconds": params.get("total_wait_seconds"),
            "retry_count": r.get("metadata", {}).get("retry_count"),
        }

    if not by_strategy:
        return []

    strategies = sorted(by_strategy.keys())
    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []

    fig, ax = new_figure(figsize=(7, 5))
    values = [100.0 if by_strategy[s].get("succeeded") else 0.0 for s in strategies]
    ax.bar(range(len(strategies)), values, color=CATEGORICAL_DARK[0], width=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Succeeded (legacy single run, %)")
    ax.set_title("Success Rate by Strategy (legacy, n=1 per strategy)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "success_rate_by_strategy.png"))

    fig, ax = new_figure(figsize=(7, 5))
    values = [by_strategy[s].get("retry_count") or 0 for s in strategies]
    ax.bar(range(len(strategies)), values, color=CATEGORICAL_DARK[7], width=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Retry Count (proxy for HTTP 429 responses)")
    ax.set_title("Total HTTP 429 Responses (legacy)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "total_429_responses.png"))

    fig, ax = new_figure(figsize=(7, 5))
    values = [by_strategy[s].get("total_wait_seconds") for s in strategies]
    usable = [(s, v) for s, v in zip(strategies, values) if v is not None]
    if usable:
        ax.bar(range(len(usable)), [v for _, v in usable], color=CATEGORICAL_DARK[2], width=0.5)
        ax.set_xticks(range(len(usable)))
        ax.set_xticklabels([s for s, _ in usable], fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("Total Wait Seconds (legacy real time.sleep())")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Total Scheduled Wait (legacy, real wall-clock)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "total_wait_seconds.png"))

    return written
