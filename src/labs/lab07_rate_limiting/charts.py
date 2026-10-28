"""Chart generation for Lab07's offline retry-strategy simulation.

Reads only `experiment_summary.json` from a completed run directory (same
convention as every other redesigned lab). Every axis showing a time
value is explicitly labeled "virtual" -- this lab never measures real
wall-clock time for its simulated events (Part O's explicit "virtual vs
real time labels" requirement).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import (
    CATEGORICAL_DARK,
    MUTED_INK,
    add_footer_caption,
    new_figure,
    no_data_text,
    sample_size_caption,
    save_png,
)
from labs.lab07_rate_limiting.models import STRATEGIES
from labs.lab07_rate_limiting.scenarios import (
    SCENARIO_RETRY_AFTER_VARIATIONS,
    SCENARIO_SERVER_RECOVERY_WINDOW,
    SCENARIO_SHARED_CONTENTION,
    SCENARIO_TRANSIENT_FAILURES,
)

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"

STRATEGY_COLOR = {
    "no_backoff": CATEGORICAL_DARK[7],
    "exponential_backoff": CATEGORICAL_DARK[0],
    "exponential_backoff_with_jitter": CATEGORICAL_DARK[2],
    "rate_limit_aware": CATEGORICAL_DARK[4],
}
STRATEGY_LABEL = {
    "no_backoff": "No Backoff",
    "exponential_backoff": "Exponential Backoff",
    "exponential_backoff_with_jitter": "Exponential + Jitter",
    "rate_limit_aware": "Rate-Limit-Aware",
}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summaries_for_scenario(summaries: list[dict], scenario_id: str) -> dict[str, dict]:
    return {s["strategy_id"]: s for s in summaries if s["scenario_id"] == scenario_id}


def _draw_retry_timeline(ax, records: list[dict], scenario_id: str) -> list[int]:
    by_strategy = {s: [r for r in records if r["scenario_id"] == scenario_id and r["strategy_id"] == s and r["client_id"] == 0] for s in STRATEGIES}
    usable = [s for s in STRATEGIES if by_strategy[s]]
    if not usable:
        no_data_text(ax)
        ax.set_title("Retry Timeline (Virtual Time)", fontsize=10)
        return []
    for s in usable:
        rows = sorted(by_strategy[s], key=lambda r: r["attempt_number"])
        xs = [r["virtual_timestamp_seconds"] for r in rows]
        ys = [r["attempt_number"] for r in rows]
        ax.plot(xs, ys, marker="o", color=STRATEGY_COLOR[s], label=STRATEGY_LABEL[s])
    ax.set_xlabel("Virtual Time (s)")
    ax.set_ylabel("Attempt Number")
    ax.set_title("Retry Timeline (Virtual Time)", fontsize=10)
    ax.legend(fontsize=6, loc="upper left")
    return [1 for _ in usable]


def _draw_backoff_schedule(ax) -> list[int]:
    from lab_core.retry import compute_backoff_delay_seconds

    retry_indices = list(range(1, 7))
    delays = [compute_backoff_delay_seconds(i, 1.0, 30.0, jitter=False) for i in retry_indices]
    ax.plot(retry_indices, delays, marker="o", color=CATEGORICAL_DARK[0])
    ax.set_xlabel("Retry Index")
    ax.set_ylabel("Nominal Delay (virtual s)")
    ax.set_title("Exponential Backoff Schedule", fontsize=10)
    return [1]


def _draw_jitter_distribution(ax, records: list[dict]) -> list[int]:
    usable = [
        r["scheduled_delay_seconds"] for r in records
        if r["strategy_id"] == "exponential_backoff_with_jitter" and r.get("scheduled_delay_seconds") is not None
    ]
    if not usable:
        no_data_text(ax)
        ax.set_title("Backoff vs Jitter Distribution", fontsize=10)
        return []
    ax.hist(usable, bins=min(10, max(3, len(usable))), color=CATEGORICAL_DARK[2], edgecolor=MUTED_INK)
    ax.set_xlabel("Scheduled Delay (virtual s)")
    ax.set_ylabel("Count")
    ax.set_title("Backoff vs Jitter Distribution (seeded)", fontsize=10)
    return [len(usable)]


def _draw_success_rate_by_strategy(ax, summaries: list[dict], scenario_id: str) -> list[int]:
    by_strategy = _summaries_for_scenario(summaries, scenario_id)
    usable = [s for s in STRATEGIES if by_strategy.get(s, {}).get("success_rate") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Success Rate by Strategy", fontsize=10)
        return []
    values = [by_strategy[s]["success_rate"] * 100 for s in usable]
    colors = [STRATEGY_COLOR[s] for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in usable], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Success Rate by Strategy ({scenario_id})", fontsize=9)
    return [by_strategy[s]["total_clients"] for s in usable]


def _draw_total_429_by_strategy(ax, summaries: list[dict], scenario_id: str) -> list[int]:
    by_strategy = _summaries_for_scenario(summaries, scenario_id)
    usable = [s for s in STRATEGIES if s in by_strategy]
    if not usable:
        no_data_text(ax)
        ax.set_title("Total HTTP 429 Responses", fontsize=10)
        return []
    values = [by_strategy[s]["total_429_responses"] for s in usable]
    colors = [STRATEGY_COLOR[s] for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in usable], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Total HTTP 429 Responses")
    ax.set_title(f"Total HTTP 429 Responses ({scenario_id})", fontsize=9)
    return [by_strategy[s]["total_clients"] for s in usable]


def _draw_virtual_recovery_time(ax, summaries: list[dict], scenario_id: str) -> list[int]:
    by_strategy = _summaries_for_scenario(summaries, scenario_id)
    usable = [s for s in STRATEGIES if by_strategy.get(s, {}).get("completion_time_seconds") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Virtual Recovery Time", fontsize=10)
        return []
    values = [by_strategy[s]["completion_time_seconds"] for s in usable]
    colors = [STRATEGY_COLOR[s] for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in usable], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Virtual Completion Time (s)")
    ax.set_title(f"Virtual Recovery Time ({scenario_id})", fontsize=9)
    return [by_strategy[s]["successful_clients"] for s in usable]


def _draw_retry_collision_analysis(ax, summaries: list[dict], scenario_id: str) -> list[int]:
    by_strategy = _summaries_for_scenario(summaries, scenario_id)
    usable = [s for s in STRATEGIES if s in by_strategy]
    if not usable:
        no_data_text(ax)
        ax.set_title("Retry Collision Analysis", fontsize=10)
        return []
    values = [by_strategy[s]["retry_collision_count"] for s in usable]
    colors = [STRATEGY_COLOR[s] for s in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in usable], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Retry Collisions")
    ax.set_title(f"Retry Collision Analysis ({scenario_id})", fontsize=9)
    return [by_strategy[s]["total_clients"] for s in usable]


def _draw_retry_after_compliance(ax, records: list[dict]) -> list[int]:
    variant_records = [
        r for r in records
        if r["scenario_id"] == SCENARIO_RETRY_AFTER_VARIATIONS and r["strategy_id"] == "rate_limit_aware" and r["attempt_number"] == 1
    ]
    if not variant_records:
        no_data_text(ax, "Veri yok: retry_after_variations senaryosu çalıştırılmadı (extended profile gerekir)")
        ax.set_title("Retry-After Compliance", fontsize=10)
        return []
    honored = sum(1 for r in variant_records if r["delay_reason"] == "retry_after_header")
    fallback = len(variant_records) - honored
    ax.bar(["Honored", "Fallback"], [honored, fallback], color=[CATEGORICAL_DARK[2], CATEGORICAL_DARK[7]], width=0.5)
    ax.set_ylabel("Client Count")
    ax.set_title("Retry-After Compliance", fontsize=10)
    return [len(variant_records)]


def generate_retry_charts(run_dir: Path) -> list[Path]:
    """Render every Lab07 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    summaries = summary.get("strategy_summaries", [])
    if not summaries:
        return []
    records = summary.get("attempt_records", [])
    scenario_ids_present = {s["scenario_id"] for s in summaries}
    primary_scenario = SCENARIO_TRANSIENT_FAILURES if SCENARIO_TRANSIENT_FAILURES in scenario_ids_present else next(iter(scenario_ids_present))
    recovery_scenario = SCENARIO_SERVER_RECOVERY_WINDOW if SCENARIO_SERVER_RECOVERY_WINDOW in scenario_ids_present else primary_scenario
    contention_scenario = SCENARIO_SHARED_CONTENTION if SCENARIO_SHARED_CONTENTION in scenario_ids_present else primary_scenario

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_retry_timeline(ax, records, primary_scenario)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "retry_timeline.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_backoff_schedule(ax)
    add_footer_caption(fig, None)
    written.append(save_png(fig, charts_dir / "exponential_backoff_schedule.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_jitter_distribution(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "backoff_vs_jitter_distribution.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_success_rate_by_strategy(ax, summaries, primary_scenario)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "success_rate_by_strategy.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_total_429_by_strategy(ax, summaries, primary_scenario)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "total_429_responses.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_virtual_recovery_time(ax, summaries, recovery_scenario)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "virtual_recovery_time.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_retry_collision_analysis(ax, summaries, contention_scenario)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "retry_collision_analysis.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_retry_after_compliance(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "retry_after_compliance.png"))

    fig, axes = new_figure(figsize=(20, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_success_rate_by_strategy(axes[0][0], summaries, primary_scenario))
    all_counts.extend(_draw_total_429_by_strategy(axes[0][1], summaries, primary_scenario))
    all_counts.extend(_draw_virtual_recovery_time(axes[0][2], summaries, recovery_scenario))
    all_counts.extend(_draw_retry_collision_analysis(axes[0][3], summaries, contention_scenario))
    all_counts.extend(_draw_retry_timeline(axes[1][0], records, primary_scenario))
    all_counts.extend(_draw_jitter_distribution(axes[1][1], records))
    all_counts.extend(_draw_retry_after_compliance(axes[1][2], records))
    no_data_text(axes[1][3], "")
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB07 Resilience Trade-off Dashboard -- {summary.get('experiment_run_id', run_dir.name)} (SIMULATION ONLY)",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "resilience_tradeoff_dashboard.png"))

    return written
