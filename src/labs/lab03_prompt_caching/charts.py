"""Chart generation for Lab03's prompt-caching experiment.

Reads only `experiment_summary.json` (condition-level summaries, including
per-call records needed for the request-indexed charts) from a completed
run directory, and renders PNGs into a `charts/` subdirectory next to it.
Never reads or writes `cache_measurements.*`, `experiment_plan.csv`,
`experiment_manifest.json`, or `analysis_report.md` directly (those are
read by the CLI to BUILD experiment_summary.json, not by this module).

Same data-integrity rules as labs.lab02_context_growth.charts: a condition
with 0 successful calls is never drawn as a zero-height bar ("Veri yok"
instead); n=1 groups never get an error bar; the two conditions are never
averaged together; a request that never reported cached_tokens is shown as
missing, never as a zero-token cache hit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import (
    CATEGORICAL_DARK,
    MUTED_INK,
    SECONDARY_INK,
    add_footer_caption,
    new_figure,
    no_data_text,
    sample_size_caption,
    save_png,
)
from labs.lab03_prompt_caching.cache_status import CACHE_HIT, CACHE_MISS, CACHE_USAGE_UNAVAILABLE, INVALID_USAGE
from labs.lab03_prompt_caching.experiments import CONDITION_ORDER

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"
PROBE_EXPERIMENT_MODE = "cache_probe"

CACHE_STATUS_COLOR = {
    CACHE_HIT: CATEGORICAL_DARK[2],  # aqua/green-ish: a real, positive observation
    CACHE_MISS: CATEGORICAL_DARK[0],  # blue: a real, reported-zero observation
    CACHE_USAGE_UNAVAILABLE: MUTED_INK,  # muted/gray: never colored like real data
    INVALID_USAGE: CATEGORICAL_DARK[7],  # red: flagged as unreliable
}
CACHE_STATUS_ORDER = (CACHE_HIT, CACHE_MISS, CACHE_USAGE_UNAVAILABLE, INVALID_USAGE)

CONDITION_COLOR = {
    "repeated_prefix": CATEGORICAL_DARK[0],
    "cache_unfriendly": CATEGORICAL_DARK[1],
}
CONDITION_LABEL = {
    "repeated_prefix": "Repeated Prefix",
    "cache_unfriendly": "Cache Unfriendly",
}


def _load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_conditions_present(conditions: list[dict]) -> list[str]:
    present = {c["condition_id"] for c in conditions}
    ordered = [c for c in CONDITION_ORDER if c in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _usable(conditions_by_id: dict[str, dict], order: list[str], field: str) -> list[str]:
    usable = []
    for cid in order:
        cond = conditions_by_id.get(cid)
        if cond is None or cond.get("successful_calls", 0) == 0:
            continue
        value = cond.get(field)
        mean = value.get("mean") if isinstance(value, dict) else value
        if mean is not None:
            usable.append(cid)
    return usable


def _mean_and_err(cond: dict, field: str) -> tuple[Optional[float], Optional[float]]:
    value = cond.get(field)
    if isinstance(value, dict):
        return value.get("mean"), value.get("sample_standard_deviation")
    return value, None


def _draw_single_metric_bar(ax, conditions_by_id, order, field, title, ylabel) -> list[int]:
    usable = _usable(conditions_by_id, order, field)
    if not usable:
        no_data_text(ax)
        ax.set_title(title, fontsize=10)
        return []
    values, errors, colors, labels = [], [], [], []
    for cid in usable:
        mean, err = _mean_and_err(conditions_by_id[cid], field)
        values.append(mean)
        errors.append(err if err is not None else 0.0)
        colors.append(CONDITION_COLOR.get(cid, SECONDARY_INK))
        labels.append(CONDITION_LABEL.get(cid, cid))
    x = range(len(usable))
    ax.bar(x, values, yerr=errors, color=colors, capsize=4, width=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    missing = [cid for cid in order if cid not in usable]
    if missing:
        ax.text(
            0.5, -0.22, f"Veri yok: {', '.join(missing)}", transform=ax.transAxes,
            ha="center", va="top", color=SECONDARY_INK, fontsize=7, style="italic",
        )
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_cached_vs_uncached_tokens(ax, conditions_by_id, order) -> list[int]:
    usable = [
        cid for cid in order
        if cid in _usable(conditions_by_id, order, "cached_tokens")
        or cid in _usable(conditions_by_id, order, "uncached_prompt_tokens")
    ]
    if not usable:
        no_data_text(ax)
        ax.set_title("Cached vs Uncached Input Tokens", fontsize=10)
        return []
    width = 0.3
    x = list(range(len(usable)))
    for field, offset, label in [("cached_tokens", -width / 2, "Cached"), ("uncached_prompt_tokens", width / 2, "Uncached")]:
        values = [(_mean_and_err(conditions_by_id[cid], field)[0] or 0) for cid in usable]
        ax.bar([i + offset for i in x], values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Token Sayısı")
    ax.set_title("Cached vs Uncached Input Tokens", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_cache_ratio_by_request(ax, records: list[dict]) -> list[int]:
    usable = [r for r in records if r.get("request_cache_ratio") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Cache Ratio by Request", fontsize=10)
        return []
    xs = [r["execution_index"] for r in usable]
    ys = [r["request_cache_ratio"] * 100 for r in usable]
    colors = [CONDITION_COLOR.get(r.get("condition_id"), SECONDARY_INK) for r in usable]
    ax.scatter(xs, ys, c=colors, s=28)
    ax.set_xlabel("Execution Index")
    ax.set_ylabel("Cache Ratio (%)")
    ax.set_title("Cache Ratio by Request", fontsize=10)
    return [1 for _ in usable]


def _draw_request_hit_rate_by_condition(ax, conditions_by_id, order) -> list[int]:
    usable = [cid for cid in order if conditions_by_id.get(cid, {}).get("observed_request_cache_hit_rate") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Request Cache Hit Rate by Condition", fontsize=10)
        return []
    values = [conditions_by_id[cid]["observed_request_cache_hit_rate"] * 100 for cid in usable]
    colors = [CONDITION_COLOR.get(cid, SECONDARY_INK) for cid in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Observed Request Hit Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Request Cache Hit Rate by Condition (Metric A)", fontsize=10)
    missing = [cid for cid in order if cid not in usable]
    if missing:
        ax.text(0.5, -0.22, f"Veri yok: {', '.join(missing)}", transform=ax.transAxes, ha="center", va="top", color=SECONDARY_INK, fontsize=7, style="italic")
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_cache_status_by_request(ax, records: list[dict]) -> list[int]:
    """Required chart #2: Cache Hit / Miss / Unavailable by Request.

    Every record with a known cache_status is plotted (including
    CACHE_USAGE_UNAVAILABLE and INVALID_USAGE) -- this is the one chart
    where "no data" is itself the data point being shown, by design, so it
    never calls no_data_text for an empty y-axis; it only shows truly
    empty axes if there are literally no records at all.
    """
    usable = [r for r in records if r.get("cache_status")]
    if not usable:
        no_data_text(ax)
        ax.set_title("Cache Hit / Miss / Unavailable by Request", fontsize=10)
        return []
    for status in CACHE_STATUS_ORDER:
        points = [r for r in usable if r["cache_status"] == status]
        if not points:
            continue
        xs = [r["execution_index"] for r in points]
        ys = [1] * len(points)
        ax.scatter(xs, ys, c=CACHE_STATUS_COLOR.get(status, SECONDARY_INK), s=60, label=status, marker="s")
    ax.set_yticks([])
    ax.set_xlabel("Execution Index")
    ax.set_title("Cache Hit / Miss / Unavailable by Request", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", ncols=2)
    return [1 for _ in usable]


def _draw_cache_friendly_vs_unfriendly(ax_hit_rate, ax_cached_tokens, conditions_by_id, order) -> list[int]:
    """Required chart #4: Cache-Friendly vs Unfriendly Comparison (2 panels:
    observed request hit rate, and mean cached tokens, both by condition)."""
    counts = list(_draw_request_hit_rate_by_condition(ax_hit_rate, conditions_by_id, order))
    counts += _draw_single_metric_bar(ax_cached_tokens, conditions_by_id, order, "cached_tokens", "Mean Cached Tokens by Condition", "Token Sayısı")
    return counts


def _draw_probe_sequence(ax, records: list[dict]) -> list[int]:
    """Required chart #7: Exact-Match Probe Sequence -- only meaningful for a
    `cache_probe` run (every record shares condition_id=repeated_prefix and
    an identical request_hash); plots cached_tokens per sequential probe
    call, colored by cache_status, so a hit appearing on e.g. the 2nd/3rd
    identical call is visible at a glance."""
    usable = [r for r in records if r.get("cache_status")]
    if not usable:
        no_data_text(ax)
        ax.set_title("Exact-Match Probe Sequence", fontsize=10)
        return []
    usable = sorted(usable, key=lambda r: r["execution_index"])
    xs = [r["execution_index"] for r in usable]
    ys = [r.get("cached_tokens") if r.get("cached_tokens") is not None else 0 for r in usable]
    colors = [CACHE_STATUS_COLOR.get(r["cache_status"], SECONDARY_INK) for r in usable]
    ax.plot(xs, ys, color=MUTED_INK, linewidth=1, zorder=1)
    ax.scatter(xs, ys, c=colors, s=80, zorder=2)
    for r in usable:
        if r.get("cached_tokens") is None:
            ax.text(r["execution_index"], 0, "N/A", ha="center", va="bottom", color=MUTED_INK, fontsize=7)
    ax.set_xticks(xs)
    ax.set_xlabel("Probe Request # (sequential, byte-identical requests)")
    ax.set_ylabel("Cached Tokens")
    ax.set_title("Exact-Match Probe Sequence", fontsize=10)
    return [1 for _ in usable]


def _draw_cost_savings(ax, conditions_by_id, order) -> list[int]:
    usable = [cid for cid in order if conditions_by_id.get(cid, {}).get("total_estimated_cache_savings_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Estimated Cost Savings", fontsize=10)
        return []
    values = [conditions_by_id[cid]["total_estimated_cache_savings_usd"] for cid in usable]
    colors = [CONDITION_COLOR.get(cid, SECONDARY_INK) for cid in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Tahmini Tasarruf (USD)")
    ax.set_title("Estimated Cost Savings (counterfactual)", fontsize=10)
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def _draw_cached_tokens_vs_latency(ax, records: list[dict]) -> list[int]:
    usable = [r for r in records if r.get("cached_tokens") is not None and r.get("total_latency_ms") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Cached Tokens vs Latency", fontsize=10)
        return []
    xs = [r["cached_tokens"] for r in usable]
    ys = [r["total_latency_ms"] for r in usable]
    colors = [CONDITION_COLOR.get(r.get("condition_id"), SECONDARY_INK) for r in usable]
    ax.scatter(xs, ys, c=colors, s=28)
    ax.set_xlabel("Cached Tokens")
    ax.set_ylabel("Gecikme (ms)")
    ax.set_title("Cached Tokens vs Latency", fontsize=10)
    return [1 for _ in usable]


def _draw_answer_correctness(ax, conditions_by_id, order) -> list[int]:
    usable = [cid for cid in order if conditions_by_id.get(cid, {}).get("successful_calls", 0) > 0]
    if not usable:
        no_data_text(ax)
        ax.set_title("Answer Correctness by Condition", fontsize=10)
        return []
    statuses = [("correct_count", "Correct"), ("incorrect_count", "Incorrect"), ("format_invalid_count", "Format Invalid"), ("missing_response_count", "Missing")]
    x = list(range(len(usable)))
    bottom = [0.0] * len(usable)
    for field, label in statuses:
        values = [conditions_by_id[cid].get(field, 0) or 0 for cid in usable]
        if all(v == 0 for v in values):
            continue
        ax.bar(x, values, width=0.5, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL.get(cid, cid) for cid in usable])
    ax.set_ylabel("Çağrı Sayısı")
    ax.set_title("Answer Correctness by Condition", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [conditions_by_id[cid].get("successful_calls", 0) for cid in usable]


def generate_cache_charts(run_dir: Path) -> list[Path]:
    """Render every Lab03 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    conditions = (summary or {}).get("conditions", []) if summary else []
    records = (summary or {}).get("measurements", []) if summary else []
    conditions_with_data = [c for c in conditions if c.get("successful_calls", 0) > 0]
    is_probe_run = (summary or {}).get("experiment_mode") == PROBE_EXPERIMENT_MODE

    if not conditions_with_data:
        return []

    order = _ordered_conditions_present(conditions)
    conditions_by_id = {c["condition_id"]: c for c in conditions}
    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. Cached vs uncached input tokens.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cached_vs_uncached_tokens(ax, conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cached_vs_uncached_tokens.png"))

    # 2. Cache Hit / Miss / Unavailable by request (required chart #2;
    # CACHE_USAGE_UNAVAILABLE is always shown distinctly, never as a miss).
    fig, ax = new_figure(figsize=(9, 3.5))
    counts = _draw_cache_status_by_request(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_status_by_request.png"))

    # 3. Token cache ratio by request.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_cache_ratio_by_request(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_ratio_by_request.png"))

    # 4. Cache-friendly vs unfriendly comparison (required chart #4; two
    # panels: observed request hit rate, and mean cached tokens).
    fig, axes = new_figure(figsize=(12, 5), ncols=2)
    counts = _draw_cache_friendly_vs_unfriendly(axes[0], axes[1], conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cache_friendly_vs_unfriendly_comparison.png"))

    if is_probe_run:
        # 7. Exact-Match Probe Sequence -- only produced for a cache_probe
        # run (never fabricated for a regular all/single-condition run,
        # which has no sequential-identical-request data to show).
        fig, ax = new_figure(figsize=(8, 5))
        counts = _draw_probe_sequence(ax, records)
        add_footer_caption(fig, sample_size_caption(counts))
        written.append(save_png(fig, charts_dir / "probe_sequence.png"))

    # 5. Estimated cost by condition (supporting detail alongside required chart #5, savings).
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "mean_estimated_cost_usd", "Estimated Cost by Condition", "Tahmini Maliyet (USD)")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "estimated_cost_by_condition.png"))

    # 5. Estimated input cost savings (required chart #5).
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cost_savings(ax, conditions_by_id, order)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "estimated_cost_savings.png"))

    # Latency by condition (supporting detail alongside required chart #6).
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_single_metric_bar(ax, conditions_by_id, order, "latency_ms", "Latency by Condition", "Gecikme (ms)")
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "latency_by_condition.png"))

    # 6. Cached tokens vs latency (required chart #6).
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_cached_tokens_vs_latency(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "cached_tokens_vs_latency.png"))

    # 8. Summary dashboard (required chart #8).
    fig, axes = new_figure(figsize=(22, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_request_hit_rate_by_condition(axes[0][0], conditions_by_id, order))
    all_counts.extend(_draw_cache_status_by_request(axes[0][1], records))
    all_counts.extend(_draw_cost_savings(axes[0][2], conditions_by_id, order))
    all_counts.extend(_draw_single_metric_bar(axes[0][3], conditions_by_id, order, "latency_ms", "Latency", "Gecikme (ms)"))
    all_counts.extend(_draw_cached_vs_uncached_tokens(axes[1][0], conditions_by_id, order))
    all_counts.extend(_draw_answer_correctness(axes[1][1], conditions_by_id, order))
    all_counts.extend(_draw_cache_ratio_by_request(axes[1][2], records))
    if is_probe_run:
        all_counts.extend(_draw_probe_sequence(axes[1][3], records))
    else:
        no_data_text(axes[1][3], "")
        axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB03 Prompt Caching Summary -- {(summary or {}).get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "summary_dashboard.png"))

    return written
