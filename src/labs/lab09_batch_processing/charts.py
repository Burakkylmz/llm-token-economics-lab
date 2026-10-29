"""Chart generation for Lab09 -- Batch Processing.

Reads only `experiment_summary.json` from a completed run directory (same
convention as every other redesigned lab). Any section whose underlying
data is unavailable (most commonly: the Batch job has not completed yet)
is rendered with an explicit "no data" marker -- never a zero-height bar
standing in for missing data (spec Part 13's educational rules).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import CATEGORICAL_DARK, add_footer_caption, new_figure, no_data_text, sample_size_caption, save_png

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"

_CONDITION_COLOR = {"sequential": CATEGORICAL_DARK[0], "concurrent": CATEGORICAL_DARK[2], "batch": CATEGORICAL_DARK[3]}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _draw_realtime_wall_clock(ax, sequential: Optional[dict], concurrent: Optional[dict]) -> list[int]:
    rows = [(name, c) for name, c in (("sequential", sequential), ("concurrent", concurrent)) if c and c.get("batch_wall_clock_ms") is not None]
    if not rows:
        no_data_text(ax)
        ax.set_title("Sequential vs Concurrent -- Wall-Clock Time", fontsize=10)
        return []
    xs = [name for name, _ in rows]
    ys = [c["batch_wall_clock_ms"] for _, c in rows]
    colors = [_CONDITION_COLOR[name] for name in xs]
    ax.bar(xs, ys, color=colors, width=0.5)
    ax.set_ylabel("Batch Wall-Clock Time (ms)")
    ax.set_title("Sequential vs Concurrent -- Wall-Clock Time", fontsize=10)
    return [c["successful_requests"] for _, c in rows]


def _draw_throughput_comparison(ax, sequential: Optional[dict], concurrent: Optional[dict]) -> list[int]:
    rows = [(name, c) for name, c in (("sequential", sequential), ("concurrent", concurrent)) if c and c.get("successful_requests_per_second") is not None]
    if not rows:
        no_data_text(ax)
        ax.set_title("Throughput Comparison", fontsize=10)
        return []
    xs = [name for name, _ in rows]
    ys = [c["successful_requests_per_second"] for _, c in rows]
    colors = [_CONDITION_COLOR[name] for name in xs]
    ax.bar(xs, ys, color=colors, width=0.5)
    ax.set_ylabel("Successful Requests/sec")
    ax.set_title("Throughput Comparison", fontsize=10)
    return [c["successful_requests"] for _, c in rows]


def _draw_latency_distribution(ax, sequential: Optional[dict], concurrent: Optional[dict]) -> list[int]:
    rows = [(name, c) for name, c in (("sequential", sequential), ("concurrent", concurrent)) if c and c.get("request_service_latency_ms", {}).get("mean") is not None]
    if not rows:
        no_data_text(ax)
        ax.set_title("Request Latency Distribution", fontsize=10)
        return []
    xs = list(range(len(rows)))
    means = [c["request_service_latency_ms"]["mean"] for _, c in rows]
    p95s = [c["request_service_latency_ms"].get("p95") for _, c in rows]
    ax.bar(xs, means, color=CATEGORICAL_DARK[4], width=0.5, label="Mean")
    ax.scatter(xs, p95s, color=CATEGORICAL_DARK[7], zorder=3, label="P95", marker="D")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _ in rows])
    ax.set_ylabel("Request Service Latency (ms)")
    ax.set_title("Request Latency Distribution", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [c["successful_requests"] for _, c in rows]


def _draw_token_consumption_breakdown(ax, token_totals: dict) -> list[int]:
    rows = [(name, t) for name, t in token_totals.items() if t and (t.get("prompt_tokens") or t.get("completion_tokens"))]
    if not rows:
        no_data_text(ax)
        ax.set_title("Token Consumption Breakdown", fontsize=10)
        return []
    xs = [name for name, _ in rows]
    prompt = [t.get("prompt_tokens") or 0 for _, t in rows]
    completion = [t.get("completion_tokens") or 0 for _, t in rows]
    colors = [_CONDITION_COLOR.get(name, CATEGORICAL_DARK[5]) for name in xs]
    ax.bar(xs, prompt, width=0.5, label="Prompt Tokens", color=colors, alpha=0.6)
    ax.bar(xs, completion, width=0.5, bottom=prompt, label="Completion Tokens", color=colors)
    ax.set_ylabel("Tokens")
    ax.set_title("Token Consumption Breakdown", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [1 for _ in rows]


def _draw_standard_vs_batch_cost(ax, pricing: Optional[dict]) -> list[int]:
    if not pricing or pricing.get("estimated_standard_cost_usd") is None or pricing.get("estimated_batch_cost_usd") is None:
        no_data_text(ax, "BATCH_PENDING")
        ax.set_title("Standard vs Batch Cost", fontsize=10)
        return []
    xs = ["standard", "batch"]
    ys = [pricing["estimated_standard_cost_usd"], pricing["estimated_batch_cost_usd"]]
    ax.bar(xs, ys, color=[CATEGORICAL_DARK[1], CATEGORICAL_DARK[3]], width=0.5)
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("Standard vs Batch Cost (Estimated)", fontsize=10)
    return [1, 1]


def _draw_batch_request_status(ax, counts: Optional[dict]) -> list[int]:
    if not counts:
        no_data_text(ax, "BATCH_PENDING")
        ax.set_title("Batch Request Status", fontsize=10)
        return []
    labels = ["completed", "failed", "missing", "duplicate", "unexpected"]
    keys = ["completed_requests", "failed_requests", "missing_requests", "duplicate_result_ids", "unexpected_result_ids"]
    values = [counts.get(k) or 0 for k in keys]
    if sum(values) == 0:
        no_data_text(ax, "BATCH_PENDING")
        ax.set_title("Batch Request Status", fontsize=10)
        return []
    ax.bar(labels, values, color=CATEGORICAL_DARK[:5], width=0.6)
    ax.set_ylabel("Request Count")
    ax.set_title("Batch Request Status", fontsize=10)
    return [max(sum(values), 1)]


def _draw_quality_comparison(ax, quality_summaries: list[dict]) -> list[int]:
    usable = [q for q in quality_summaries if q.get("accuracy") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Quality Comparison (Accuracy)", fontsize=10)
        return []
    xs = [q["condition"] for q in usable]
    ys = [q["accuracy"] * 100 for q in usable]
    colors = [_CONDITION_COLOR.get(name, CATEGORICAL_DARK[5]) for name in xs]
    ax.bar(xs, ys, color=colors, width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Quality Comparison (Accuracy)", fontsize=10)
    return [q["total_evaluated"] for q in usable]


def generate_batch_charts(run_dir: Path) -> list[Path]:
    """Render every Lab09 chart for one run, into `<run_dir>/charts/`.
    Returns the list of PNG paths written. Never raises on sparse/partial
    data (e.g. Batch not yet submitted) -- renders those panels as
    explicit BATCH_PENDING markers instead."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    sequential = summary.get("sequential_summary")
    concurrent = summary.get("concurrent_summary")
    pricing = summary.get("pricing_comparison")
    batch_counts = summary.get("batch_counts")
    quality_summaries = summary.get("quality_summaries", [])
    token_totals = summary.get("token_totals", {})

    if not sequential and not concurrent:
        return []

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_realtime_wall_clock(ax, sequential, concurrent)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "sequential_vs_concurrent_batch_time.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_throughput_comparison(ax, sequential, concurrent)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "throughput_comparison.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_latency_distribution(ax, sequential, concurrent)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "request_latency_distribution.png"))

    fig, ax = new_figure(figsize=(7, 5))
    _draw_token_consumption_breakdown(ax, token_totals)
    written.append(save_png(fig, charts_dir / "token_consumption_breakdown.png"))

    fig, ax = new_figure(figsize=(7, 5))
    _draw_standard_vs_batch_cost(ax, pricing)
    written.append(save_png(fig, charts_dir / "standard_vs_batch_cost.png"))

    fig, ax = new_figure(figsize=(7, 5))
    _draw_batch_request_status(ax, batch_counts)
    written.append(save_png(fig, charts_dir / "batch_request_status.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_quality_comparison(ax, quality_summaries)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "quality_comparison.png"))

    fig, axes = new_figure(figsize=(18, 9), nrows=2, ncols=4)
    _draw_realtime_wall_clock(axes[0][0], sequential, concurrent)
    _draw_throughput_comparison(axes[0][1], sequential, concurrent)
    _draw_latency_distribution(axes[0][2], sequential, concurrent)
    _draw_token_consumption_breakdown(axes[0][3], token_totals)
    _draw_standard_vs_batch_cost(axes[1][0], pricing)
    _draw_batch_request_status(axes[1][1], batch_counts)
    _draw_quality_comparison(axes[1][2], quality_summaries)
    no_data_text(axes[1][3], "")
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB09 Batch Processing Dashboard -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    written.append(save_png(fig, charts_dir / "batch_processing_dashboard.png"))

    return written
