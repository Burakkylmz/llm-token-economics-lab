"""Chart generation for Lab08's sequential-vs-concurrent async execution
experiment.

Reads only `experiment_summary.json` from a completed run directory (same
convention as every other redesigned lab). Charts 6/7 (in-flight timeline,
Gantt) pick the LARGEST concurrency level actually present as their
representative batch -- the one condition where queueing/overlap behavior
is most visible -- rather than fabricating a chart for a level that wasn't
run.
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

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_conditions(conditions: list[dict]) -> list[dict]:
    return sorted(conditions, key=lambda c: c.get("max_concurrent", 0))


def _usable(conditions: list[dict]) -> list[dict]:
    return [c for c in conditions if c.get("total_requests", 0) > 0]


def _draw_throughput_vs_concurrency(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable(conditions) if c.get("successful_requests_per_second") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Throughput vs Concurrency", fontsize=10)
        return []
    xs = [c["max_concurrent"] for c in usable]
    ys = [c["successful_requests_per_second"] for c in usable]
    ax.plot(xs, ys, marker="o", color=CATEGORICAL_DARK[2])
    ax.set_xticks(xs)
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Successful Requests/sec")
    ax.set_title("Throughput vs Concurrency", fontsize=10)
    return [c["successful_requests"] for c in usable]


def _draw_batch_completion_time(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable(conditions) if c.get("batch_wall_clock_ms") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Batch Completion Time", fontsize=10)
        return []
    xs = [str(c["max_concurrent"]) for c in usable]
    ys = [c["batch_wall_clock_ms"] for c in usable]
    ax.bar(xs, ys, color=CATEGORICAL_DARK[0], width=0.5)
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Batch Wall-Clock Time (ms)")
    ax.set_title("Batch Completion Time", fontsize=10)
    return [c["total_requests"] for c in usable]


def _draw_service_latency_distribution(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable(conditions) if c.get("request_service_latency_ms", {}).get("mean") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Service Latency Distribution", fontsize=10)
        return []
    xs = list(range(len(usable)))
    means = [c["request_service_latency_ms"]["mean"] for c in usable]
    p95s = [c["request_service_latency_ms"].get("p95") for c in usable]
    ax.bar(xs, means, color=CATEGORICAL_DARK[4], width=0.5, label="Mean")
    ax.scatter(xs, p95s, color=CATEGORICAL_DARK[7], zorder=3, label="P95", marker="D")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c["max_concurrent"]) for c in usable])
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Request Service Latency (ms)")
    ax.set_title("Service Latency Distribution", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [c["successful_requests"] for c in usable]


def _draw_queue_waiting_time(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable(conditions) if c.get("queue_wait_ms", {}).get("mean") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Queue Waiting Time", fontsize=10)
        return []
    xs = [str(c["max_concurrent"]) for c in usable]
    ys = [c["queue_wait_ms"]["mean"] for c in usable]
    ax.bar(xs, ys, color=CATEGORICAL_DARK[1], width=0.5)
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Mean Queue Wait (ms)")
    ax.set_title("Queue Waiting Time", fontsize=10)
    return [c["total_requests"] for c in usable]


def _draw_end_to_end_latency(ax, conditions: list[dict]) -> list[int]:
    usable = [
        c for c in _usable(conditions)
        if c.get("queue_wait_ms", {}).get("mean") is not None and c.get("request_service_latency_ms", {}).get("mean") is not None
    ]
    if not usable:
        no_data_text(ax)
        ax.set_title("Task End-to-End Latency", fontsize=10)
        return []
    xs = [str(c["max_concurrent"]) for c in usable]
    queue = [c["queue_wait_ms"]["mean"] for c in usable]
    service = [c["request_service_latency_ms"]["mean"] for c in usable]
    ax.bar(xs, queue, width=0.5, label="Queue Wait", color=CATEGORICAL_DARK[1])
    ax.bar(xs, service, width=0.5, bottom=queue, label="Service Latency", color=CATEGORICAL_DARK[4])
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Task End-to-End Latency (Queue + Service)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return [c["total_requests"] for c in usable]


def _representative_level(conditions: list[dict]) -> Optional[int]:
    usable = _usable(conditions)
    if not usable:
        return None
    return max(c["max_concurrent"] for c in usable)


def _draw_in_flight_timeline(ax, events: list[dict], level: Optional[int]) -> list[int]:
    if level is None:
        no_data_text(ax)
        ax.set_title("Observed In-Flight Requests Timeline", fontsize=10)
        return []
    level_events = sorted(
        (e for e in events if e.get("max_concurrent") == level and e.get("event_type") in ("REQUEST_STARTED", "REQUEST_COMPLETED")),
        key=lambda e: e["relative_ms"],
    )
    if not level_events:
        no_data_text(ax)
        ax.set_title("Observed In-Flight Requests Timeline", fontsize=10)
        return []
    xs, ys = [0.0], [0]
    current = 0
    for e in level_events:
        current += 1 if e["event_type"] == "REQUEST_STARTED" else -1
        xs.append(e["relative_ms"])
        ys.append(current)
    ax.step(xs, ys, where="post", color=CATEGORICAL_DARK[2])
    ax.axhline(level, color=MUTED_INK, linestyle="--", linewidth=1, label=f"Configured limit ({level})")
    ax.set_xlabel("Elapsed Time (ms)")
    ax.set_ylabel("In-Flight Requests")
    ax.set_title(f"Observed In-Flight Requests Timeline (concurrency={level})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    return [1]


def _draw_gantt(ax, events: list[dict], level: Optional[int]) -> list[int]:
    if level is None:
        no_data_text(ax)
        ax.set_title("Request Execution Gantt Chart", fontsize=10)
        return []
    by_task: dict[str, dict] = {}
    for e in events:
        if e.get("max_concurrent") != level:
            continue
        by_task.setdefault(e["task_id"], {})[e["event_type"]] = e["relative_ms"]
    task_ids = sorted(by_task.keys())
    if not task_ids:
        no_data_text(ax)
        ax.set_title("Request Execution Gantt Chart", fontsize=10)
        return []
    for i, task_id in enumerate(task_ids):
        spans = by_task[task_id]
        queued = spans.get("QUEUED")
        permit = spans.get("PERMIT_ACQUIRED")
        started = spans.get("REQUEST_STARTED")
        completed = spans.get("REQUEST_COMPLETED")
        if queued is not None and permit is not None:
            ax.barh(i, permit - queued, left=queued, color=MUTED_INK, height=0.5, label="Queue Wait" if i == 0 else None)
        if started is not None and completed is not None:
            ax.barh(i, completed - started, left=started, color=CATEGORICAL_DARK[2], height=0.5, label="Service" if i == 0 else None)
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=7)
    ax.set_xlabel("Elapsed Time (ms)")
    ax.set_title(f"Request Execution Gantt Chart (concurrency={level})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    return [len(task_ids)]


def _draw_token_cost_comparison(ax, conditions: list[dict]) -> list[int]:
    usable = [c for c in _usable(conditions) if c.get("total_estimated_cost_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Token & Cost Comparison", fontsize=10)
        return []
    xs = [str(c["max_concurrent"]) for c in usable]
    ys = [c["total_estimated_cost_usd"] for c in usable]
    ax.bar(xs, ys, color=CATEGORICAL_DARK[3], width=0.5)
    ax.set_xlabel("Configured Concurrency")
    ax.set_ylabel("Total Estimated Cost (USD)")
    ax.set_title("Token & Cost Comparison", fontsize=10)
    return [c["successful_requests"] for c in usable]


def generate_concurrency_charts(run_dir: Path) -> list[Path]:
    """Render every Lab08 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    conditions = _ordered_conditions(summary.get("conditions", []))
    if not _usable(conditions):
        return []
    events = summary.get("timeline_events", [])
    representative_level = _representative_level(conditions)

    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_throughput_vs_concurrency(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "throughput_vs_concurrency.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_batch_completion_time(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "batch_completion_time.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_service_latency_distribution(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "service_latency_distribution.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_queue_waiting_time(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "queue_waiting_time.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_end_to_end_latency(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "task_end_to_end_latency.png"))

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_in_flight_timeline(ax, events, representative_level)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "observed_in_flight_timeline.png"))

    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_gantt(ax, events, representative_level)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "request_execution_gantt.png"))

    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_token_cost_comparison(ax, conditions)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "token_cost_comparison.png"))

    fig, axes = new_figure(figsize=(20, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_throughput_vs_concurrency(axes[0][0], conditions))
    all_counts.extend(_draw_batch_completion_time(axes[0][1], conditions))
    all_counts.extend(_draw_service_latency_distribution(axes[0][2], conditions))
    all_counts.extend(_draw_end_to_end_latency(axes[0][3], conditions))
    all_counts.extend(_draw_queue_waiting_time(axes[1][0], conditions))
    all_counts.extend(_draw_token_cost_comparison(axes[1][1], conditions))
    all_counts.extend(_draw_in_flight_timeline(axes[1][2], events, representative_level))
    no_data_text(axes[1][3], "")
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB08 Performance & Reliability Dashboard -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "performance_reliability_dashboard.png"))

    return written
