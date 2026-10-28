"""Chart generation for Lab06's streaming/non-streaming perceived-latency
experiment.

Reads only `experiment_summary.json` from a completed run directory (same
convention as every other redesigned lab), and renders PNGs into a
`charts/` subdirectory next to it. Same data-integrity rules: a pair
missing either condition's successful measurement is never charted with a
fabricated placeholder; negative perceived-latency-gain pairs are drawn
exactly as measured, never clamped to zero.
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

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
CHARTS_DIRNAME = "charts"

CONDITION_COLOR = {"non_streaming": CATEGORICAL_DARK[0], "streaming": CATEGORICAL_DARK[2]}
CONDITION_LABEL = {"non_streaming": "Non-Streaming", "streaming": "Streaming"}
CONDITION_ORDER = ("non_streaming", "streaming")


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _conditions_by_id(conditions: list[dict]) -> dict[str, dict]:
    return {c["condition"]: c for c in conditions}


def _usable_pairs(comparisons: list[dict]) -> list[dict]:
    return comparisons  # every entry in task_comparisons is already a real, paired comparison


def _draw_first_event_vs_first_visible(ax, comparisons: list[dict]) -> list[int]:
    usable = [c for c in comparisons if c.get("streaming_first_visible_content_ms") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("First Stream Event vs First Visible Content", fontsize=10)
        return []
    pair_ids = [c["pair_id"] for c in usable]
    first_visible = [c["streaming_first_visible_content_ms"] for c in usable]
    ax.plot(pair_ids, first_visible, marker="o", color=CATEGORICAL_DARK[2], label="First Visible Content")
    ax.set_xticks(pair_ids)
    ax.set_xlabel("Pair")
    ax.set_ylabel("ms")
    ax.set_title("First Stream Event vs First Visible Content", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [1 for _ in usable]


def _draw_perceived_latency_comparison(ax, comparisons: list[dict]) -> list[int]:
    usable = [c for c in comparisons if c.get("non_streaming_total_latency_ms") is not None and c.get("streaming_first_visible_content_ms") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Perceived Latency Comparison", fontsize=10)
        return []
    pair_ids = [c["pair_id"] for c in usable]
    ax.plot(pair_ids, [c["non_streaming_total_latency_ms"] for c in usable], marker="o", color=CATEGORICAL_DARK[0], label="Non-Streaming Full Response")
    ax.plot(pair_ids, [c["streaming_first_visible_content_ms"] for c in usable], marker="o", color=CATEGORICAL_DARK[2], label="Streaming First Visible Content")
    ax.set_xticks(pair_ids)
    ax.set_xlabel("Pair")
    ax.set_ylabel("ms")
    ax.set_title("Perceived Latency Comparison", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [1 for _ in usable]


def _draw_total_latency_comparison(ax, comparisons: list[dict]) -> list[int]:
    usable = [c for c in comparisons if c.get("non_streaming_total_latency_ms") is not None and c.get("streaming_total_latency_ms") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Total Latency Comparison", fontsize=10)
        return []
    pair_ids = [c["pair_id"] for c in usable]
    ax.plot(pair_ids, [c["non_streaming_total_latency_ms"] for c in usable], marker="o", color=CATEGORICAL_DARK[0], label="Non-Streaming Total")
    ax.plot(pair_ids, [c["streaming_total_latency_ms"] for c in usable], marker="o", color=CATEGORICAL_DARK[2], label="Streaming Total")
    ax.set_xticks(pair_ids)
    ax.set_xlabel("Pair")
    ax.set_ylabel("ms")
    ax.set_title("Total Latency Comparison", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return [1 for _ in usable]


def _draw_paired_perceived_gain(ax, comparisons: list[dict]) -> list[int]:
    usable = [c for c in comparisons if c.get("perceived_latency_gain_ms") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A)")
        ax.set_title("Paired Perceived Latency Gain", fontsize=10)
        return []
    pair_ids = [c["pair_id"] for c in usable]
    gains = [c["perceived_latency_gain_ms"] for c in usable]
    colors = [CATEGORICAL_DARK[2] if g >= 0 else CATEGORICAL_DARK[7] for g in gains]
    ax.bar(pair_ids, gains, color=colors, width=0.5)
    ax.axhline(0, color=MUTED_INK, linewidth=1)
    ax.set_xticks(pair_ids)
    ax.set_xlabel("Pair")
    ax.set_ylabel("Perceived Latency Gain (ms)")
    ax.set_title("Paired Perceived Latency Gain", fontsize=10)
    return [1 for _ in usable]


def _draw_event_to_visible_gap(ax, records: list[dict]) -> list[int]:
    usable = [r for r in records if r.get("condition") == "streaming" and r.get("first_event_to_visible_gap_ms") is not None]
    if not usable:
        no_data_text(ax)
        ax.set_title("Event-to-Visible Gap", fontsize=10)
        return []
    pair_ids = [r["pair_id"] for r in usable]
    gaps = [r["first_event_to_visible_gap_ms"] for r in usable]
    ax.bar(pair_ids, gaps, color=CATEGORICAL_DARK[4], width=0.5)
    ax.set_xticks(pair_ids)
    ax.set_xlabel("Pair")
    ax.set_ylabel("First Event -> First Visible Content (ms)")
    ax.set_title("Event-to-Visible Gap", fontsize=10)
    return [1 for _ in usable]


def _draw_content_arrival_timeline(ax, events: list[dict]) -> list[int]:
    if not events:
        no_data_text(ax, "Veri yok: stream_events kaydedilmedi (record_event_timeline=false) veya hiç başarılı streaming çağrısı yok")
        ax.set_title("Streaming Content Arrival Timeline", fontsize=10)
        return []
    by_call: dict[int, list[dict]] = {}
    for e in events:
        by_call.setdefault(e["execution_index"], []).append(e)
    usable_calls = sorted(by_call.keys())[:5]  # cap drawn series to keep the legend readable
    for i, execution_index in enumerate(usable_calls):
        call_events = sorted(by_call[execution_index], key=lambda e: e["relative_arrival_ms"])
        xs = [e["relative_arrival_ms"] for e in call_events]
        ys = [e["cumulative_visible_character_count"] for e in call_events]
        ax.plot(xs, ys, label=f"call #{execution_index}", color=CATEGORICAL_DARK[i % len(CATEGORICAL_DARK)], linewidth=1.5)
    ax.set_xlabel("Elapsed Time (ms)")
    ax.set_ylabel("Cumulative Visible Characters")
    ax.set_title("Streaming Content Arrival Timeline", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    return [1 for _ in usable_calls]


def _draw_effective_throughput(ax, conditions_by_id: dict, records: list[dict]) -> list[int]:
    usable_conditions = [c for c in CONDITION_ORDER if conditions_by_id.get(c, {}).get("usage_available_count", 0) > 0]
    if not usable_conditions:
        no_data_text(ax, "Veri yok: usage hiçbir koşulda mevcut değil")
        ax.set_title("Effective Token Throughput", fontsize=10)
        return []
    values = []
    for c in usable_conditions:
        rates = [r["effective_completion_tokens_per_second"] for r in records if r.get("condition") == c and r.get("effective_completion_tokens_per_second") is not None]
        values.append((sum(rates) / len(rates)) if rates else 0)
    colors = [CONDITION_COLOR[c] for c in usable_conditions]
    ax.bar(range(len(usable_conditions)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable_conditions)))
    ax.set_xticklabels([CONDITION_LABEL[c] for c in usable_conditions])
    ax.set_ylabel("Effective Completion Tokens/sec")
    ax.set_title("Effective Token Throughput", fontsize=10)
    return [conditions_by_id[c]["usage_available_count"] for c in usable_conditions]


def _draw_token_cost_comparison(ax, conditions_by_id: dict) -> list[int]:
    usable = [c for c in CONDITION_ORDER if conditions_by_id.get(c, {}).get("total_estimated_cost_usd") is not None]
    if not usable:
        no_data_text(ax, "Veri yok (N/A) -- streaming usage unavailable olabilir")
        ax.set_title("Token & Cost Comparison", fontsize=10)
        return []
    values = [conditions_by_id[c]["total_estimated_cost_usd"] for c in usable]
    colors = [CONDITION_COLOR[c] for c in usable]
    ax.bar(range(len(usable)), values, color=colors, width=0.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([CONDITION_LABEL[c] for c in usable])
    ax.set_ylabel("Total Estimated Cost (USD)")
    ax.set_title("Token & Cost Comparison", fontsize=10)
    return [conditions_by_id[c]["successful_calls"] for c in usable]


def generate_streaming_charts(run_dir: Path) -> list[Path]:
    """Render every Lab06 chart for one completed run, into
    `<run_dir>/charts/`. Returns the list of PNG paths written (empty if
    there is no usable data at all -- never raises on sparse/empty results)."""
    summary = _load_json(run_dir / EXPERIMENT_SUMMARY_FILENAME)
    if not summary:
        return []

    conditions = summary.get("conditions", [])
    conditions_with_data = [c for c in conditions if c.get("successful_calls", 0) > 0]
    if not conditions_with_data:
        return []

    conditions_by_id = _conditions_by_id(conditions)
    comparisons = summary.get("pair_comparisons", [])
    records = summary.get("measurements", [])
    events = summary.get("stream_events", [])
    charts_dir = run_dir / CHARTS_DIRNAME
    written: list[Path] = []

    # 1. First Stream Event vs First Visible Content.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_first_event_vs_first_visible(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "first_event_vs_first_visible.png"))

    # 2. Perceived Latency Comparison.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_perceived_latency_comparison(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "perceived_latency_comparison.png"))

    # 3. Total Latency Comparison.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_total_latency_comparison(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "total_latency_comparison.png"))

    # 4. Paired Perceived Latency Gain.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_paired_perceived_gain(ax, comparisons)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "paired_perceived_latency_gain.png"))

    # 5. Event-to-Visible Gap.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_event_to_visible_gap(ax, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "event_to_visible_gap.png"))

    # 6. Streaming Content Arrival Timeline.
    fig, ax = new_figure(figsize=(8, 5))
    counts = _draw_content_arrival_timeline(ax, events)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "streaming_content_arrival_timeline.png"))

    # 7. Effective Token Throughput.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_effective_throughput(ax, conditions_by_id, records)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "effective_token_throughput.png"))

    # 8. Token & Cost Comparison.
    fig, ax = new_figure(figsize=(7, 5))
    counts = _draw_token_cost_comparison(ax, conditions_by_id)
    add_footer_caption(fig, sample_size_caption(counts))
    written.append(save_png(fig, charts_dir / "token_cost_comparison.png"))

    # 9. Streaming Performance Dashboard.
    fig, axes = new_figure(figsize=(20, 9), nrows=2, ncols=4)
    all_counts: list[int] = []
    all_counts.extend(_draw_perceived_latency_comparison(axes[0][0], comparisons))
    all_counts.extend(_draw_total_latency_comparison(axes[0][1], comparisons))
    all_counts.extend(_draw_paired_perceived_gain(axes[0][2], comparisons))
    all_counts.extend(_draw_event_to_visible_gap(axes[0][3], records))
    all_counts.extend(_draw_content_arrival_timeline(axes[1][0], events))
    all_counts.extend(_draw_effective_throughput(axes[1][1], conditions_by_id, records))
    all_counts.extend(_draw_token_cost_comparison(axes[1][2], conditions_by_id))
    no_data_text(axes[1][3], "")
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"LAB06 Streaming Performance Summary -- {summary.get('experiment_run_id', run_dir.name)}",
        color="#f0efec", fontsize=13, y=1.02,
    )
    add_footer_caption(fig, sample_size_caption(all_counts))
    written.append(save_png(fig, charts_dir / "streaming_performance_dashboard.png"))

    return written
