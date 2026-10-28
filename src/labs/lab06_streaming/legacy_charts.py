"""Chart generation from Lab06's pre-existing flat result file
(`results/lab06_streaming.json`), for the "Legacy Backfill" requirement.

Only the flat `.json` file is read, never the accompanying `.csv` -- same
rule as every other redesigned lab's legacy_charts module.

The legacy schema's `ttft_ms` is the OLD code's first-SSE-event proxy, not
a verified provider token timing -- every legacy chart/caption says so
explicitly. The legacy runs also never guaranteed real request-pairing
(non_streaming and streaming calls ran in two separate sequential blocks,
not interleaved) and never requested/received streaming token usage, so
this module reconstructs only an INDEX-BASED best-effort pairing (i-th
non_streaming call with i-th streaming call, in file order) -- captioned
as approximate, never presented as this redesign's verified paired
comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lab_core.charts import add_footer_caption, new_figure, no_data_text, save_png

LEGACY_JSON_FILENAME = "lab06_streaming.json"
LEGACY_CHARTS_SUBDIR = Path("lab06_streaming") / "_legacy_backfill" / "charts"

LEGACY_CAPTION = (
    "Legacy backfill: pre-dates per-run isolation, paired/interleaved execution, and streaming "
    "token usage support. Pairing is an index-based best-effort reconstruction (i-th non_streaming "
    "call with i-th streaming call), not a verified request-equivalence pairing. 'ttft_ms' here is "
    "the old code's first-SSE-event proxy, not a verified provider token timing."
)


def _load_legacy_records(results_dir: Path) -> Optional[list[dict]]:
    path = results_dir / LEGACY_JSON_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records if records else None


def generate_legacy_streaming_charts(results_dir: Path) -> list[Path]:
    """Render whatever charts are reconstructable from the legacy flat
    `results/lab06_streaming.json` file, into
    `results/lab06_streaming/_legacy_backfill/charts/`. Never reads or
    modifies the legacy `.json`/`.csv` files. Returns an empty list if the
    legacy file is absent/empty or has no recognizable Lab06 records."""
    records = _load_legacy_records(results_dir)
    if records is None:
        return []

    non_streaming = [r for r in records if r.get("metadata", {}).get("experiment_id") == "lab06_non_streaming"]
    streaming = [r for r in records if r.get("metadata", {}).get("experiment_id") == "lab06_streaming"]
    if not non_streaming and not streaming:
        return []

    charts_dir = results_dir / LEGACY_CHARTS_SUBDIR
    written: list[Path] = []
    pair_count = min(len(non_streaming), len(streaming))

    fig, ax = new_figure(figsize=(8, 5))
    if pair_count:
        pair_indices = list(range(1, pair_count + 1))
        non_streaming_latency = [non_streaming[i].get("performance", {}).get("client_elapsed_ms") for i in range(pair_count)]
        streaming_first_visible = [streaming[i].get("performance", {}).get("first_visible_content_ms") for i in range(pair_count)]
        ax.plot(pair_indices, non_streaming_latency, marker="o", label="Non-Streaming Full Response")
        ax.plot(pair_indices, streaming_first_visible, marker="o", label="Streaming First Visible Content (legacy)")
        ax.set_xticks(pair_indices)
        ax.set_xlabel("Index-Based Pair (best-effort)")
        ax.set_ylabel("ms")
        ax.legend(fontsize=7, loc="upper right")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Perceived Latency Comparison (legacy, approximate pairing)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "perceived_latency_comparison.png"))

    fig, ax = new_figure(figsize=(8, 5))
    if pair_count:
        pair_indices = list(range(1, pair_count + 1))
        non_streaming_latency = [non_streaming[i].get("performance", {}).get("client_elapsed_ms") for i in range(pair_count)]
        streaming_latency = [streaming[i].get("performance", {}).get("client_elapsed_ms") for i in range(pair_count)]
        ax.plot(pair_indices, non_streaming_latency, marker="o", label="Non-Streaming Total")
        ax.plot(pair_indices, streaming_latency, marker="o", label="Streaming Total")
        ax.set_xticks(pair_indices)
        ax.set_xlabel("Index-Based Pair (best-effort)")
        ax.set_ylabel("ms")
        ax.legend(fontsize=7, loc="upper right")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Total Latency Comparison (legacy, approximate pairing)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "total_latency_comparison.png"))

    fig, ax = new_figure(figsize=(7, 5))
    rates = [
        r.get("performance", {}).get("output_tokens_per_second")
        for r in non_streaming
        if r.get("performance", {}).get("output_tokens_per_second") is not None
    ]
    if rates:
        ax.bar(["non_streaming"], [sum(rates) / len(rates)], width=0.4)
        ax.set_ylabel("Effective Completion Tokens/sec")
    else:
        no_data_text(ax, "Veri yok (N/A)")
    ax.set_title("Effective Token Throughput (legacy -- streaming usage never available)", fontsize=9)
    add_footer_caption(fig, LEGACY_CAPTION)
    written.append(save_png(fig, charts_dir / "effective_token_throughput.png"))

    return written
