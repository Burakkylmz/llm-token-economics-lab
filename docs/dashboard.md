# Read-only Streamlit dashboard (MVP)

The dashboard displays **existing saved results** from the 10 training labs.
It does not call Groq (or any other provider), modify experiment results, or
fabricate measurements. No API key is required.

## Run locally

From the repository root, with uv installed:

    uv run --with "streamlit>=1.40,<2" streamlit run dashboard/app.py

This installs Streamlit as a runtime-only extra and leaves the project's
existing locked offline-test dependencies unchanged. On Windows PowerShell the
same command works. Open the local URL displayed by Streamlit (usually
http://localhost:8501).

The dashboard reads only the repository's local results/ directory. Results
are not automatically downloaded from GitHub or any remote service.
If no supported files are present, an empty-state explanation is shown.

For an API-free example, run this command in another terminal:

    uv run lab-cli lab07 all

Then click **Refresh saved results** in the sidebar. Lab07 is a simulator,
so it does not generate actual provider token billing or model-answer quality.

## Features

- Discover runs from existing per-lab experiment directories.
- Normalize approved numeric metrics across the heterogeneous lab schemas.
- Choose from all 10 implemented labs, including those with no saved results.
- Filter by lab, run, model, and strategy/condition.
- Inspect measured requests, coverage, response quality, estimated cost,
  latency and token totals.
- Compare groups descriptively with bar charts.
- Export only approved normalized metrics to CSV.

## Accuracy and privacy

- Missing values remain missing and are never treated as zero.
- Costs are configured estimates, not provider invoices; discounted batch
  estimates are labeled separately from hypothetical standard prices.
- Quality is reported only when a saved measurement explicitly contains
  answer_correct. Structural validation does not become ground-truth accuracy.
- Latency semantics differ between labs (for example, queueing and streaming).
  Do not infer causal differences from cross-lab charts.
- Compare matched workloads within one experiment run whenever possible.
- The loader only scans an explicit allowlist of measurement file names; raw
  prompts, outputs, traces and summaries are neither read nor exported.
- Run this app on a trusted local machine; Streamlit is a local web server.
- Large, malformed and inaccessible files are skipped with warnings.

## Future work

Paired experiment-level comparisons with uncertainty bands, per-lab specialized
charts, dashboard UI integration tests and deployment hardening.

## Lab07 simulator comparison

Lab07's saved measurements contain simulation scenarios (`scenario_id`),
retry strategies (`strategy_id`), virtual wait times, retry counts and
synthetic HTTP 429 responses. The dashboard now groups them by run, scenario
and strategy to show simulated success rate, retry attempts, rate-limit
responses and virtual wait.

Use **Retry simulator** to select exactly one saved Lab07 run and one
scenario at a time. Four two-column charts compare the four strategies,
using short, readable names (No backoff, Exponential, Exponential + jitter,
Rate-limit aware). This avoids long run IDs on the chart axis and prevents
accidentally comparing unlike scenario conditions.

The **Model comparison** tab instead focuses on saved real-model experiments
within a chosen lab and run. The sidebar filters apply to all tabs.

These simulation values are **not** real provider response times,
model-answer accuracy, or API billing records.

## Why a lab may show "No saved results"

The Lab selector lists all ten implemented labs, whether or not they have
been run. A count of zero is not a missing module: it means the selected
lab has no supported measurement files under the local `results/` directory.
Dashboard filters never start an experiment or make a provider API call.

Lab07 is the API-free retry simulator; for the other labs, creating new
provider measurements normally requires deliberately configuring and
authorizing a live run. See [lab instructions](README.md) for each workflow.
