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

Then click **Refresh results** in the sidebar. Lab07 is a simulator,
so it does not generate actual provider token billing or model-answer quality.

## Docker Compose (no local Python needed)

From the repository root, with Docker Desktop/Engine and Compose:

```powershell
docker compose build lab-runner dashboard
docker compose up -d dashboard
# Open http://localhost:8502 (Docker default host port)
docker compose run --rm lab-runner lab07 all
# In the sidebar, click "Refresh results".
docker compose ps
docker compose stop dashboard
```

The dashboard is built from the dedicated `Dockerfile.dashboard`,
separately from the existing CLI `Dockerfile`. Streamlit is installed
**during the image build** at a pinned direct version, rather than being
downloaded on every startup; the project's core runtime dependencies
come from the committed `uv.lock`. The Streamlit dependency graph is
not yet separately transitive-locked, so rebuilding the image at a later
date can resolve newer indirect dependencies. See the Docker CI smoke
test for verified build/runtime behavior.

**Security boundaries:** the dashboard receives no `GROQ_API_KEY`
environment variable, runs as an unprivileged user, mounts the local
`results/` directory read-only, has a read-only container filesystem
with ephemeral `/tmp`, and publishes TCP port **8502** on `127.0.0.1`
only by default. Streamlit still listens on internal container port
**8501**; only the host mapping is configurable. This is a local development tool, not an authenticated internet
service. CLI experiments can still write to `results/` via their
separate container and need explicit live opt-in.

The Compose healthcheck calls Streamlit's local
`/_stcore/health` endpoint. Run `docker compose ps` to see the
healthy status or `docker compose logs dashboard` for details.

If Docker reports that host port **8502** is already allocated, keep
other containers running and override just this dashboard's host port:

```powershell
$env:DASHBOARD_PORT = "8503"
docker compose up -d --force-recreate dashboard
# Open http://localhost:8503
```

To return to port 8502 in PowerShell, run
`Remove-Item Env:DASHBOARD_PORT` and recreate the container.
A locally launched Streamlit session can continue using port 8501
without conflicting with this Docker service. On Windows, Docker
Desktop should be running Linux containers.

### API-free smoke test

A dedicated GitHub Actions workflow builds both images, executes Lab07
in the CLI container, starts the dashboard without provider credentials,
checks Streamlit's health endpoint and verifies that the read-only
dashboard sees the generated Lab07 measurements. It never issues live
Groq requests.

## Features

- Discover runs from existing per-lab experiment directories.
- Normalize approved numeric metrics across the heterogeneous lab schemas.
- Choose from all 10 implemented labs, including those with no saved results.
- Filter by lab, run, model, and strategy/condition.
- Inspect measured requests, coverage, response quality, estimated cost,
  latency and token totals.
- Compare groups descriptively with bar charts.
- Export only approved normalized metrics to CSV.

For detailed aggregation formulas and source-field definitions, see
[Benchmark Methodology](benchmark-methodology.md) and
[Data Dictionary](data-dictionary.md). The dashboard's own overview
summaries may use different grouping/output fields from the offline
benchmark reporter; do not silently treat them as identical statistics.

## Advanced analytics tab

This tab uses the same **PR #15 comparison engine** as the CLI, not a
separate implementation of statistical formulas.

1. Open the Dashboard and choose **Advanced analytics**.
2. Select a **Lab and saved run**, then one **Scenario**.
3. Select a **Matched measurement context**: one experiment family,
   measurement source filename, scenario and cost basis. Contexts with
   fewer than two distinct conditions are not offered.
4. Select a **Baseline condition** and a different **Candidate
   condition**; both labels also show the saved model identifier.
5. If model identifiers differ, explicitly acknowledge the exploratory
   cross-model comparison. This consent is tied to the selected pair.
6. The Dashboard runs the **offline experiment validator** before
   presenting any comparison. Validation errors block the result.
   Unversioned input warnings are displayed rather than hidden.
7. Inspect the **measured values and differences** table and choose
   a metric to display in the side-by-side chart. Download a JSON
   or Markdown summary if needed.

**All sidebar lab/run/model/condition/scenario filters continue to
apply.** If no matching pairs are available, change the filters or
generate a new saved experiment first. No option runs an experiment
from the Dashboard.

The table includes baseline/candidate recorded values, separate
**observed-n** denominators, absolute delta, relative delta percentage,
and percentage-point delta for success/answer accuracy. A recorded
zero is not an unavailable value. Relative percentage is undefined
when the baseline is zero, and is displayed as unavailable.

The charts display **two recorded group values**, not confidence
intervals or inference about causal effects. For Lab07 these are
**synthetic client outcomes**, retries, rate-limit counts and virtual
seconds, never real API latency, provider token billing or response
quality. Do not call the lowest-wait strategy the best overall without
reviewing reliability and the experimental design.

The comparison UI imports only approved fields from the existing
local measurement loader. JSON and Markdown downloads are generated
in memory; no source measurements are altered. The Docker service
continues to have a read-only results mount and no API key.

See [Comparison Guide](experiment-comparison.md) for the exact
pair-selection and delta rules, including limitations.

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

For optional offline validation of saved data, see the
[Experiment Validation Guide](experiment-validation.md). The Dashboard
currently does **not** block display of runs that have not been validated;
run the validator independently before relying on measurements.

## Future work

Paired experiment-level comparisons with uncertainty bands, per-lab specialized
charts, dashboard UI integration tests, full transitive dependency locking
for the dashboard image, and authenticated deployment hardening.

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
For a complete quick start and troubleshooting table, see the
[project README](../README.md).
