# Advanced Experiment Comparison Engine

The read-only comparison engine compares **exactly two saved condition
groups** within the same laboratory **and the same run**. It reuses the
existing normalized measurement reader, benchmark statistic aggregation and
offline validation engine; it does not start experiments, call Groq, or
read raw prompts/model completions.

This tool produces **descriptive group differences**, not paired per-request
effects, significance tests or a recommendation of the "best" strategy.

## Interactive Streamlit visualization (PR #16)

The [Advanced analytics Dashboard tab](dashboard.md#advanced-analytics-tab)
offers the **same comparison engine** through Streamlit, with scoped
lab/run/scenario/context selectors, saved-data validation, a comparison
table with individual observed-n denominators, a side-by-side metric
chart and in-memory JSON/Markdown downloads. This does **not** alter
the CLI contract or run new experiments. Model differences still
require explicit acknowledgement.

## Windows quick start — your existing Lab07 run

Your `validation-demo-01` run was successfully verified against normalized
schema v1. Compare two of its retry strategies in PowerShell:

```powershell
git pull origin master
uv sync --frozen --group dev
uv run python -m lab_core.experiment_comparison --lab lab07 --run-id validation-demo-01 --scenario transient_failures --baseline-condition no_backoff --candidate-condition rate_limit_aware --format md --output results/comparisons/validation-demo-01.md
```

You can compare a different pair by replacing the two condition names.
The Lab07 strategies are `no_backoff`, `exponential_backoff`,
`exponential_backoff_with_jitter` and `rate_limit_aware`.
Use an existing run ID; comparison never creates a new run.

To write machine-readable JSON instead:

```powershell
uv run python -m lab_core.experiment_comparison --lab lab07 --run-id validation-demo-01 --scenario transient_failures --baseline-condition no_backoff --candidate-condition exponential_backoff --format json --output results/comparisons/validation-demo-01.json
```

To run with Docker, after a Lab07 run has generated local results:

```powershell
docker compose run --rm --entrypoint uv lab-runner run python -m lab_core.experiment_comparison --lab lab07 --run-id YOUR_RUN_ID --scenario transient_failures --baseline-condition no_backoff --candidate-condition rate_limit_aware --format json --output results/comparisons/docker.json
```

The CLI returns a nonzero argument error for missing/ambiguous groups,
incompatible source contexts or a run rejected by the existing validator.
It never alters saved measurements; `--output` must be outside the input
run directory. Omit `--output` to print the result to your terminal.

## Method and output contract

1. `scan_results` reads the documented allowlisted metric files and
   normalizes only approved scalar fields.
2. `build_benchmark` selects the exact `--lab`/`--run-id`, and
   `--scenario` where applicable. Lab07 requires a scenario when a
   run has more than one.
3. Groups are selected by the distinct `--baseline-condition` and
   `--candidate-condition`. If a condition matches multiple groups,
   you **must** disambiguate with `--experiment`, `--source`,
   `--model`, or `--cost-basis`.
4. Selected groups must match on `experiment`, `source`, `scenario`,
   `cost_basis` and simulation status. They must normally match on
   `model` as well. Cross-model comparison requires the explicit
   `--allow-model-difference` flag and remains **exploratory**.
5. The **exact saved run** is checked using the offline validation
   engine. Validation errors block the comparison. Legacy unversioned
   runs are permitted with a warning; unsupported schema versions fail.
6. JSON and Markdown contain only summary metrics and approved group
   labels; no user prompt or response content is exported.

`comparison_report_version = 1` versions this **comparison output**,
not the original raw input files or the normalized v1 observation schema.

Each JSON metric has these fields:

| Field | Meaning |
| --- | --- |
| `baseline`, `candidate` | Recorded group mean, fraction or percentile |
| `baseline_observed`, `candidate_observed` | Independent sample size for this metric in each group |
| `absolute_delta` | Candidate minus baseline, in the metric's stated unit |
| `relative_delta_percent` | `100 * (candidate - baseline) / abs(baseline)`; `null` when baseline = 0 or either value missing |
| `percentage_point_delta` | `100 * absolute_delta` for success/accuracy fractions; otherwise `null` |
| `unit` | Units: tokens/observed record, estimated USD/record, ms, simulated retries/client, virtual seconds/client, etc. |

For rate fields the underlying fraction is on a 0–1 scale. For example,
an increase in client success from **0.5 to 0.75** has an **absolute
delta of +0.25**, a **+25 percentage-point** change and a relative
increase of **+50%**. A baseline of zero **never** produces an infinite
relative percentage; it is reported as `null` / `N/A`.

The engine reports **mean** differences for six numeric benchmark
metrics plus **p50 and p95 latency** differences using the benchmark
reporter's nearest-rank method. No interpolation, bootstrapping,
confidence intervals, significance claims, or paired-row matching
are performed. All values carry their own observed denominator.

## Lab07 interpretation

A Lab07 record is a **simulated client outcome**, not an API request.
The engine can compare simulated client success, retries, 429s and
virtual scheduled wait for two strategies in the **same scenario/run**.
Real token usage, cost, model-answer accuracy, and HTTP latency are
unavailable for Lab07. Missing measurements remain unavailable, not
zero. A smaller retry count or zero virtual wait does not by itself
make a strategy more reliable or "best."

## Provider measurement guardrails

The tool also accepts saved provider-lab runs, but will not pool
different experimental sources or cost bases. For comparing two
strategies using different model IDs, explicitly add
`--allow-model-difference`; this acknowledges, but **does not
resolve**, model, prompt, token-pricing and workload confounding.

This comparison is **within one run only**. The current CLI does
**not** compare separate runs/datasets or verify that all input prompts
and model settings are truly matched. Before drawing conclusions,
review the manifests, price table, recorded usage denominators,
latency definitions and experimental design.

## Troubleshooting

| Problem | Action |
| --- | --- |
| `Ambiguous condition` | Supply `--experiment`, `--source`, `--model`, and/or `--cost-basis` |
| `Incompatible comparison` | Pick groups with the same experiment, source, scenario and cost basis |
| `Model identifiers differ` | Use a common model, or intentionally opt in with `--allow-model-difference` |
| `Selected experiment data failed validation` | Run `python -m lab_core.experiment_validation` on that run and address reported errors |
| `N/A` or `null` | The measurement is absent, or the baseline is zero for relative differences |
| Lab07 multiple scenarios | Supply a single `--scenario` |

See [Benchmark Methodology](benchmark-methodology.md),
[Data Dictionary](data-dictionary.md),
[Experiment Validation](experiment-validation.md) and
[Architecture](architecture.md).
