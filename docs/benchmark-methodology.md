# Benchmark methodology

**Scope:** implementation reference for the existing offline benchmark reporter
(`src/lab_core/benchmark_report.py`) and its input normalizer
(`src/lab_core/dashboard_data.py`). It documents what the current
code **actually computes**. This is not a new statistical experiment design,
a provider billing service, or a generalized evaluation framework.

See the [data dictionary](data-dictionary.md) for field types, file
allowlists, raw-field aliases, units, and report JSON structure.

## 1. Measurement lifecycle

1. Each laboratory writes its own saved records (file formats and schemas
   vary by lab) into a local `results/` directory.
2. `scan_results(results_dir)` scans only the files listed by
   `EXPERIMENT_SOURCES`, plus explicitly supported legacy flat files.
   For a JSON/CSV pair with the same stem, the first **successfully read**
   representation wins; if an earlier representation is invalid, a later
   eligible one can still be read. It does **not** ingest arbitrary files.
3. The normalizer produces only approved scalar fields. Unavailable and
   invalid readings remain `None`, never estimated from neighboring rows.
4. `build_benchmark(records, lab=..., run_id=..., scenario=...)` filters
   the records to **one lab and one saved run ID**, then groups them by
   `experiment`, `source`, `scenario`, `model`, `condition`, and
   `cost_basis`. Lab07 additionally requires a scenario selection when
   the chosen run contains multiple scenarios.
5. `render_markdown` or JSON serialization emits deterministic
   descriptive summaries; the reporter creates no Groq/other API calls.
   The CLI prints to standard output by default and writes a file only
   with an explicit `--output` argument.

**Important limitation:** these grouping keys prevent many accidental
cross-run comparisons, but do not prove that prompts, request settings,
randomization, workloads, or price tables match. Researchers must
check manifests/configuration before interpreting differences.

## 2. Population and denominators

Let `G` be one group of normalized records and `N = len(G)`.
The benchmark output field `requests = N` is a **record count**, not
necessarily the number of network attempts. For Lab07, a record
corresponds to one **simulated client outcome**; its retry attempts
are separate events.

For a nonnegative numeric metric `x`, define the observed set
`V_x = {x_i from G where x_i is present, numeric, finite, nonnegative, and not boolean}`.
The sample size `n_x = |V_x|` is stored as
`groups[].metrics[x].observed`.

- `None` and invalid numbers (NaN, infinity, negative values) are excluded.
- **Zero is valid measured data**, not treated as missing.
- `N` and `n_x` can differ: e.g. 20 recorded requests but only 10
  observed costs means **cost n = 10**, not 20.
- A metric can have **n = 0**; its mean, p50 and p95 are `null` in JSON
  and `N/A` in Markdown. It is never silently replaced with zero.
- The normalizer also accepts valid numeric strings for numeric fields;
  integer token/count fields must represent integral, nonnegative values.
  The reporter operates on the already-normalized values.

The rate fields use a distinct denominator:
`n_bool = number of rows for which the normalized field is exactly a Python bool`.
Missing/non-boolean values are excluded, not counted as false.

## 3. Formulas implemented in code

| Statistic | Formula or exact rule | Output |
| --- | --- | --- |
| Record count | `N = len(G)` | `groups[].requests` |
| Observed numeric sample size | `n_x = len(V_x)` | `metrics[x].observed` |
| Numeric arithmetic mean | `mean(x) = sum(V_x) / n_x`, defined only if `n_x > 0` | `metrics[x].mean` |
| Nearest-rank p50 | Sort `V_x`; select 1-based rank `ceil(0.50 * n_x)` | `metrics[x].p50` |
| Nearest-rank p95 | Sort `V_x`; select 1-based rank `ceil(0.95 * n_x)` | `metrics[x].p95` |
| Success rate | `true(succeeded) / n_succeeded` when `n_succeeded > 0` | `groups[].success.rate` |
| Answer accuracy | `true(answer_correct) / n_answer_correct` when `n_answer_correct > 0` | `groups[].answer_accuracy.rate` |

`success.evaluated` and `answer_accuracy.evaluated` report the
respective boolean denominators. Rates in JSON are fractions from
**0 to 1**; the Markdown report displays percentages.

**Nearest-rank is not interpolation.** For four sorted latency readings
`[50, 60, 70, 80]` ms, p50 = position `ceil(0.5*4) = 2`
= **60 ms** and p95 = position `ceil(0.95*4) = 4` = **80 ms**.
For a sample of one, p50 and p95 both equal that observation. Neither
percentile implies a confidence interval or statistical significance.

### Which numeric metrics receive the formula set?

<!-- benchmark-methodology:metric-fields:start -->
| Report metric | Unit | Interpretation |
| --- | --- | --- |
| `total_tokens` | tokens | Reported total token count when available |
| `estimated_cost_usd` | USD | Saved **estimated** monetary cost, not invoice |
| `latency_ms` | milliseconds | Chosen latency field; meaning depends on lab |
| `retry_count` | simulated retries/client | Lab07 only, synthetic |
| `rate_limit_response_count` | simulated responses/client | Lab07 synthetic 429 count |
| `total_scheduled_wait_seconds` | virtual seconds/client | Lab07 synthetic scheduled wait |
<!-- benchmark-methodology:metric-fields:end -->

The reporter computes mean, p50 and p95 for **every** listed numeric
metric, even if only selected ones appear in the Markdown table.
The JSON output retains the full set. A missing metric retains
`observed = 0` and `null` statistics.

## 4. Worked example: unequal observation coverage

Imagine **four** normalized provider records in one matching group.
An em dash means the metric is missing, not zero.

| Row | Tokens | Estimated cost (USD) | Latency (ms) | Succeeded | Answer correct |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 10 | 0.001 | 50 | true | true |
| 2 | — | — | 60 | false | — |
| 3 | 30 | 0.003 | 70 | — | false |
| 4 | 50 | — | 80 | false | — |

From the formula rules:

- `requests = 4`.
- Mean total tokens = `(10 + 30 + 50) / 3 = 30`
  **tokens per measured request** (`n=3`). It is **not**
  `90 / 4 = 22.5`.
- Mean estimated cost = `(0.001 + 0.003) / 2 = 0.002 USD`
  **per cost-observed request** (`n=2`). It is **not**
  `0.004 / 4 = 0.001 USD`.
- Latency mean = 65 ms (`n=4`), p50 = 60 ms, p95 = 80 ms.
- Success = 1 true out of 3 known outcomes = `1/3`, not 1/4.
- Answer accuracy = 1 true out of 2 labeled responses = `1/2`;
  unlabeled rows do **not** become incorrect answers.

This example is explanatory data, **not observed Groq benchmark data**.

## 5. Costs, tokens and timing

- `prompt_tokens`, `completion_tokens` and `total_tokens` are
  normalized from saved provider measurement fields when available.
  **Do not derive** `total_tokens` by guessing from text lengths or
  adding unavailable token categories. In particular, the preflight
  characters-per-token heuristic in `lab_core.budget` is **not**
  a recorded API token measurement.
- `estimated_cost_usd` is loaded from saved cost values. The
  normalizer selects the first present cost alias; it does **not**
  recompute provider billing in the benchmark reporter.
  The raw pricing configuration can change over time. Compare only
  compatible cost bases and pricing assumptions.
- `cost_basis` distinguishes `Estimated`,
  `Batch discounted estimate`, `Standard estimate` and
  `Unavailable`. The alias precedence and caveats are in the
  [data dictionary](data-dictionary.md).
- `latency_ms` represents the **first present** of several allowed
  lab-specific latency fields. It can represent end-to-end task latency,
  streaming total latency, follow-up latency, or client elapsed time,
  depending on the lab and source field.
  It must **not** be interpreted automatically as pure server inference
  latency or time-to-first-token (TTFT). It cannot be compared
  across different latency definitions without additional matching.
- An API response can fail or omit usage. Recorded cost totals
  may therefore **understate actual billed expenditure** in the
  presence of unknown failure billing. Cost estimates are not
  provider invoices or guaranteed spending ceilings.
- There is currently no automatically calculated token/cost
  **delta**, hypothesis test, bootstrap interval or confidence band in
  this CLI reporter. The output is **descriptive** only.

## 6. Lab07 special semantics

Lab07 uses a seeded virtual-time retry simulator, not Groq calls.
Each normalized record comes from a **client simulation result**.
Within one run, `--scenario` is mandatory if multiple scenarios exist.
The group’s `condition` represents the strategy.

- `success` is simulated client success (HTTP-like outcome), **not**
  answer correctness.
- `retry_count`, `rate_limit_response_count`, and
  `total_scheduled_wait_seconds` measure simulated outcomes
  per client. Their mean is computed over observed client rows.
- The Markdown table displays these client means and a client
  success percentage, **not** real request throughput, real latency,
  token usage or billing.
- The per-client `total_attempts` and
  `virtual_completion_time_seconds` fields exist in the normalized
  data/CSV but are **not** included in the current benchmark
  reporter's `NUMERIC_FIELDS` set.
- The Lab07 persistence layer independently calculates
  strategy-wide totals and recovery-time summaries; those are not
  the same as the per-client benchmark means.

Lab07 may return a recorded zero scheduled wait, which is different
from a missing scheduled wait.

## 7. Comparability checklist

Before comparing two model conditions or interpreting a change:

1. Confirm same or demonstrably equivalent workload/prompt set,
   scenario, model parameters, input distribution and sampling scheme.
2. Check model IDs, API/provider version, measurement definitions,
   retry/concurrency configuration and price table/date.
3. Verify `requests` and all per-metric `n` values. Identify
   selective missingness before inferring lower spend or latency.
4. Keep batch-discounted estimates distinct from standard estimates.
5. Interpret Lab07 only as **simulation** and do not equate its 429
   counts or virtual seconds with real provider telemetry.
6. Record source run IDs and configurations before publishing a
   comparative conclusion.

The current reporter separates records by
`experiment/source/scenario/model/condition/cost_basis`, but that
**does not prove** the workload is controlled. Benchmark outputs
are not evidence of causal superiority without an appropriate
experimental design and analysis.

## Condition-to-condition comparison

The optional [Experiment Comparison Engine](experiment-comparison.md)
builds on this benchmark aggregation. It computes candidate-minus-baseline
differences with explicit observed denominators; success/accuracy differences
are in fractions and percentage points. Numeric means and nearest-rank
p50/p95 latencies remain descriptive. The engine refuses unrelated source,
scenario and cost-basis groups and requires acknowledgement for model
differences. It never pairs individual requests and never claims statistical
significance.

## 8. Reproduce an existing report

```powershell
uv sync --frozen --group dev
uv run python -m lab_core.benchmark_report --lab lab07 --run-id YOUR_RUN_ID --scenario transient_failures --format json --output results/benchmarks/lab07.json
```

The reporter reads existing locally saved results only. For a new,
seeded Lab07 example, see [examples](../examples/README.md).
JSON `schema_version = 1` identifies the **report output format**,
not a versioned validation contract for every lab’s raw files.
An opt-in [Experiment Validation Engine](experiment-validation.md)
now checks selected raw measurements and the versioned normalized v1
contract. **Universal raw-file schemas, automatic migrations and
Dashboard-level enforcement remain future work.**
