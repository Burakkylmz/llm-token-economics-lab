# Data dictionary: normalized measurements and benchmark JSON

**Canonical implementation:** `src/lab_core/dashboard_data.py`
(`EXPERIMENT_SOURCES`, `_normalize`, `EXPORT_COLUMNS`) and
`src/lab_core/benchmark_report.py` (`GROUP_KEYS`, `NUMERIC_FIELDS`,
`build_benchmark`). This dictionary describes the **read-only
normalized/output layers**. It is **not** a single common, enforced
on-disk schema for all 10 laboratories.

For calculation formulas and interpretation limitations see
[Benchmark Methodology](benchmark-methodology.md).

## Input layouts and allowed measurement files

Modern experiments use `results/<experiment>/<run_id>/<filename>`.
Directory names identify an experiment family and run; the normalized
`run_id` is the *directory name*, not an arbitrary embedded row value.

<!-- data-dictionary:experiment-sources:start -->
| Experiment directory | Allowed measurement filenames, in priority order |
| --- | --- |
| `lab01_effort` | `raw_measurements.json`, `raw_measurements.csv` |
| `lab01_factorial` | `raw_measurements.json`, `raw_measurements.csv` |
| `lab02_context_growth` | `context_measurements.json`, `multi_turn_measurements.json` |
| `lab03_prompt_caching` | `cache_measurements.json` |
| `lab04_output_control` | `output_limit_measurements.json`, `invocation_measurements.json` |
| `lab05_model_routing` | `routing_measurements.json` |
| `lab06_streaming` | `stream_measurements.json` |
| `lab07_rate_limiting` | `raw_measurements.json` |
| `lab08_concurrency` | `raw_measurements.json` |
| `lab09_batch_processing` | `sequential_measurements.json`, `concurrent_measurements.json`, `batch_measurements.json` |
| `lab10_conversation_summarization` | `summary_measurements.csv`, `followup_measurements.csv` |
<!-- data-dictionary:experiment-sources:end -->

For a JSON and CSV alternative sharing the same stem, the first
**successfully parsed** file wins, to avoid double-counting.
Other files with different stems within one experiment can contribute
different measurement families, each labeled by its `source`.
The loader also supports a limited set of older flat files:
`results/labNN.json`, `results/labNN_token_anatomy.json`, and
`results/labNN_rate_limiting.json` (where `NN` is 01–10).
These are marked `experiment = legacy` and `run_id = legacy`.
The legacy reader copies **only selected metric-bearing nested fields**.

Safety limits in `scan_results`: 20 MB per measurement file,
50,000 rows per file, and a 100 KB manifest limit.
Symlinked experiment/run directories and measurement files are ignored
or rejected. Invalid/unreadable input is skipped with warnings.
A missing results folder yields an empty record set, not fake readings.

## Normalized observation contract

Each record represents one selected measurement entry. For Lab07, it
represents **one simulated client result**, not a real API request.
These are the current `EXPORT_COLUMNS` used by the dashboard's
approved CSV export. Fields absent in an individual raw input may be
represented by `None` (JSON `null` in a serialized report).

Notation: `str` = text; `int?` = integer or missing;
`float?` = nonnegative finite number or missing; `bool?` =
true/false or missing. Empty text/default labels are stated explicitly.

<!-- data-dictionary:normalized-fields:start -->
| Field | Normalized type | Unit or definition | Null/default and interpretation |
| --- | --- | --- | --- |
| `lab` | str | `lab01` through `lab10` | Derived from experiment directory; legacy keeps lab |
| `experiment` | str | Measurement family directory | `legacy` for supported flat files |
| `run_id` | str | Saved run directory name | `legacy` for supported flat files |
| `source` | str | Allowed source filename | Tracks which family of observations was read |
| `model` | str | Model identifier | `Unknown` when unavailable; `Simulation (no LLM)` for Lab07 |
| `condition` | str | Experimental condition/strategy | `Unspecified` when absent |
| `scenario` | str | Scenario label | `Unspecified` when `scenario_id` absent |
| `timestamp` | str | Raw saved timestamp label | Empty string when unavailable; not parsed/validated |
| `status` | str | Provider/simulation status | `Unknown` when neither status nor success is available |
| `succeeded` | bool? | Call/client success indicator | `None` when unknown; may be inferred from recognized status |
| `answer_correct` | bool? | Explicitly labeled response correctness | `None` when no ground-truth label; never inferred from success |
| `prompt_tokens` | int? | Prompt tokens | Provider/report measurement when present; missing stays null |
| `completion_tokens` | int? | Completion tokens | Provider/report measurement when present; missing stays null |
| `total_tokens` | int? | Total tokens | Saved measurement when present, not inferred from text |
| `estimated_cost_usd` | float? | USD | Saved **estimate**, not provider invoice |
| `cost_basis` | str | Cost interpretation | `Estimated`, `Batch discounted estimate`, `Standard estimate`, or `Unavailable` |
| `latency_ms` | float? | Milliseconds | First available lab-specific duration; definition varies |
| `simulated` | bool | True when lab = `lab07` | Indicates synthetic retry measurements, not API telemetry |
| `total_attempts` | int? | Simulated attempts per client | Lab07 only; otherwise null |
| `retry_count` | int? | Simulated retries per client | Lab07 only; otherwise null |
| `rate_limit_response_count` | int? | Simulated 429 responses per client | Lab07 only; otherwise null |
| `total_scheduled_wait_seconds` | float? | Virtual seconds per simulated client | Lab07 only; not measured wall time |
| `virtual_completion_time_seconds` | float? | Virtual seconds from start to client completion | Lab07 only; may be missing for failed clients |
<!-- data-dictionary:normalized-fields:end -->

The normalizer accepts valid numeric strings in CSV and enforces
nonnegative finite numeric values. Token/count fields additionally
require an *integral* value; fractional counts are rejected.
`0` is a legitimate measurement and must **not** become `None`.
The boolean normalizer accepts native booleans, numeric `0`/`1`,
and literal strings `"0"`, `"1"`, `"true"`, `"false"`,
`"True"`, `"False"`. Other representations remain unknown.
Invalid numbers and unrecognized booleans become `None`.

`EXPORT_COLUMNS` is a **whitelist**, not a promise that every raw
source has all fields. The CSV export escapes formula-like prefixes
in text to reduce spreadsheet injection risk; it does not include
raw prompts or completions. The loader parses source JSON/CSV rows
locally and discards unapproved fields during normalization.
Do not treat the raw results directory as safe to publish.

## Raw field mapping and precedence

These are **ordered aliases** implemented in `_normalize` using the
first present, nonblank raw value. For example, where both raw
`total_latency_ms` and `performance.client_elapsed_ms` exist,
the first one wins. No extra field is fabricated to fill a gap.

| Normalized output | Raw field lookup or rule |
| --- | --- |
| `model` | First of `selected_model`, `model`, `request_model`, `metadata.model`; then `experiment_manifest.json`'s `experiment_configuration.model` or `default_model`, then manifest `model` |
| `condition` | First of `strategy_id`, `strategy`, `condition_id`, `condition`, `execution_mode`, `call_purpose`, `prompt_type`, `metadata.experiment_id` |
| `scenario` | `scenario_id` or `Unspecified` |
| `status` | First of `request_status`, `reconciliation_status`, `stream_status`, `outcome`; if absent, derive `SUCCESS` or `FAILED` from known `succeeded` |
| `succeeded` | Parse direct `succeeded`; otherwise `SUCCESS` implies true and `RATE_LIMITED`, `TIMEOUT`, `CONNECTION_ERROR`, `PROVIDER_ERROR`, `CANCELLED`, `UNKNOWN_ERROR`, `FAILED`, `ERROR` imply false |
| `answer_correct` | Parse direct `answer_correct` only |
| `prompt_tokens` | `prompt_tokens`, `followup_prompt_tokens`, `summary_generation_prompt_tokens`, `tokens_and_cost.prompt_tokens` |
| `completion_tokens` | `completion_tokens`, `followup_completion_tokens`, `summary_generation_completion_tokens`, `tokens_and_cost.completion_tokens` |
| `total_tokens` | `total_tokens`, `followup_total_tokens`, `summary_generation_total_tokens`, `tokens_and_cost.total_tokens` |
| `estimated_cost_usd` | `estimated_cost_usd`, `estimated_batch_cost_usd`, `estimated_standard_cost_usd`, `followup_estimated_cost_usd`, `summary_generation_estimated_cost_usd`, `tokens_and_cost.estimated_cost_usd` |
| `latency_ms` | `total_latency_ms`, `task_end_to_end_latency_ms`, `total_stream_latency_ms`, `non_streaming_total_latency_ms`, `followup_latency_ms`, `summary_generation_latency_ms`, `performance.client_elapsed_ms` |
| `timestamp` | `timestamp`, `metadata.timestamp`; otherwise empty string |
| Lab07 simulation metrics | Direct matching raw per-client field names only; not extrapolated from strategy summaries |

**Important cost-basis nuance:** `estimated_batch_cost_usd` produces
`Batch discounted estimate`; `estimated_standard_cost_usd` produces
`Standard estimate`. All other chosen cost aliases produce
`Estimated`. If no candidate exists, `cost_basis` is `Unavailable`.
If the **first present** cost alias contains invalid numeric data, the
normalized cost can be null *even though the cost basis records the
selected alias*: the normalizer does not try the next alias.

For legacy files, the loader first flattens a subset of
`metadata`, `tokens_and_cost`, and `performance` into
dotted keys. This is **not** a general-purpose nested-field
resolver for all modern JSON formats.

## Benchmark report output (JSON `schema_version = 1`)

The reporter accepts a selected `lab`, exact `run_id`, optional
`scenario`, and emits one deterministic document.
`schema_version = 1` is the **report output version**, not a
schema version assigned to every raw laboratory measurement file.

Report top-level fields:

| JSON path | Type | Definition |
| --- | --- | --- |
| `schema_version` | int | Constant value `1` for the reporter output |
| `lab` | str | Selected lab ID |
| `run_id` | str | Selected exact saved run ID |
| `scenario` | str or null | Selected scenario; if not selected, the sole scenario when unique, otherwise null |
| `measurement_count` | int | Number of selected normalized rows, after optional scenario filtering |
| `group_count` | int | Number of distinct comparison groups |
| `groups` | list | Per-group descriptive summaries |

Per-group keys:

<!-- data-dictionary:group-keys:start -->
| Grouping key | Purpose |
| --- | --- |
| `experiment` | Keep distinct experiment families apart |
| `source` | Keep different saved measurement sources apart |
| `scenario` | Keep simulation/task scenarios apart |
| `model` | Separate model identifiers |
| `condition` | Separate strategies/experimental conditions |
| `cost_basis` | Avoid pooling batch/standard cost estimates |
<!-- data-dictionary:group-keys:end -->

Each group also contains:

| JSON path | Type | Semantics |
| --- | --- | --- |
| `requests` | int | Count of normalized records in the group, **not necessarily actual HTTP attempts** |
| `simulated` | bool | True when all records in the group carry `simulated = true` |
| `success.evaluated` | int | Count with explicitly known normalized bool `succeeded` |
| `success.rate` | float or null | Fraction of those known outcomes that are true |
| `answer_accuracy.evaluated` | int | Count with explicitly known normalized bool `answer_correct` |
| `answer_accuracy.rate` | float or null | Fraction of correct outcomes among labeled responses |
| `metrics` | object | One statistics object for every current `NUMERIC_FIELDS` entry |

Per-group numeric statistics have `observed`, `mean`, `p50`
and `p95`. `observed` counts only valid measured values;
the other three are null when observed = 0. Rates are fractions
0–1, not percentages, in JSON.

<!-- data-dictionary:report-metrics:start -->
| Metric key under `groups[].metrics` | Unit |
| --- | --- |
| `total_tokens` | tokens |
| `estimated_cost_usd` | USD |
| `latency_ms` | milliseconds |
| `retry_count` | simulated retries per client |
| `rate_limit_response_count` | simulated 429 responses per client |
| `total_scheduled_wait_seconds` | virtual seconds per client |
<!-- data-dictionary:report-metrics:end -->

The reporter **does not** currently aggregate `prompt_tokens`,
`completion_tokens`, `total_attempts`, or
`virtual_completion_time_seconds` into the benchmark JSON
metrics map. Those fields remain available in normalized CSV
or other lab-specific reports.

## Contract boundaries / migration status

- The data dictionary is **versioned by the Git commit that contains
  it**. There is no enforced universal raw schema registry.
- Existing loader/source types are heterogeneous: modern JSON, CSV and
  supported historical flat files.
- `answer_correct` means an explicitly recorded correctness label.
  A valid HTTP response is **not** automatically a correct model answer.
- Lab07's synthetic values must never be converted into real Groq
  token/cost/latency measurements.
- A future Experiment Validation Engine should introduce explicit
  raw-format schema versions, migration policies, validation errors
  and backwards-compatibility tests **before** claiming compatibility.
  Those capabilities are **not** implemented by this documentation PR.

See [Benchmark Methodology](benchmark-methodology.md) for formulas,
observability and comparability rules.
