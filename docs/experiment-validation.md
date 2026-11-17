# Experiment Validation Engine (v1)

This is the **offline, opt-in** validator for saved experiment measurements.
It does not launch labs, contact Groq, recalculate provider billing, modify
original results, or upload raw prompts. The existing Dashboard and benchmark
reporter remain read-only and continue to work independently.

The first versioned contract is for **normalized metric records**, not a
promise that Lab01–Lab10 share one raw JSON/CSV schema.

## Quick start

Run from the repository root, using Python 3.12 and uv:

```powershell
uv sync --frozen --group dev
uv run python -m lab_core.experiment_validation --help
uv run lab-cli lab07 all --profile pilot --seed 42 --experiment-run-id v1-validation-example
uv run python -m lab_core.experiment_validation --experiment lab07_rate_limiting --run-id v1-validation-example --strict --format json --output results/validation-reports/lab07.json
```

A run ID must be the name of an **existing** run directory. Choose a new
one for subsequent Lab07 experiments; Lab07 does not overwrite existing runs.

Docker Compose without Python installed on the host:

```powershell
docker compose build lab-runner
docker compose run --rm lab-runner lab07 all --profile pilot --seed 42 --experiment-run-id docker-validate-example
docker compose run --rm --entrypoint uv lab-runner run python -m lab_core.experiment_validation --experiment lab07_rate_limiting --run-id docker-validate-example --strict --format json --output results/validation-reports/docker-lab07.json
```

The `lab-runner` service writes reports to the existing bind-mounted
`results/` directory; no Groq API key is needed. The report output is
saved outside the selected run folder so the original inputs are not changed.

For a provider experiment with previously saved measurements:

```powershell
uv run python -m lab_core.experiment_validation --experiment lab05_model_routing --run-id YOUR_RUN_ID --format text
```

Older Lab05 outputs lack the new manifest version marker and receive
a **legacy_unversioned** warning. Normal validation accepts warnings;
`--strict` returns exit code 1 if any warning is present.

## Version and compatibility semantics

The v1 machine-readable schema is
[normalized-measurement-v1.schema.json](../schemas/normalized-measurement-v1.schema.json).

| Condition | Result | Policy |
| --- | --- | --- |
| Manifest contains `normalized_contract_version: 1` | `v1` | Check fields and raw values, with strict Lab07 required fields |
| Manifest absent or lacks that key | `legacy_unversioned` | Check fields with compatibility warnings; no automatic mutation |
| Manifest declares an unknown future version such as `2` | `unsupported` | **Fail closed**; no silent downgrade |
| Manifest version has invalid type/value (including `true`, string `"1"`, or `0`) | `unsupported` | Error; no coercion |
| Manifest unreadable/malformed/symlinked | `invalid_manifest` | Error; not silently treated as legacy |

**Important:** `simulation_version` in Lab07 describes its simulator;
`schema_version` in the benchmark JSON describes a benchmark output.
Neither denotes this normalized measurement contract.
Only the explicit `normalized_contract_version` manifest field is
used for validation compatibility.

New Lab07 manifests now declare v1. Other labs keep their historical
manifests; this PR does not rewrite them or assert that their heterogeneous
raw formats have migrated.

## Validation checks

- Only the existing [allowlisted sources](data-dictionary.md) are scanned
  beneath an explicitly named experiment and run. Unknown arbitrary files
  are not scanned; symbolic links are rejected.
- Input size/row-count bounds match the existing Dashboard reader:
  at most 20 MB and 50,000 rows per file; manifests at most 100 KB.
- Present numeric raw fields must be finite, nonnegative and integral
  where applicable; invalid numeric data is reported **before**
  normalization can turn it into `None`.
- The normalized v1 record must contain exactly the 23 approved
  fields, with compatible text/bool/integer/number/null types.
- Missing data remains missing; a real zero stays zero. Unlabeled
  `answer_correct` does not automatically mean incorrect.
- Fully known prompt/completion/total token counts are checked for
  arithmetic consistency (mismatch is a **warning**, not assumed fraud).
- Lab07 checks required per-client fields on **new v1 runs**,
  simulated flag, zero real provider measurements, attempts/retries
  and simulated HTTP 429 counts.
- Invalid or missing measurement files are errors. No provider SDK is
  imported and no request is sent.
- Diagnostics have stable `severity`, `code`, `source`, `row`
  and `field` keys; they never include raw field values, prompts
  or model responses. At most 100 are included; overflow is counted
  and marks the validation unsuccessful.

The validator verifies structural and selected semantic invariants.
It does **not** prove experiment scientific validity, correct label
provenance, up-to-date pricing, semantic workload matching, or complete
provider telemetry. Those require separate controls.

## Exit codes and reports

| Exit code | Meaning |
| --- | --- |
| `0` | Validation passed (warnings allowed unless `--strict`) |
| `1` | Errors or truncated diagnostics, or warnings under `--strict` |
| `2` | Invalid arguments, unknown experiment, unsafe run ID or missing run |

`--format text` (default) is for terminals; `--format json` provides
a versioned machine-readable report. The report has
`validation_report_version: 1`, `normalized_contract_version: 1`,
`declared_normalized_contract_version`, `compatibility`, `valid`,
`files_checked`, `records_checked`, counts, and diagnostics.

`--output` is optional; when absent the report goes to stdout.
A report cannot be written anywhere **inside the selected input run**.
Reports under `results/validation-reports/` are ignored by Git.

## Schema evolution policy

Read [Schema Evolution](schema-evolution.md) before changing field
definitions or adding future contract versions. In particular, do
not silently map an unsupported future version to v1, and do not
reinterpret old measurements without preserving their provenance.

## Using the validation engine with comparisons

The [Experiment Comparison Engine](experiment-comparison.md) performs
a read-only validation of the selected saved run before producing
differences. Validation errors block comparison. Unversioned legacy
runs remain permitted with compatibility warnings, but workload
equivalence, model pricing and experimental design still require
independent review.

## Current limitations

- This is an opt-in CLI. The Dashboard and benchmark reporter do
  not automatically block consumption of validation-failed data.
- There is no universal typed raw-file schema, live provider validation,
  auto-repair, historical data rewriting, or published migration CLI.
- Version support is **v1 only**. Adding v2 requires a new contract,
  tests, explicit compatibility/migration policy, and an intentional
  version registry change.
