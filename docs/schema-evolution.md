# Schema evolution and compatibility policy

The project has multiple *distinct* version concepts. Confusing them
could cause silently incorrect benchmark comparisons.

## Contract ownership

| Version | Owned by | Meaning |
| --- | --- | --- |
| `normalized_contract_version: 1` | New Lab07 experiment manifest | The saved run's metrics can be normalized to the v1, 23-field measurement contract |
| `validation_report_version: 1` | Offline validator JSON | Shape of a structured validation **report** |
| `schema_version: 1` | Benchmark JSON exporter | Shape of a descriptive **benchmark report** |
| `simulation_version` | Lab07 | Revision of virtual retry simulator behavior, **not** a measurement schema |
| Missing normalized version | Historical laboratory runs | `legacy_unversioned` adapter and explicit warning |

The current normalized machine-readable schema is
[schemas/normalized-measurement-v1.schema.json](../schemas/normalized-measurement-v1.schema.json);
see the [Data Dictionary](data-dictionary.md) for types, units and source
field precedence. This is **not** a universal schema for every raw file.

## Accepted compatibility states

- **v1**: manifest declares integer `1` (not boolean or string).
  The validator checks normalized fields, input values and versioned
  Lab07 required fields.
- **legacy_unversioned**: no version marker. The parser attempts
  read-only normalization using existing source mappings and warns.
  It does **not** automatically stamp a version or mutate old results.
  Use `--strict` if unversioned data should cause a nonzero exit.
- **unsupported**: future version, zero, negative, wrong type.
  The validator fails closed. There is **no** implicit v2-to-v1
  downgrade.
- **invalid_manifest**: unsafe, oversized or malformed manifest,
  treated as an error rather than legacy.

## Evolving to a later version

A future v2 PR should include all of:

1. A new immutable machine-readable schema file for v2. Preserve v1
   rather than editing historical contract meaning.
2. An explicit version dispatch rule and deterministic compatibility
   behavior for each supported version; unknown future versions
   must remain rejected.
3. Read-only migration/conversion functions with independently
   testable inputs/outputs. Do not change saved source records in
   place or fabricate values for fields absent in v1.
4. Golden fixture tests for v1, v2, legacy runs, malformed files
   and cross-version unknown fields; include CI against real
   locally generated Lab07 fixtures.
5. An updated Data Dictionary, benchmark methodology and migration
   guidance defining units, accuracy, cost and latency differences.
6. A documented opt-in path for consumers (Dashboard and reporter);
   never silently change historical interpretation without a
   compatibility note and test coverage.

A version marker in the manifest alone is not proof that every raw
field is valid. Validation errors must remain visible and block
strict CI publishing workflows.

## Data handling principles

- Missing reported tokens or prices remain `null`, not zero.
- A virtual Lab07 second is never a measured API millisecond.
- Cost estimates are not invoices; price basis changes require
  careful comparison and provenance.
- The validator never exports prompts, completions or credentials
  into diagnostic output.
- Source runs and experiment manifests remain immutable from the
  validator's perspective.
