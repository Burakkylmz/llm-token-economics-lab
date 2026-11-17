# Changelog

All notable project changes should be summarized here by contributors.
The Python package version in `pyproject.toml` is presently `0.1.0`,
but **no GitHub Release or SemVer compatibility promise is implied**.
Do not add a release tag before maintainers approve a version and release plan.

## Unreleased

### Added
- Architecture and explicit runtime/data-boundary documentation.
- Benchmark Methodology and Data Dictionary with verified field mappings,
  worked calculation examples, and documentation-to-code contract tests.
- Reproducible offline Lab07 examples for PowerShell and Bash.
- Read-only benchmark reporter for existing saved measurements in Markdown
  and JSON. Reports retain comparable workload groups and observed sample
  counts for token, cost, latency, quality and synthetic retry outcomes.
- Offline reporter regression tests for grouping, missing data and
  deterministic output.

### Existing capabilities
- Ten CLI training laboratories, with Lab07 always simulation-only.
- Streamlit dashboard in a separate local-only Docker Compose service.
- Offline tests, code-quality/security CI and Docker health smoke tests.

## Release policy (planned, not yet active)

When maintainers choose a public license and release policy, document
the compatibility promises, create a version tag and include the CI
evidence and migration notes in the corresponding GitHub Release.
