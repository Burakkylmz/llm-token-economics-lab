# Contributing to LLM Token Economics Lab

Thank you for improving the experiments, tooling or documentation.

## Repository workflow

The upstream repository is [Burakkylmz/llm-token-economics-lab](https://github.com/Burakkylmz/llm-token-economics-lab).
Contributions are proposed through a focused branch and Pull Request. Do not
force-push a maintainer's branch or bypass review and automated checks.

1. Fork or branch from the latest `master` revision.
2. Make one logically cohesive change per PR, with descriptive commit messages.
3. Add or update unit tests for changed behavior, and describe any schema changes.
4. Run the checks below locally.
5. Open a PR stating the goal, test evidence, migration impact, and risks.
6. Wait for required reviews and the **offline tests** and **quality gates** to
   pass before merging. The maintainers make the final merge decision.

## Local checks

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --group dev
uv run pytest -q
uvx --from "ruff==0.14.4" ruff check src dashboard tests --select E4,E7,E9,F
uv run --with "mypy==1.18.2" mypy --follow-imports=skip --ignore-missing-imports --check-untyped-defs src/lab_core/dashboard_view.py src/lab_core/budget.py
uv run --with "pytest-cov==6.2.1" pytest -q --cov=lab_core --cov=labs --cov-report=term-missing --cov-fail-under=65
uvx --from "bandit==1.8.6" bandit -r src dashboard --severity-level high --confidence-level high --quiet
```

The CI quality workflow also performs an **advisory** dependency audit on
the locked runtime dependencies. Advisory does **not** mean no vulnerability;
review its warnings. Package dependencies should be updated intentionally
with `uv lock`, including review of the resulting `uv.lock` diff.

## Live API safety and privacy

- The default configuration has `live_api_enabled = false`. Do not change
  this default or trigger provider requests during import, dashboard
  rendering, tests, or normal CI.
- Live experiments require both `--live` and a local configuration with
  `live_api_enabled = true`; never put real credentials in tests.
- Never commit `.env`, `config/live.toml`, private prompts, raw user
  conversations, account identifiers, or `results/`.
- When adding a measurement, distinguish **reported provider usage** from
  **derived estimates**, and missing values from zero.
- Keep simulator/virtual-time measurements separate from real API latency.
- Do not claim causal differences between unmatched experiments.
- Redact secrets and personal information from screenshots and issue logs.

## Code conventions

Follow clear module boundaries: reusable policies belong in `lab_core`,
not in another lab's internal package. Prefer injected dependencies and
small pure functions for calculations, with deterministic mock-driven tests.

When changing result formats, provide backward-compatible readers or
document an explicit data migration. Add regression tests for bad input,
empty data and repeated-run reproducibility.

## Licensing

No public `LICENSE` file has been selected at this time. This project
should not be described as OSI-licensed open source until the maintainers
agree on a license and add the appropriate file.
