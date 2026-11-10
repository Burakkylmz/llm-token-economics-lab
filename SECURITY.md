# Security policy

## Scope

This project is an educational LLM experiment runner and local results
viewer, **not** a production API gateway, authentication service or
multi-tenant hosting platform. The supported development line is
`master`; older commits are not promised security patches.

## Reporting a vulnerability

Please **do not** disclose API keys, personal data, exploit details or
unpatched vulnerabilities in public GitHub issues, PRs or discussions.

Use GitHub's **private vulnerability reporting** feature on the upstream
repository if it is available (the repository's **Security** tab), or
contact the repository maintainers privately through an existing channel
listed on their GitHub profiles. If neither private channel is available,
open a public issue requesting a private reporting contact **without
including any vulnerability details**.

Useful private reports include: affected revision, minimal reproduction,
attack surface, impact, and suggested mitigation. Do not include working
credentials, private prompts or real user data in reports.

The maintainers will assess reports and choose a remediation and
disclosure plan. No response-time or support SLA is promised.

## Security boundaries

- `GROQ_API_KEY` must be provided through a local environment, never in
  source control. `.env` and `config/live.toml` are git-ignored.
- Default configuration disallows live API requests. Both `--live` and
  explicit `live_api_enabled = true` are required for normal live runs.
- The Streamlit dashboard reads allowlisted local measurement files and
  never initiates provider API calls. It is not hardened for public
  internet deployment or untrusted multi-tenant filesystems.
- Local results may contain sensitive prompts or model outputs in other
  files. Do not publish the `results/` directory.
- Dependencies are checked by CI with a Bandit static scan and an
  advisory dependency audit. Any flagged vulnerability needs triage;
  an advisory check does not certify the environment as safe.
- The project measures cost using configuration-time **estimates**, not
  verified provider invoices or a guaranteed billing cap.

## Known limitations

The dashboard's file-size and field allowlists reduce accidental disclosure
but do not replace host-level filesystem access control. Do not expose the
dashboard on an untrusted network without authentication, transport
security, and a separate deployment security review.

Maintainers should use protected branches and require passing tests and
review before promoting PRs to `master`.
