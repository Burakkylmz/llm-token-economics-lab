# Participant Guide

> Stage 0 only. Lab-specific setup, objectives, execution steps, expected
> observations, and troubleshooting sections will be added here as each
> lab is implemented.

## Setup

1. Install Docker Engine/Desktop + Docker Compose, or `uv` if you prefer
   running without Docker.
2. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. Never
   share this key or commit it to version control.
3. Build and run:
   - Docker: `docker compose run --rm lab-runner --help`
   - uv: `uv sync && uv run lab-cli --help`

## Safety

No lab will ever send a real request to the Groq API — and incur real
cost on your account — unless you explicitly pass `--live` on the CLI.
Without `--live`, runs are either rejected or (once labs exist) run
against recorded/mocked data.
