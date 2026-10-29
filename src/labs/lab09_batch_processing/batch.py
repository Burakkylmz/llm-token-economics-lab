"""Groq Batch API helpers: build the JSONL input from the golden dataset,
submit a job, check status (full field set), and parse/download the
JSONL output and error files.

Verified against the installed `groq` Python SDK and
https://console.groq.com/docs/batch (2026-10-29):
  client.files.create(file=..., purpose="batch") -> FileCreateResponse (.id)
  client.batches.create(input_file_id=..., endpoint="/v1/chat/completions",
                         completion_window=...) -> BatchCreateResponse
  client.batches.retrieve(batch_id) -> BatchRetrieveResponse (full field set
                         below)
  client.files.content(file_id) -> BinaryAPIResponse (.text)

A batch job's lifecycle is validating -> in_progress -> finalizing ->
completed (or failed/expired/cancelling/cancelled), and the
completion_window ranges from 24h to 7 days -- this module only
submits/polls/parses; it NEVER waits for completion itself, since that
would block far longer than a class session (spec Part 3's explicit
constraint).

`custom_id` is always the golden task's own `task_id` -- the stable key
used for reconciliation (see reconciliation.py), never a positional index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from groq import Groq

from labs.lab09_batch_processing.datasets import GoldenTask


def build_batch_requests(tasks: list[GoldenTask], model: str) -> list[dict]:
    """One JSONL line (as a dict) per task, keyed by task_id as custom_id."""
    return [
        {
            "custom_id": task.task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": task.prompt}]},
        }
        for task in tasks
    ]


def write_batch_jsonl(requests: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def parse_batch_output_jsonl(raw_jsonl: str) -> list[dict]:
    """Parse a Batch API output/error file (one JSON object per line).

    A line that fails to parse as JSON is NOT silently dropped -- it is
    returned as a dict carrying `_malformed_line` so callers (reconciliation)
    can report it rather than lose it.
    """
    parsed: list[dict] = []
    for line in raw_jsonl.splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            parsed.append({"_malformed_line": line})
    return parsed


def submit_batch_job(client: Groq, jsonl_path: Path, completion_window: str) -> tuple[str, str]:
    """Upload the JSONL file and create the batch job. Returns (input_file_id,
    batch_id). Does not wait for completion."""
    with open(jsonl_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return uploaded_file.id, batch.id


def fetch_batch_job_fields(client: Groq, batch_id: str) -> dict[str, Any]:
    """A single, non-blocking status check -- the full documented field set,
    each defaulting to None (never fabricated) when the SDK response omits it."""
    batch = client.batches.retrieve(batch_id)
    request_counts = getattr(batch, "request_counts", None)
    return {
        "provider_status": getattr(batch, "status", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "created_at": getattr(batch, "created_at", None),
        "in_progress_at": getattr(batch, "in_progress_at", None),
        "finalizing_at": getattr(batch, "finalizing_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "request_counts_total": getattr(request_counts, "total", None) if request_counts else None,
        "request_counts_completed": getattr(request_counts, "completed", None) if request_counts else None,
        "request_counts_failed": getattr(request_counts, "failed", None) if request_counts else None,
    }


def download_batch_file(client: Groq, file_id: Optional[str]) -> list[dict]:
    """Download and parse a Batch output or error file. Returns [] if
    file_id is None (nothing available yet) -- never raises for that case."""
    if file_id is None:
        return []
    response = client.files.content(file_id)
    return parse_batch_output_jsonl(response.text)
