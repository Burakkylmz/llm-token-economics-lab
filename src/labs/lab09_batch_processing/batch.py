"""Groq Batch API helpers: build the JSONL input, submit a job, check status,
and parse the JSONL output.

Verified against the installed `groq` Python SDK (Oct 2026):
  client.files.create(file=..., purpose="batch") -> FileCreateResponse (.id)
  client.batches.create(input_file_id=..., endpoint="/v1/chat/completions",
                         completion_window=...) -> BatchCreateResponse (.id, .status)
  client.batches.retrieve(batch_id) -> BatchRetrieveResponse (.status, .output_file_id)
  client.files.content(file_id) -> BinaryAPIResponse (.text)

A batch job's lifecycle is validating -> in_progress -> finalizing ->
completed (or expired/failed), and the completion_window ranges from 24h
to 7 days -- this module only submits/polls/parses; it never waits for
completion itself, since that would block far longer than a class session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groq import Groq


def build_batch_requests(prompts: list[str], model: str) -> list[dict]:
    """One JSONL line (as a dict) per prompt, per Groq's documented batch request schema."""
    return [
        {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": prompt}]},
        }
        for i, prompt in enumerate(prompts)
    ]


def write_batch_jsonl(requests: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def parse_batch_output_jsonl(raw_jsonl: str) -> list[dict]:
    """Parse the Batch API's output file (one JSON object per line) into a list of dicts."""
    return [json.loads(line) for line in raw_jsonl.splitlines() if line.strip()]


def submit_batch_job(client: Groq, jsonl_path: Path, completion_window: str) -> str:
    """Upload the JSONL file and create the batch job. Returns the batch id. Does not wait."""
    with open(jsonl_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch.id


def poll_batch_status(client: Groq, batch_id: str) -> dict[str, Any]:
    """A single, non-blocking status check -- callers decide whether/how to poll further."""
    batch = client.batches.retrieve(batch_id)
    return {"status": batch.status, "output_file_id": getattr(batch, "output_file_id", None)}


def download_batch_results(client: Groq, output_file_id: str) -> list[dict]:
    response = client.files.content(output_file_id)
    return parse_batch_output_jsonl(response.text)
