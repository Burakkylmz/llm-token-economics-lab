"""SHA256 of the semantic request payload -- deliberately EXCLUDES the
`stream` transport flag and any stream-only option (e.g.
`stream_options`/`extra_body`), so hashing both calls in a pair confirms
they are the same semantic request differing only in transport, never in
content (Part D's Request Equivalence/Request Hash requirement).

Same canonical-JSON + sha256 pattern as
labs.lab03_prompt_caching.diagnostics.compute_request_hash, written
locally rather than imported cross-lab since the hashed field set differs
(adds reasoning_effort, which Lab03 never varies).
"""

from __future__ import annotations

import hashlib
import json


def compute_semantic_payload_hash(model: str, messages: list[dict], max_completion_tokens: int, reasoning_effort: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
