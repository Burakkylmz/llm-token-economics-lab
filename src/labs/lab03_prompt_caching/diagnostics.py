"""Bounded, safe diagnostics for Lab03's cache investigation.

Two independent concerns:

  1. `compute_request_hash` -- a SHA256 of the EXACT payload sent to the
     API (model, messages, max_completion_tokens, canonically
     JSON-serialized with sorted keys), computed from the very same
     values passed to the SDK call, never a separately reconstructed
     approximation. Used to offline-verify that "identical" probe
     requests really are byte-identical, and recorded per call as a
     lightweight, content-revealing-nothing fingerprint (a hash cannot be
     reversed into the prompt text it was computed from).

  2. `usage_diagnostic_fields` -- structural facts about the RAW usage
     object the SDK returned (is `prompt_tokens_details` present at all,
     is `cached_tokens` present within it, and the raw set of field names
     the response actually populated), extracted WITHOUT ever touching
     prompt or completion text, an API key, or a reasoning trace. This is
     what distinguishes "the provider omitted cache usage reporting
     entirely" (prompt_tokens_details_present=False) from "the provider
     reported it with cached_tokens=0" (present=True, cached_tokens=0) --
     a distinction `labs.lab02_context_growth... / TokenCostMetrics.cached_tokens`
     alone cannot make once it has already collapsed to None in both cases
     at the normalized-metrics layer only by convention, not by losing data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def compute_request_hash(model: str, messages: list[dict], max_completion_tokens: int) -> str:
    canonical = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _usage_field_names(raw_usage: Any) -> list[str]:
    """The set of field names the response actually populated on the raw
    usage object -- pydantic v2's `model_fields_set` (fields explicitly
    present in the parsed response, not merely defaulted), plus any
    `model_extra` keys (unknown-to-the-SDK fields, preserved because the
    SDK's BaseModel uses `extra="allow"`). Never includes field VALUES,
    only names."""
    if raw_usage is None:
        return []
    names: set[str] = set()
    fields_set = getattr(raw_usage, "model_fields_set", None)
    if fields_set:
        names |= set(fields_set)
    extra = getattr(raw_usage, "model_extra", None)
    if extra:
        names |= set(extra.keys())
    return sorted(names)


def usage_diagnostic_fields(raw_usage: Any) -> dict:
    """Structural presence facts about the raw usage object, never its
    prompt/completion content. `prompt_tokens_details_present` is True
    only if the response actually carried that object (not merely that
    the SDK's dataclass field exists with a None default)."""
    prompt_tokens_details = getattr(raw_usage, "prompt_tokens_details", None) if raw_usage is not None else None
    prompt_tokens_details_present = prompt_tokens_details is not None
    cached_tokens = (
        getattr(prompt_tokens_details, "cached_tokens", None) if prompt_tokens_details_present else None
    )
    return {
        "prompt_tokens_details_present": prompt_tokens_details_present,
        "cached_tokens_field_present": cached_tokens is not None,
        "raw_usage_field_names": _usage_field_names(raw_usage),
    }


def groq_sdk_version() -> Optional[str]:
    try:
        import groq

        return getattr(groq, "__version__", None)
    except ImportError:
        return None
