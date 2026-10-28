"""Classifies raw Groq streaming chunks into one of eight event types,
verified against the installed SDK's own schema
(`groq.types.chat.chat_completion_chunk`, groq==1.7.0):

  - `choices` CAN be an empty list (the docstring on `ChatCompletionChunk.
    choices` says so explicitly: "Can also be empty for the last chunk if
    you set `stream_options: {"include_usage": true}`") -- `choices[0]` is
    never indexed without checking first.
  - `delta.content` can be `None` (no content this chunk), `""` (an
    explicit empty string), or a non-empty string.
  - `delta.reasoning` is a real, separate field (only populated when the
    request set `reasoning_format="parsed"`, which this lab always does --
    see labs.lab06_streaming.runner) -- distinct from `delta.content`, so
    reasoning text is never mistaken for visible content.
  - Usage can arrive on the top-level `chunk.usage` field (when
    `extra_body={"stream_options": {"include_usage": True}}` was sent --
    `stream_options` is NOT a typed SDK parameter, so it must go through
    `extra_body`, confirmed by reading `completions.create`'s full
    parameter list) OR on `chunk.x_groq.usage` (Groq-specific metadata,
    documented as "Only present when...the final chunk" independently of
    stream_options) -- both locations are checked.

Whitespace policy (Part C.C, explicitly required to be documented): a
chunk whose `delta.content` is a non-empty string is classified
VISIBLE_CONTENT even if that string is pure whitespace (it is still real,
provider-generated content). However, `first_visible_content_ms`/
`last_visible_content_ms` (computed in labs.lab06_streaming.runner, not
here) only advance on content that has at least one non-whitespace
character -- "first MEANINGFUL visible content", not "first non-None
content object".
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

EVENT_ROLE_ONLY = "ROLE_ONLY"
EVENT_REASONING_DELTA = "REASONING_DELTA"
EVENT_VISIBLE_CONTENT = "VISIBLE_CONTENT"
EVENT_EMPTY_CONTENT = "EMPTY_CONTENT"
EVENT_USAGE_ONLY = "USAGE_ONLY"
EVENT_FINISH_EVENT = "FINISH_EVENT"
EVENT_OTHER_METADATA = "OTHER_METADATA"
EVENT_UNKNOWN = "UNKNOWN"

EVENT_TYPES = (
    EVENT_ROLE_ONLY, EVENT_REASONING_DELTA, EVENT_VISIBLE_CONTENT, EVENT_EMPTY_CONTENT,
    EVENT_USAGE_ONLY, EVENT_FINISH_EVENT, EVENT_OTHER_METADATA, EVENT_UNKNOWN,
)


class ClassifiedChunk(NamedTuple):
    event_type: str
    visible_content: Optional[str]
    finish_reason: Optional[str]
    has_usage: bool


def extract_chunk_usage(chunk: Any) -> Optional[Any]:
    """Checks both documented usage locations (see module docstring),
    top-level first."""
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        return usage
    x_groq = getattr(chunk, "x_groq", None)
    return getattr(x_groq, "usage", None) if x_groq is not None else None


def classify_chunk(chunk: Any) -> ClassifiedChunk:
    usage = extract_chunk_usage(chunk)
    has_usage = usage is not None
    choices = getattr(chunk, "choices", None) or []

    if not choices:
        if has_usage:
            return ClassifiedChunk(EVENT_USAGE_ONLY, None, None, has_usage)
        return ClassifiedChunk(EVENT_UNKNOWN, None, None, has_usage)

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(delta, "content", None) if delta is not None else None
    reasoning = getattr(delta, "reasoning", None) if delta is not None else None
    role = getattr(delta, "role", None) if delta is not None else None

    if content:
        return ClassifiedChunk(EVENT_VISIBLE_CONTENT, content, finish_reason, has_usage)
    if content == "":
        return ClassifiedChunk(EVENT_EMPTY_CONTENT, None, finish_reason, has_usage)
    if reasoning:
        return ClassifiedChunk(EVENT_REASONING_DELTA, None, finish_reason, has_usage)
    if finish_reason is not None:
        return ClassifiedChunk(EVENT_FINISH_EVENT, None, finish_reason, has_usage)
    if role is not None:
        return ClassifiedChunk(EVENT_ROLE_ONLY, None, finish_reason, has_usage)
    if has_usage:
        return ClassifiedChunk(EVENT_USAGE_ONLY, None, finish_reason, has_usage)
    return ClassifiedChunk(EVENT_OTHER_METADATA, None, finish_reason, has_usage)
