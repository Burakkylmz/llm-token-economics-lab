"""Versioned, fixed prompt text for Lab10.

Both the summarization instruction and the answering instruction are kept
as single, stable constants (never dynamically modified per dataset,
question, or condition) so the ONLY variable across Full History /
Summarization Low / Summarization Medium is the context representation
(Part 5.1/6.1 of the task spec). Each constant's SHA-256 hash is recorded
in the experiment manifest so a prompt-wording change is always visible in
persisted results, never a silent drift.
"""

from __future__ import annotations

import hashlib

PROMPT_VERSION = "v1"

SUMMARIZATION_INSTRUCTION = """You are creating a compact conversation memory for future chatbot turns.

Summarize the supplied conversation faithfully.

Your objective is to reduce context length while preserving information
that may be required to answer future questions.

Preserve:
1. Important identifiers and exact values.
2. Confirmed facts.
3. The latest explicitly stated decisions and preferences.
4. Facts that were changed or superseded.
5. Important chronological relationships.
6. Unresolved issues and pending actions.
7. Explicit uncertainty or missing information when relevant.

Rules:
- Do not invent facts.
- Do not turn an assumption into a confirmed fact.
- Do not treat an earlier decision as the current decision.
- Keep identifiers, numeric values and dates exact.
- Distinguish temporary actions from permanent resolutions.
- Do not answer possible future questions.
- Do not include explanations about how you created the summary.
- Use only information contained in the supplied conversation.
- Keep the summary substantially shorter than the original conversation.

Return a compact JSON object following the specified summary schema."""

# Appended to SUMMARIZATION_INSTRUCTION so the model's JSON-mode output
# matches the generic, dataset-agnostic schema from Part 5.2. Kept as a
# separate constant (rather than inlined) so the schema text itself has
# its own stable, hashable identity.
SUMMARY_JSON_SCHEMA_DESCRIPTION = """Summary JSON schema (use exactly these top-level keys; omit nothing):
{
  "overview": "Brief summary of the conversation",
  "facts": [
    {"name": "fact name", "value": "exact value", "status": "confirmed"}
  ],
  "decisions": [
    {"topic": "decision topic", "current_value": "latest value", "superseded_value": "previous value or null"}
  ],
  "open_issues": [
    {"issue": "issue description", "status": "current status"}
  ],
  "timeline": [
    {"event": "significant event", "date": "date or null"}
  ]
}

This schema is generic and must be used for any conversation topic -- do
not add topic-specific top-level keys."""

SUMMARIZATION_SYSTEM_PROMPT = f"{SUMMARIZATION_INSTRUCTION}\n\n{SUMMARY_JSON_SCHEMA_DESCRIPTION}"

ANSWER_INSTRUCTION = """You are answering a question about a supplied conversation context.

Use only information present in the supplied context.

Return a JSON object containing exactly one key: "answer".

If the requested information is not available in the context,
return "NOT_STATED".

Do not guess.
Do not add an explanation.
Do not include markdown.

For identifiers, preserve exact spelling.
For dates, use YYYY-MM-DD.
For integer answers, use decimal digits.
For enumerated answers, use the required canonical value."""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SUMMARIZATION_PROMPT_HASH = prompt_hash(SUMMARIZATION_SYSTEM_PROMPT)
ANSWER_PROMPT_HASH = prompt_hash(ANSWER_INSTRUCTION)
