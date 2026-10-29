"""Loading and integrity-checking for Lab10's versioned synthetic golden
datasets (`data/lab10/*.json`).

All conversations are fictional (Part 4.1): no real customers, orders, or
confidential information. Dataset integrity (Part 4.4/19.1) is checked by
`check_dataset_integrity`, which every dataset must pass before it can be
used in an experiment -- this never requires a model call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "lab10"

DATASET_IDS = ("customer_support_001", "technical_incident_001")

ANSWER_TYPES = ("exact_identifier", "enum", "iso_date", "integer", "unknown")


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Conversation:
    dataset_id: str
    scenario: str
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    expected_answer: str
    answer_type: str
    allowed_enum_values: Optional[tuple[str, ...]]


@dataclass(frozen=True)
class CriticalFact:
    name: str
    category: str
    canonical_value: Optional[str]
    superseded_value: Optional[str] = None


@dataclass(frozen=True)
class KnownFactViolationCheck:
    check_id: str
    description: str


@dataclass(frozen=True)
class DatasetFacts:
    critical_facts: tuple[CriticalFact, ...]
    known_fact_violation_checks: tuple[KnownFactViolationCheck, ...]


def _sha256_of_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _conversations_path(version: str) -> Path:
    return DATA_DIR / f"conversations_{version}.json"


def _questions_path(version: str) -> Path:
    return DATA_DIR / f"golden_questions_{version}.json"


def _facts_path(version: str) -> Path:
    return DATA_DIR / f"golden_facts_{version}.json"


def dataset_version_hashes(version: str = "v1") -> dict[str, str]:
    """SHA-256 of each of the three source files for this version -- recorded
    in the experiment manifest so the exact dataset content used is always
    traceable, never just a version label."""
    return {
        "conversations": _sha256_of_json(_conversations_path(version)),
        "golden_questions": _sha256_of_json(_questions_path(version)),
        "golden_facts": _sha256_of_json(_facts_path(version)),
    }


def load_conversation(dataset_id: str, version: str = "v1") -> Conversation:
    data = _load_json(_conversations_path(version))
    for entry in data["conversations"]:
        if entry["dataset_id"] == dataset_id:
            messages = tuple(Message(role=m["role"], content=m["content"]) for m in entry["messages"])
            return Conversation(dataset_id=dataset_id, scenario=entry["scenario"], messages=messages)
    raise KeyError(f"No conversation found for dataset_id '{dataset_id}' in version '{version}'")


def load_golden_questions(dataset_id: str, version: str = "v1") -> list[GoldenQuestion]:
    data = _load_json(_questions_path(version))
    entries = data["questions"][dataset_id]
    questions = []
    for entry in entries:
        enum_values = entry.get("allowed_enum_values")
        questions.append(
            GoldenQuestion(
                question_id=entry["question_id"],
                question=entry["question"],
                expected_answer=entry["expected_answer"],
                answer_type=entry["answer_type"],
                allowed_enum_values=tuple(enum_values) if enum_values else None,
            )
        )
    return questions


def load_dataset_facts(dataset_id: str, version: str = "v1") -> DatasetFacts:
    data = _load_json(_facts_path(version))
    entry = data["facts"][dataset_id]
    critical_facts = tuple(
        CriticalFact(
            name=f["name"], category=f["category"], canonical_value=f.get("canonical_value"),
            superseded_value=f.get("superseded_value"),
        )
        for f in entry["critical_facts"]
    )
    checks = tuple(
        KnownFactViolationCheck(check_id=c["check_id"], description=c["description"])
        for c in entry["known_fact_violation_checks"]
    )
    return DatasetFacts(critical_facts=critical_facts, known_fact_violation_checks=checks)


def render_transcript(conversation: Conversation) -> str:
    """The Full History baseline's context payload: a plain role-labeled
    transcript of every message, verbatim -- no summarization, no
    truncation, no reordering."""
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in conversation.messages)


def check_dataset_integrity(dataset_id: str, version: str = "v1") -> list[str]:
    """Returns a list of integrity problems (empty = fully valid). Pure,
    offline, deterministic -- no model call is ever needed to validate a
    dataset fixture (Part 4.4)."""
    problems: list[str] = []
    conversation = load_conversation(dataset_id, version)
    transcript = render_transcript(conversation)
    questions = load_golden_questions(dataset_id, version)
    facts = load_dataset_facts(dataset_id, version)

    if len(conversation.messages) < 24:
        problems.append(f"{dataset_id}: expected >= 24 messages, found {len(conversation.messages)}")

    seen_question_ids = set()
    for q in questions:
        if q.question_id in seen_question_ids:
            problems.append(f"{dataset_id}: duplicate question_id '{q.question_id}'")
        seen_question_ids.add(q.question_id)
        if q.answer_type not in ANSWER_TYPES:
            problems.append(f"{dataset_id}: question '{q.question_id}' has unsupported answer_type '{q.answer_type}'")
        if q.answer_type == "unknown" and q.expected_answer != "NOT_STATED":
            problems.append(f"{dataset_id}: 'unknown' question '{q.question_id}' must have expected_answer NOT_STATED")
        elif q.answer_type == "enum":
            # Enum canonical forms (e.g. "AFTERNOON") are deliberately
            # upper-cased labels for a concept the transcript expresses in
            # natural language (e.g. "afternoon") -- grounded case-insensitively.
            if q.expected_answer.lower() not in transcript.lower():
                problems.append(
                    f"{dataset_id}: enum expected_answer '{q.expected_answer}' for question '{q.question_id}' "
                    "is not grounded anywhere in the source conversation"
                )
        elif q.answer_type != "unknown" and q.expected_answer not in transcript:
            problems.append(
                f"{dataset_id}: expected_answer '{q.expected_answer}' for question '{q.question_id}' "
                "is not grounded anywhere in the source conversation"
            )
        if q.answer_type == "iso_date" and q.expected_answer != "NOT_STATED":
            try:
                date.fromisoformat(q.expected_answer)
            except ValueError:
                problems.append(f"{dataset_id}: question '{q.question_id}' expected_answer is not a valid ISO date")

    seen_fact_names = set()
    for fact in facts.critical_facts:
        if fact.name in seen_fact_names:
            problems.append(f"{dataset_id}: duplicate critical fact name '{fact.name}'")
        seen_fact_names.add(fact.name)
        if fact.canonical_value is None:
            # A deliberately unstated fact: the canonical (non-)value must
            # genuinely be absent from the transcript, never accidentally present.
            continue
        # Case-insensitive: a fact's canonical label (e.g. "AFTERNOON", "NO")
        # is often an upper-cased/boolean-style label for a concept the
        # transcript expresses in natural language (e.g. "afternoon", "No,").
        if fact.canonical_value.lower() not in transcript.lower():
            problems.append(f"{dataset_id}: critical fact '{fact.name}' canonical_value is not grounded in the transcript")
        if fact.superseded_value is not None and fact.superseded_value.lower() not in transcript.lower():
            problems.append(f"{dataset_id}: critical fact '{fact.name}' superseded_value is not grounded in the transcript")
        if fact.superseded_value is not None and fact.superseded_value == fact.canonical_value:
            problems.append(f"{dataset_id}: critical fact '{fact.name}' canonical_value equals its own superseded_value")

    if not any(fact.canonical_value is None for fact in facts.critical_facts):
        problems.append(f"{dataset_id}: expected at least one deliberately unstated critical fact")

    return problems
