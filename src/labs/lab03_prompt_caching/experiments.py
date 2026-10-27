"""Lab03 experiments: prompt caching, cache-friendly prompt design, and
token economics on Groq.

Prompt caching on Groq (as of Oct 2026, see https://console.groq.com/docs/
prompt-caching) is:
  - supported only on openai/gpt-oss-20b, openai/gpt-oss-120b, and
    openai/gpt-oss-safeguard-20b;
  - fully automatic/implicit -- there is no cache-control request
    parameter, and it cannot be disabled;
  - keyed on an EXACT leading-byte prefix match, so only identical leading
    text (down to the very first character) can hit;
  - subject to a minimum cacheable prompt length that "varies by model,
    from 128 to 1024 tokens" per the official docs, which do not give an
    exact per-model number -- this module deliberately does not invent
    one. STATIC_KNOWLEDGE_BASE + STATIC_RESPONSE_RULES below is sized to
    comfortably clear the upper end of that documented range for ANY of
    the three supported models, rather than guessing which model-specific
    threshold applies.
  - not guaranteed: provider-side cache eviction, load, and a documented
    ~2h TTL mean a hit is an *observation*, never something this code
    should assume, force, or treat as a bug when absent. A single miss --
    even on the very first repeated_prefix call of a fresh run -- is not
    evidence of a defect.

Two paired conditions, both built from the SAME static knowledge base and
the SAME set of paired questions, so the only systematically varied factor
is whether the leading prefix can repeat byte-for-byte across calls:

  - repeated_prefix (cache-friendly): a constant, fixed-length tag + the
    knowledge base + the response rules, IDENTICAL on every call.
  - cache_unfriendly: a changing, but SAME-LENGTH, tag in that exact same
    leading position, so no exact-prefix match is possible, while the
    overall prompt length stays comparable to repeated_prefix's.

Message structure (both conditions): [STABLE/CHANGING TAG][STATIC
KNOWLEDGE BASE][STATIC RESPONSE RULES] as the system message, then
[DYNAMIC USER QUESTION][ANSWER FORMAT INSTRUCTION] as the user message --
static content first, dynamic content last, as Groq's own docs recommend
for cache-friendly prompt layout.
"""

from __future__ import annotations

from typing import NamedTuple

CONDITION_REPEATED_PREFIX = "repeated_prefix"
CONDITION_CACHE_UNFRIENDLY = "cache_unfriendly"
CONDITION_ORDER = (CONDITION_REPEATED_PREFIX, CONDITION_CACHE_UNFRIENDLY)

# A realistic enterprise customer-support knowledge base: several policy
# sections, each containing exactly one fact this lab asks about (see
# PAIRED_QUESTIONS below), plus FAQ/response-guideline filler that is
# genuinely part of the business content (not meaningless repeated text --
# a deliberate departure from the old STATIC_PREFIX, which was the same
# paragraph repeated 10 times with no educational content of its own).
STATIC_KNOWLEDGE_BASE = """\
=== ACME CLOUD -- ENTERPRISE CUSTOMER SUPPORT KNOWLEDGE BASE (v2026.3) ===

SECTION 1: SUPPORT POLICY OVERVIEW
Acme Cloud provides tiered customer support for all paid API platform
subscribers. Every subscriber account is assigned a support tier (Standard,
Business, or Enterprise) at signup, which determines response-time targets
and the channels available to that account (email, chat, or phone).
Support agents must always consult the subscriber's current tier before
quoting any response-time commitment, since tiers are the primary driver
of the service levels described in this document. All support
interactions are logged in the ticketing system described in Section 5,
and agents must record the resolution outcome before closing any ticket.

SECTION 2: REFUND POLICY
Refunds are evaluated case by case and require an explicit approval step
from a support agent before being issued. Once a refund has been approved,
Acme Cloud's billing system issues the refund to the original payment
method. The maximum number of business days within which an approved
refund must be issued is 5 business days. This 5-business-day maximum
applies uniformly across all support tiers (Standard, Business, and
Enterprise) and all supported payment methods; it is not shortened or
extended by tier. Partial refunds for partial-month usage follow the same
5-business-day maximum once approved. If a refund has not arrived within
5 business days of approval, the subscriber should open a billing escalation
ticket referencing the original refund approval.

SECTION 3: ESCALATION POLICY
Most support requests are first handled by an automated resolution system
that attempts to resolve common issues (password resets, billing lookups,
quota questions) without a human agent. When the automated system attempts
to resolve an issue and fails, that failure is logged as one failed
automated resolution attempt. After 2 consecutive failed automated
resolution attempts on the same ticket, the ticket is automatically
escalated to a human agent; the subscriber is not required to request
escalation manually once this threshold is reached. A successful automated
resolution resets the failure count for that ticket back to zero, so only
CONSECUTIVE failures count toward the 2-attempt escalation threshold.
Enterprise-tier subscribers may additionally request immediate escalation
to a human agent at any time, bypassing the automated system entirely.

SECTION 4: SLA DEFINITIONS (SERVICE LEVEL AGREEMENT)
Acme Cloud's support SLA defines first-response and resolution targets per
ticket priority (see Section 5 for the priority definitions themselves).
For a Standard-priority ticket, the target for first response -- the first
substantive reply from a human or automated agent, not an auto-acknowledgment
receipt -- is within 24 hours of ticket submission. This 24-hour first-response
target is measured from the timestamp the ticket is created in the
ticketing system, not from when the subscriber first attempted to reach
support through any other channel. Resolution-time targets (how long until
the issue is fully closed, not just first responded to) are tracked
separately per priority level and are not the same number as the
24-hour first-response target above -- the two should never be confused
when reporting SLA compliance.

SECTION 5: TICKET PRIORITY RULES
Every support ticket is assigned exactly one of 3 priority levels: High,
Medium, or Low. There are no other priority levels in the current
ticketing system; "Critical" and "Urgent" are informal terms sometimes
used by subscribers but are not valid priority values in the system of
record. High priority is reserved for complete service outages affecting
production traffic. Medium priority covers degraded performance or partial
feature unavailability. Low priority covers cosmetic issues, documentation
questions, and feature requests. A ticket's priority may be changed by a
support agent if the subscriber's initial priority selection does not
match the actual technical severity of the issue, but it always remains
exactly one of these 3 levels at any given time.

SECTION 6: FREQUENTLY ASKED QUESTIONS
Q: Can I request a refund for a service I no longer use but have not
formally cancelled? A: No -- a refund request first requires account
cancellation, and only then is the refund evaluated under the policy in
Section 2. Q: Does opening a second ticket for the same issue speed up
escalation? A: No -- it does not affect the 2-consecutive-failure
escalation threshold in Section 3, and duplicate tickets are typically
merged by support agents. Q: Can a Standard-tier subscriber upgrade mid-
ticket to get faster SLA targets? A: No -- the SLA target in Section 4
that applies is the one in effect for the subscriber's tier at the moment
the ticket was created, not at the moment of any later tier change. Q: Who
decides a ticket's priority level? A: The subscriber selects an initial
priority from the 3 levels in Section 5 when filing the ticket, but a
support agent may re-classify it afterward. Q: Is phone support available
on the Standard tier? A: No -- phone support is only available on the
Business and Enterprise tiers; Standard-tier subscribers use email or chat.

SECTION 7: RESPONSE GUIDELINES FOR SUPPORT AGENTS
Agents must be concise, professional, and specific in every response: cite
the exact policy section being applied (e.g. "per Section 2, refunds are
issued within 5 business days") rather than paraphrasing loosely. Agents
must never promise a response-time or refund-time commitment that is
faster than the numbers defined in Sections 2 and 4 above, regardless of
how the subscriber frames the urgency of their request. When a subscriber
asks a factual question that is directly answered by one of the numbered
sections above, the agent's answer must match that section's number
exactly -- support quality reviews specifically check for this.
"""

STATIC_RESPONSE_RULES = """\
=== OUTPUT RULES FOR THIS EXERCISE ===
You will be asked ONE factual question whose answer is a single integer
that appears explicitly in the knowledge base above. Answer using ONLY the
knowledge base above. Do not explain your reasoning in the final answer.
Do not restate the question. Your final answer must be exactly one line
containing only that integer, with no words, units, or punctuation.
"""


class PairedQuestion(NamedTuple):
    question: str
    expected_answer: str


# Four paired questions, each answered by a different numbered section of
# STATIC_KNOWLEDGE_BASE above, so the dynamic suffix genuinely changes
# per-repetition while the expected answer stays deterministic and
# verifiable. Both conditions in a pair use the SAME question (see
# labs.lab03_prompt_caching.matrix.build_cache_plan), satisfying the
# pairing requirement that the only systematically varied factor is
# whether the prefix can repeat.
PAIRED_QUESTIONS: tuple[PairedQuestion, ...] = (
    PairedQuestion(
        "Per Section 2 (Refund Policy) of the knowledge base, what is the maximum number "
        "of business days within which an approved refund must be issued?",
        "5",
    ),
    PairedQuestion(
        "Per Section 3 (Escalation Policy) of the knowledge base, after how many consecutive "
        "failed automated resolution attempts is a ticket escalated to a human agent?",
        "2",
    ),
    PairedQuestion(
        "Per Section 4 (SLA Definitions) of the knowledge base, within how many hours of "
        "ticket submission should a Standard-priority ticket receive its first response?",
        "24",
    ),
    PairedQuestion(
        "Per Section 5 (Ticket Priority Rules) of the knowledge base, how many distinct "
        "ticket priority levels are defined in the current ticketing system?",
        "3",
    ),
)

ANSWER_FORMAT_INSTRUCTION = "Return only the integer answer. Do not include any words, units, or punctuation."

# Fixed-length (16-char) "core" tags, deliberately the same length in both
# conditions so a comparison of prompt length is not confounded by the tag
# itself (Groq's own prefix match is byte-exact, so even a single differing
# leading character breaks it -- these differ from position 0).
_TAG_CORE_LENGTH = 16
_CACHE_FRIENDLY_CORE = "STABLEPREFIXTAG0"  # constant: reused byte-for-byte on every repeated_prefix call
assert len(_CACHE_FRIENDLY_CORE) == _TAG_CORE_LENGTH


def _cache_unfriendly_core(pair_index: int) -> str:
    """A changing 16-character tag core, unique per pair_index, that breaks
    the exact-prefix match from the very first character onward."""
    return f"CALLNONCE{pair_index:07d}"


assert len(_cache_unfriendly_core(1)) == _TAG_CORE_LENGTH

CACHE_FRIENDLY_TAG = f"[SESSION:{_CACHE_FRIENDLY_CORE}]"


def _cache_unfriendly_tag(pair_index: int) -> str:
    return f"[SESSION:{_cache_unfriendly_core(pair_index)}]"


assert len(CACHE_FRIENDLY_TAG) == len(_cache_unfriendly_tag(1))


def tag_for_condition(condition_id: str, pair_index: int) -> str:
    if condition_id == CONDITION_REPEATED_PREFIX:
        return CACHE_FRIENDLY_TAG
    if condition_id == CONDITION_CACHE_UNFRIENDLY:
        return _cache_unfriendly_tag(pair_index)
    raise ValueError(f"Unknown condition_id '{condition_id}'; must be one of {CONDITION_ORDER}")


def build_cache_messages(condition_id: str, pair_index: int, question_index: int) -> list[dict]:
    """Builds the [system: tag + knowledge base + rules][user: question +
    format instruction] message pair for one planned call.

    `question_index` is taken as given (not recomputed here) so the exact
    same question a dry-run plan displays is guaranteed to be the one
    actually sent -- the plan is the single source of truth for pairing.
    """
    tag = tag_for_condition(condition_id, pair_index)
    paired_question = PAIRED_QUESTIONS[question_index % len(PAIRED_QUESTIONS)]
    system_content = f"{tag}\n\n{STATIC_KNOWLEDGE_BASE}\n\n{STATIC_RESPONSE_RULES}"
    user_content = f"{paired_question.question}\n\n{ANSWER_FORMAT_INSTRUCTION}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def expected_answer_for(question_index: int) -> str:
    return PAIRED_QUESTIONS[question_index % len(PAIRED_QUESTIONS)].expected_answer
