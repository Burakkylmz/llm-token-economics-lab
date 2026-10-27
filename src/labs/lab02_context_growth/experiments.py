"""Lab02 context fixtures: how growing/irrelevant context affects input
tokens, cost, latency, reasoning overhead, and answer correctness.

All three context conditions ask the EXACT same question with the EXACT
same expected answer, so completion content should be directly comparable;
what varies is the size and relevance of the context placed before that
question. Keeping prompt/question/expected-answer/system-instruction/output
format identical across conditions is this experiment's core control (see
Part 5.4 "Control Requirements" in this lab's design task).

Scientific limitation (documented, not hidden): relevance change is also a
semantic content change, so this is not a perfectly isolated causal
experiment. Known confounders that are NOT separated out by this design:
information position (the canonical fact sits at the START of
expanded_context but at the END of irrelevant_context -- see
EXPANDED_RELEVANT_CONTEXT/IRRELEVANT_CONTEXT below), relevant-fact
repetition count, output length, reasoning-token variability, and
provider load at the time of measurement. These are restated in the
generated analysis_report.md, never silently assumed away.
"""

from __future__ import annotations

QUESTION = "According to the policy, what is the maximum number of retry attempts, excluding the initial attempt?"

# Deliberately unambiguous (the original "retries up to 5 times total" text
# conflated "total attempts" with "retry attempts" -- this fixture states
# both explicitly, and the expected answer is only correct under this exact
# reading: 5 total attempts = 1 initial attempt + 4 retry attempts).
RELEVANT_CONTEXT = (
    "Our API client retry policy:\n"
    "- Maximum total attempts: 5\n"
    "- Initial attempt: 1\n"
    "- Maximum retry attempts: 4\n"
    "On a transient failure (HTTP 429 or 5xx), the client retries using "
    "exponential backoff with full jitter, starting at a 1 second base "
    "delay and capping at 30 seconds."
)

EXPECTED_ANSWER = "4"

ANSWER_FORMAT_INSTRUCTION = "Return only the integer number of retry attempts. Do not include any other text."

SYSTEM_INSTRUCTION = f"Answer using only the information in the user's context. {ANSWER_FORMAT_INSTRUCTION}"

# Additional content for expanded_context: genuinely related to the retry
# policy (transient-failure detection, backoff mechanics, timeout handling,
# logging) but none of it repeats or restates the canonical "4" answer --
# the point is to grow relevant context without artificially cueing the
# model via repetition.
_ADDITIONAL_RELEVANT_PARAGRAPHS = [
    "Transient failures are detected when the HTTP response status code is "
    "429 (rate limited) or falls within the 5xx server-error range; any "
    "other status code is treated as a non-transient failure and is never "
    "retried, regardless of how many retry attempts remain available.",
    "The exponential backoff delay doubles after each failed attempt, "
    "starting from the configured base delay, and is capped at the "
    "configured maximum delay so a long-running outage cannot cause an "
    "unbounded wait between attempts.",
    "Full jitter means each actual wait time is drawn uniformly at random "
    "between zero and the computed exponential backoff value for that "
    "attempt, rather than waiting the full computed value every time; this "
    "spreads out retries from many clients so they do not all retry at "
    "exactly the same moment.",
    "A request that exceeds the configured request timeout is treated the "
    "same way as a transient HTTP failure and is eligible for a retry "
    "attempt, provided the retry attempt budget described above has not "
    "already been exhausted.",
    "Every retry attempt is logged with the attempt number, the computed "
    "delay before that attempt, and the HTTP status code or timeout that "
    "triggered it, so the full retry history of a request can be "
    "reconstructed later from the logs during an incident review.",
    "If every retry attempt is exhausted without a successful response, "
    "the client surfaces the final failure to the caller together with the "
    "status code and attempt count from the last attempt, rather than "
    "retrying indefinitely.",
]

EXPANDED_RELEVANT_CONTEXT = RELEVANT_CONTEXT + "\n\n" + "\n\n".join(_ADDITIONAL_RELEVANT_PARAGRAPHS * 4)

# Filler clearly unrelated to the retry question, to simulate "context
# bloat" (irrelevant information inflating the prompt). The canonical
# retry-policy fact is still present (at the end), so answer quality is
# held constant across conditions -- only size/relevance/position varies.
_IRRELEVANT_PARAGRAPHS = [
    "The cafeteria on the third floor serves lunch from 11:30 to 14:00 and "
    "the vegetarian option changes weekly according to a rotating seasonal "
    "menu planned by the facilities team.",
    "Meeting room B on the fifth floor was recently repainted and now "
    "requires a badge swipe at the door in addition to the usual calendar "
    "booking, following a facilities policy change announced last quarter.",
    "The office plant-watering rotation is managed by a volunteer sign-up "
    "sheet near the elevator bank, and reminders are sent out every other "
    "Friday afternoon to whoever is scheduled for the following week.",
    "Parking permits for the north lot are renewed annually every March "
    "and require a short online form plus proof of current vehicle "
    "registration to be submitted to the facilities administration team.",
    "The building's quarterly fire-drill schedule is posted on the lobby "
    "noticeboard, and attendance is tracked by badge scan at each of the "
    "designated assembly points outside the building.",
]

IRRELEVANT_CONTEXT = "\n\n".join(_IRRELEVANT_PARAGRAPHS * 6) + "\n\n" + RELEVANT_CONTEXT

CONTEXT_CONDITION_ORDER = ("minimal_context", "expanded_context", "irrelevant_context")

CONDITIONS: dict[str, str] = {
    "minimal_context": RELEVANT_CONTEXT,
    "expanded_context": EXPANDED_RELEVANT_CONTEXT,
    "irrelevant_context": IRRELEVANT_CONTEXT,
}


def build_context_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]
