"""Lab06's shared task: Production API Incident Explanation.

Replaces the old open-ended "write a short story" prompt (no controllable
length, no deterministic completeness check) with a fixed, deterministic
source text the model must explain under exactly five required headings.
This produces enough visible text to make streaming's incremental-delivery
behavior observable, while still allowing an auditable completeness check
(see labs.lab06_streaming.evaluation) -- never a semantic/factual-accuracy
claim about the explanation's content, only a structural one (are the five
headings present, each with a non-empty explanation).

The SAME messages are sent for both the streaming and the non-streaming
condition in a pair (Part D's request-equivalence rule) -- the only
intended difference between the two calls is the `stream` transport flag
itself.
"""

from __future__ import annotations

INCIDENT_DESCRIPTION = """\
At 14:32 UTC, the checkout-service API began returning HTTP 504 Gateway \
Timeout errors for approximately 8 minutes. The root cause was connection \
pool exhaustion in the payment-gateway client, caused by a recent \
deployment that removed the client's connection timeout configuration. \
Roughly 3,200 checkout requests failed during the incident window. The \
on-call engineer restarted the payment-gateway client pods, which restored \
normal operation within 90 seconds of the restart.\
"""

# Order matters: this is also the required output order the prompt asks
# for, and the order labs.lab06_streaming.evaluation checks sections in.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Incident Summary",
    "Possible Causes",
    "User Impact",
    "Immediate Mitigation",
    "Prevention Measures",
)


def build_incident_messages() -> list[dict]:
    instruction = (
        "Below is a short production incident description. Explain it under EXACTLY "
        "the following five headings, each followed by a colon and a non-empty "
        "explanation of at least one full sentence, in this exact order, and with no "
        "other text before, between, or after them:\n\n"
        + "\n".join(f"{heading}:" for heading in REQUIRED_SECTIONS)
        + "\n\nIncident description:\n"
        + INCIDENT_DESCRIPTION
    )
    return [{"role": "user", "content": instruction}]
