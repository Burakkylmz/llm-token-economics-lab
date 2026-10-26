"""Shared measurement schema used by every lab.

Fields the Groq API does not report for a given call (e.g. reasoning_tokens
on a model that doesn't support reasoning, or cached_tokens on a model
without prompt-caching support) MUST be left as `None`, never defaulted to
0 or fabricated. A `None` means "unavailable/not reported", which is a
different fact from "reported as zero".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenCostMetrics:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    # Why the model stopped generating ("stop", "length", or whatever else the
    # provider returns) -- reported as-is, never inferred from token counts
    # alone (e.g. completion_tokens == max_tokens is NOT proof of truncation
    # without this field confirming it).
    finish_reason: Optional[str] = None
    api_call_count: int = 0


@dataclass
class DerivedMetrics:
    """Values this project calculates from reported fields -- never numbers
    the provider itself sent. Kept separate from TokenCostMetrics so a reader
    never mistakes a derived number for an API-reported one.
    """

    # completion_tokens - reasoning_tokens, only when both are known and
    # consistent. Deliberately not called `visible_output_tokens`: tokens
    # outside the reasoning span are not guaranteed to equal the token count
    # of the final answer text a user sees.
    non_reasoning_completion_tokens: Optional[int] = None
    # Character length of the final answer text -- a lightweight diagnostic,
    # not a substitute for a token count (chars-per-token varies by content).
    final_answer_char_count: Optional[int] = None


@dataclass
class PerformanceMetrics:
    client_elapsed_ms: Optional[float] = None
    provider_reported_processing_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    first_visible_content_ms: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    requests_per_second: Optional[float] = None
    throughput: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


@dataclass
class ExecutionMetadata:
    experiment_id: str
    run_id: str
    model_id: str
    execution_mode: str
    timestamp: str
    model_parameters: dict = field(default_factory=dict)
    # 1-based repetition number within an experiment (e.g. 3 of 5), distinct
    # from run_id (a unique id per call). Optional: only labs that repeat a
    # condition multiple times need it; everyone else leaves it None.
    run_index: Optional[int] = None
    # Echoed back by some providers for debugging/support purposes; not
    # requested by the caller, so left None when the SDK response doesn't
    # carry it rather than guessed.
    system_fingerprint: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ExperimentResult:
    """One measured observation: a single API call or one repetition of an experiment."""

    metadata: ExecutionMetadata
    tokens_and_cost: TokenCostMetrics
    performance: PerformanceMetrics
    derived: DerivedMetrics = field(default_factory=DerivedMetrics)
