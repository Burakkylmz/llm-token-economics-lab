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
    api_call_count: int = 0


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
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ExperimentResult:
    """One measured observation: a single API call or one repetition of an experiment."""

    metadata: ExecutionMetadata
    tokens_and_cost: TokenCostMetrics
    performance: PerformanceMetrics
