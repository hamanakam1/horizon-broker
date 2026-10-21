"""Job-side data contracts for The Horizon Broker.

These models are the only sanctioned wire format between the ingress API, the
scheduler, the PD-DAG router and the recovery subsystem.

Two scheduling tiers exist:

``INTERACTIVE``
    A human (or a synchronous CLI) is waiting. The broker dispatches these
    immediately, subject to the interactive concurrency budget.

``FLEX``
    Sleep-time compute. The broker deliberately delays these to build larger
    batches, maximising Model Flop Utilization and amortising the cost of
    streaming model weights out of unified memory.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from horizon_broker.contracts.recovery import ResumePoint

SCHEMA_VERSION = "1.0"
CHARS_PER_TOKEN = 4
DEFAULT_OUTPUT_TOKENS = 512


def utcnow() -> datetime:
    return datetime.now(UTC)


class BrokerModel(BaseModel):
    """Base contract: strict fields, validated on assignment, no silent extras."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PriorityTier(StrEnum):
    """Scheduling tier of a request."""

    INTERACTIVE = "interactive"
    FLEX = "flex"


class JobState(StrEnum):
    """Lifecycle of an :class:`InferenceJob`."""

    QUEUED = "queued"
    WINDOWING = "windowing"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"
    CANCELLED = "cancelled"


class ExecutionPhase(StrEnum):
    """Phase of a (possibly split) execution.

    PD-DAG disaggregation runs the prefill phase on a compute-bound node and
    the decode phase on a memory-bandwidth-bound node.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    MONOLITHIC = "monolithic"


class ChatMessage(BrokerModel):
    """One chat turn in a structured prompt."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class SamplingParams(BrokerModel):
    """Sampler configuration forwarded to the inference engine."""

    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    num_predict: int | None = Field(default=None, ge=1, description="Hard cap on generated tokens.")
    seed: int | None = None
    stop: tuple[str, ...] = ()
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)

    def to_engine_params(self) -> dict[str, Any]:
        """Ollama ``/api/generate`` option block."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = list(self.stop)
        return options


class InferencePayload(BrokerModel):
    """The inference request itself, decoupled from scheduling concerns."""

    model: str = Field(description="Model identifier understood by the target engine.")
    prompt: str = Field(default="")
    messages: list[ChatMessage] = Field(default_factory=list)
    system: str | None = None
    parameters: SamplingParams = Field(default_factory=SamplingParams)
    expected_output_tokens: int | None = Field(
        default=None, ge=1, description="Declared generation size; drives PD-DAG routing."
    )
    prompt_tokens_hint: int | None = Field(
        default=None, ge=1, description="Declared prompt size; avoids guessing when the caller knows."
    )

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @property
    def estimated_prompt_tokens(self) -> int:
        """Prompt size in tokens; uses the declared hint when available."""
        if self.prompt_tokens_hint is not None:
            return self.prompt_tokens_hint
        chars = len(self.prompt) + len(self.system or "")
        for message in self.messages:
            chars += len(message.content)
        return max(1, chars // CHARS_PER_TOKEN)

    @property
    def estimated_output_tokens(self) -> int:
        """Generation size in tokens."""
        if self.expected_output_tokens is not None:
            return self.expected_output_tokens
        if self.parameters.num_predict is not None:
            return self.parameters.num_predict
        return DEFAULT_OUTPUT_TOKENS

    @property
    def prompt_hash(self) -> str:
        """sha256 of the canonical prompt text; guards snapshot reuse."""
        parts = [self.prompt, self.system or ""]
        parts.extend(f"{m.role}:{m.content}" for m in self.messages)
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def to_engine_request(self, model_override: str | None = None) -> dict[str, Any]:
        """Render the payload as an Ollama ``/api/generate`` request body."""
        body: dict[str, Any] = {
            "model": model_override or self.model,
            "prompt": self.prompt,
            "stream": True,
            "options": self.parameters.to_engine_params(),
        }
        if self.system:
            body["system"] = self.system
        return body

    def to_chat_request(self, model_override: str | None = None) -> dict[str, Any]:
        """Render the payload as an Ollama ``/api/chat`` request body."""
        messages = [m.model_dump() for m in self.messages]
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        body: dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
            "stream": True,
            "options": self.parameters.to_engine_params(),
        }
        return body


class InferenceJob(BrokerModel):
    """A unit of inference work tracked by The Horizon Broker.

    The four fields that define the scheduling contract are ``priority_tier``,
    ``payload``, ``max_acceptable_latency_ms`` and ``kv_cache_state_id``.
    """

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    priority_tier: PriorityTier = Field(
        description="INTERACTIVE for latency-bound work, FLEX for sleep-time compute."
    )
    payload: InferencePayload
    max_acceptable_latency_ms: int = Field(
        ge=1, le=86_400_000, description="Latency budget measured from submission."
    )
    kv_cache_state_id: str | None = Field(
        default=None,
        description="EdgeRecord document id of the KV-cache/resume state for this job.",
    )
    state: JobState = JobState.QUEUED
    phase: ExecutionPhase = ExecutionPhase.MONOLITHIC
    attempt: int = Field(default=0, ge=0, description="Number of dispatch attempts made so far.")
    submitted_at: datetime = Field(default_factory=utcnow)
    deadline_at: datetime | None = Field(
        default=None, description="Absolute deadline; derived from the latency budget when omitted."
    )
    not_before: datetime | None = Field(
        default=None, description="Earliest time the job may be dispatched (sleep-time gating)."
    )
    assigned_node_id: str | None = None
    caller: str = Field(default="unknown", description="Originating agent or subsystem.")
    tags: tuple[str, ...] = ()
    resume_point: ResumePoint | None = Field(
        default=None, description="Populated when the job is re-queued after a fault."
    )
    result_text: str | None = None
    error: str | None = None
    schema_version: str = Field(default=SCHEMA_VERSION)

    @model_validator(mode="before")
    @classmethod
    def _derive_deadline(cls, data: Any) -> Any:
        """Fill ``deadline_at`` from the latency budget without touching the instance."""
        if isinstance(data, dict):
            latency = data.get("max_acceptable_latency_ms")
            if latency is not None and data.get("deadline_at") is None:
                submitted = data.get("submitted_at")
                base = submitted if isinstance(submitted, datetime) else utcnow()
                if base.tzinfo is None:
                    base = base.replace(tzinfo=UTC)
                data = {**data, "deadline_at": base + timedelta(milliseconds=int(latency))}
        return data

    @field_validator("submitted_at", "deadline_at", "not_before", mode="after")
    @classmethod
    def _require_tz(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @property
    def is_resumed(self) -> bool:
        """True when the job carries recovery state from a previous node."""
        return self.resume_point is not None

    def remaining_latency_ms(self, now: datetime | None = None) -> int:
        """Milliseconds left in the latency budget (may be negative)."""
        if self.deadline_at is None:
            return self.max_acceptable_latency_ms
        current = now or utcnow()
        return int((self.deadline_at - current).total_seconds() * 1000)

    def is_expired(self, now: datetime | None = None) -> bool:
        """True when the latency budget has been exhausted."""
        return self.remaining_latency_ms(now) <= 0
