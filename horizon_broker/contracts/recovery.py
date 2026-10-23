"""Fault-recovery contracts.

The payload committed to EdgeRecord when a job is interrupted. A snapshot
carries the minimum information required to *resume* a generation on another
node: how far the decode loop had progressed, the partial tokens already
emitted, and a reference to the KV-cache state that was persisted separately.

``horizon_broker.contracts.jobs`` imports this module; this module must never
import ``jobs`` at runtime (only under ``TYPE_CHECKING``) to keep the contract
graph acyclic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


class BrokerModel(BaseModel):
    """Base contract: strict fields, validated on assignment, no silent extras."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PartialToken(BrokerModel):
    """A single token already produced by the interrupted generation."""

    index: int = Field(ge=0, description="Zero-based position in the completion stream.")
    token_id: int | None = Field(default=None, description="Vocabulary id when known.")
    text: str = Field(default="", description="Decoded surface form of the token.")
    logprob: float | None = Field(default=None, description="Log probability at emission time.")

    @model_validator(mode="before")
    @classmethod
    def _reject_boolean_numerics(cls, data: Any) -> Any:
        """Keep malformed persisted token metadata from changing replay state."""
        if isinstance(data, dict):
            for field_name in ("index", "token_id", "logprob"):
                if isinstance(data.get(field_name), bool):
                    raise ValueError(f"{field_name} must be numeric, not boolean")
            index = data.get("index")
            token_id = data.get("token_id")
            logprob = data.get("logprob")
            text = data.get("text", "")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("index must be a non-negative integer")
            if token_id is not None and (
                not isinstance(token_id, int) or isinstance(token_id, bool)
            ):
                raise ValueError("token_id must be an integer when supplied")
            if logprob is not None:
                if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
                    raise ValueError("logprob must be a finite number when supplied")
                try:
                    finite_logprob = math.isfinite(float(logprob))
                except (OverflowError, ValueError):
                    finite_logprob = False
                if not finite_logprob:
                    raise ValueError("logprob must be a finite number when supplied")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
        return data


class ResumePoint(BrokerModel):
    """Exact resumption coordinates for an interrupted generation.

    ``decode_step`` is the authoritative position in the decode loop. The
    ``kv_cache_ref`` points at the KV-cache blob persisted in EdgeRecord; it is
    the same identifier surfaced as ``InferenceJob.kv_cache_state_id``.
    """

    decode_step: int = Field(ge=0, description="Decode iterations completed before the fault.")
    generated_token_count: int = Field(ge=0, description="Tokens emitted before the fault.")
    kv_cache_ref: str | None = Field(default=None, description="EdgeRecord document id of the KV cache.")
    kv_cache_state_id: str | None = Field(default=None, description="Alias of kv_cache_ref.")
    prompt_hash: str = Field(description="sha256 of the canonical prompt, guards against payload drift.")
    model: str = Field(default="", description="Model that produced the state.")
    checksum_algorithm: str = Field(default="sha256")

    @model_validator(mode="before")
    @classmethod
    def _reject_boolean_counters(cls, data: Any) -> Any:
        """Prevent boolean replay coordinates from becoming valid counters."""
        if isinstance(data, dict):
            for field_name in ("decode_step", "generated_token_count"):
                if isinstance(data.get(field_name), bool):
                    raise ValueError(f"{field_name} must be an integer, not boolean")
                if field_name in data and not isinstance(data.get(field_name), int):
                    raise ValueError(f"{field_name} must be an integer")
        return data

    @model_validator(mode="after")
    def _require_prompt_identity(self) -> ResumePoint:
        """Keep the replay guard meaningful when a snapshot is resumed."""
        if not re.fullmatch(r"[0-9a-f]{64}", self.prompt_hash):
            raise ValueError("prompt_hash must be a lowercase SHA-256 hex digest")
        for field_name in ("kv_cache_ref", "kv_cache_state_id"):
            value = getattr(self, field_name)
            if value is not None:
                if not value.strip():
                    raise ValueError(f"{field_name} must not be empty when supplied")
                if value != value.strip():
                    raise ValueError(f"{field_name} must not have surrounding whitespace")
        if (
            self.kv_cache_ref is not None
            and self.kv_cache_state_id is not None
            and self.kv_cache_ref != self.kv_cache_state_id
        ):
            raise ValueError("kv_cache_ref and kv_cache_state_id must match")
        if self.model and self.model != self.model.strip():
            raise ValueError("model must not have surrounding whitespace")
        if self.checksum_algorithm != "sha256":
            raise ValueError("checksum_algorithm must be sha256")
        return self


class FaultRecoverySnapshot(BrokerModel):
    """The record committed to EdgeRecord when a node dies mid-inference.

    Committing a snapshot is the only sanctioned way to make a job resumable.
    The broker writes one every ``checkpoint_every_tokens`` / ``checkpoint_every_seconds``
    (whichever fires first) while a generation is in flight.
    """

    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str = Field(description="Job this snapshot belongs to.")
    node_id: str = Field(description="Node that owned the job when the snapshot was taken.")
    attempt: int = Field(default=0, ge=0, description="Attempt counter of the owning job.")
    model: str = Field(description="Model identifier of the interrupted generation.")
    step: int = Field(ge=0, description="Decode step reached at checkpoint time.")
    partial_tokens: list[PartialToken] = Field(
        default_factory=list, description="Tokens produced so far, in emission order."
    )
    resume_point: ResumePoint
    created_at: datetime = Field(default_factory=utcnow)
    ledger_document_id: str | None = Field(
        default=None, description="Filled in once the snapshot is committed to EdgeRecord."
    )
    content_hash: str | None = Field(
        default=None, description="sha256 over the canonical payload; set on the ledger record."
    )
    schema_version: str = Field(default=SCHEMA_VERSION)

    @model_validator(mode="before")
    @classmethod
    def _reject_boolean_counters(cls, data: Any) -> Any:
        """Reject malformed snapshot counters before recovery can consume them."""
        if isinstance(data, dict):
            for field_name in ("attempt", "step"):
                if isinstance(data.get(field_name), bool):
                    raise ValueError(f"{field_name} must be an integer, not boolean")
                if field_name in data and not isinstance(data.get(field_name), int):
                    raise ValueError(f"{field_name} must be an integer")
        return data

    @model_validator(mode="after")
    def _require_snapshot_identity(self) -> FaultRecoverySnapshot:
        """Reject snapshots that cannot be attributed to a job and model."""
        for field_name in ("snapshot_id", "job_id", "node_id", "model"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if value != value.strip():
                raise ValueError(f"{field_name} must not have surrounding whitespace")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.resume_point.model and self.resume_point.model != self.model:
            raise ValueError("resume_point model must match snapshot model")
        if self.step != self.resume_point.decode_step:
            raise ValueError("step must match resume_point decode_step")
        if self.resume_point.decode_step != self.resume_point.generated_token_count:
            raise ValueError("resume_point counters must match")
        if any(token.index != index for index, token in enumerate(self.partial_tokens)):
            raise ValueError("partial_tokens indices must be sequential")
        if len(self.partial_tokens) > self.step:
            raise ValueError("partial_tokens cannot exceed the recorded decode step")
        if self.ledger_document_id is not None:
            if not self.ledger_document_id.strip():
                raise ValueError("ledger_document_id must not be empty when supplied")
            if self.ledger_document_id != self.ledger_document_id.strip():
                raise ValueError("ledger_document_id must not have surrounding whitespace")
        if self.content_hash is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
                raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
            if self.content_hash != self.compute_content_hash():
                raise ValueError("content_hash does not match the snapshot payload")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Deterministic dict used for hashing and for the ledger record body."""
        return self.model_dump(
            mode="json", exclude={"snapshot_id", "ledger_document_id", "content_hash"}
        )

    def compute_content_hash(self) -> str:
        """sha256 over the canonical payload; used for tamper detection on read."""
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_ledger_body(self) -> dict[str, Any]:
        """Body shape written to EdgeRecord."""
        type(self).model_validate(self.model_dump())
        body = self.canonical_payload()
        body["snapshot_id"] = self.snapshot_id
        body["content_hash"] = self.compute_content_hash()
        body["ledger_document_id"] = self.ledger_document_id
        return body
