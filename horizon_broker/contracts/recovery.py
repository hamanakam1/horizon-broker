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
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

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

    def canonical_payload(self) -> dict[str, Any]:
        """Deterministic dict used for hashing and for the ledger record body."""
        return self.model_dump(
            mode="json", exclude={"snapshot_id", "ledger_document_id", "content_hash"}
        )

    def compute_content_hash(self) -> str:
        """sha256 over the canonical payload; used for tamper detection on read."""
        canonical = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_ledger_body(self) -> dict[str, Any]:
        """Body shape written to EdgeRecord."""
        body = self.canonical_payload()
        body["snapshot_id"] = self.snapshot_id
        body["content_hash"] = self.compute_content_hash()
        body["ledger_document_id"] = self.ledger_document_id
        return body
