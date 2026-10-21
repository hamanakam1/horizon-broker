"""EdgeRecord ledger envelopes.

The broker never talks to the EdgeRecord internals directly; it commits
versioned records through this envelope so the ledger implementation can be
swapped without touching the scheduler or the recovery coordinator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from horizon_broker.contracts.jobs import BrokerModel

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


class LedgerRecord(BrokerModel):
    """A record persisted in EdgeRecord."""

    collection: str = Field(description="Target collection, e.g. horizon_broker.snapshots.")
    document_id: str = Field(description="Stable, deterministic document id.")
    schema_version: str = Field(default=SCHEMA_VERSION)
    payload: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = Field(default="", description="sha256 over the canonical payload.")
    created_at: datetime = Field(default_factory=utcnow)
    tags: tuple[str, ...] = ()

    def canonical_json(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"))

    def verify(self) -> bool:
        """Recompute the hash and compare it with the stored one."""
        computed = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return computed == self.content_hash

    @classmethod
    def build(
        cls,
        collection: str,
        document_id: str,
        payload: dict[str, Any],
        tags: tuple[str, ...] = (),
    ) -> LedgerRecord:
        """Create a record with a computed content hash."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(
            collection=collection,
            document_id=document_id,
            payload=payload,
            content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            tags=tags,
        )


class LedgerCommitReceipt(BrokerModel):
    """Proof that a record was durably committed."""

    collection: str
    document_id: str
    content_hash: str
    commit_url: str | None = None
    committed_at: datetime = Field(default_factory=utcnow)
