"""EdgeRecord ledger envelopes.

The broker never talks to the EdgeRecord internals directly; it commits
versioned records through this envelope so the ledger implementation can be
swapped without touching the scheduler or the recovery coordinator.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field, model_validator

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

    @model_validator(mode="after")
    def _require_canonical_envelope(self) -> LedgerRecord:
        """Keep persisted identities and routing metadata stable across replay."""
        for field_name in ("collection", "document_id"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if value != value.strip():
                raise ValueError(f"{field_name} must not have surrounding whitespace")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.content_hash:
            if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
                raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
            if self.content_hash != hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest():
                raise ValueError("content_hash does not match the canonical payload")
        for tag in self.tags:
            if not tag.strip() or tag != tag.strip():
                raise ValueError("tags must contain canonical non-empty strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must not contain duplicates")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def verify(self) -> bool:
        """Recompute the hash and compare it with the stored one."""
        try:
            canonical = self.canonical_json()
        except (TypeError, ValueError):
            return False
        computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
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
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
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

    @model_validator(mode="after")
    def _require_canonical_receipt(self) -> LedgerCommitReceipt:
        """Prevent a receipt from carrying an identity that cannot be replayed."""
        for field_name in ("collection", "document_id"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if value != value.strip():
                raise ValueError(f"{field_name} must not have surrounding whitespace")
        if self.committed_at.tzinfo is None or self.committed_at.utcoffset() is None:
            raise ValueError("committed_at must be timezone-aware")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        if self.commit_url is not None:
            if self.commit_url != self.commit_url.strip():
                raise ValueError("commit_url must not have surrounding whitespace")
            try:
                parsed = urlsplit(self.commit_url)
                valid_port = parsed.port is None or 1 <= parsed.port <= 65_535
            except ValueError:
                parsed = None
                valid_port = False
            if (
                parsed is None
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or not valid_port
            ):
                raise ValueError("commit_url must be an absolute HTTP(S) URL")
        return self
