"""Snapshot store.

The store is the single place where recovery state is written and read.

Write path (in order):

1. append the snapshot to the local journal (durable, always available),
2. commit the snapshot to EdgeRecord (durable, shared, may fail).

Read path:

1. query EdgeRecord for the newest snapshot,
2. fall back to the local journal when the ledger is unreachable.

The fallback is what makes recovery survive a ledger outage; the ledger is
what makes it survive a *broker* outage.
"""

from __future__ import annotations

import logging

from horizon_broker.config import LedgerConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.jobs import InferenceJob
from horizon_broker.contracts.ledger import LedgerCommitReceipt, LedgerRecord
from horizon_broker.contracts.recovery import FaultRecoverySnapshot
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.recovery.ledger import EdgeRecordClient

LOGGER = logging.getLogger(__name__)

SNAPSHOT_KIND = "snapshot"
JOB_KIND = "job_payload"
TRIM_EVERY = 200
"""How many commits pass between two journal trims."""


class SnapshotStore:
    """Reads and writes fault-recovery snapshots."""

    def __init__(
        self,
        *,
        ledger: EdgeRecordClient | None,
        journal: LocalJournal,
        ledger_config: LedgerConfig | None = None,
        journal_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.ledger = ledger
        self.journal = journal
        self.ledger_config = ledger_config or (ledger.config if ledger else None)
        self.collection = (
            self.ledger_config.collection if self.ledger_config else "horizon_broker.snapshots"
        )
        self.journal_max_bytes = journal_max_bytes
        self._commits = 0

    async def commit(self, snapshot: FaultRecoverySnapshot) -> LedgerCommitReceipt | None:
        """Persist a snapshot; returns the receipt when EdgeRecord accepted it."""
        body = snapshot.to_ledger_body()
        await self.journal.append(
            SNAPSHOT_KIND,
            {
                "job_id": snapshot.job_id,
                "node_id": snapshot.node_id,
                "attempt": snapshot.attempt,
                "created_at": snapshot.created_at.isoformat(),
                "snapshot": body,
            },
        )
        await self._maybe_trim()
        if self.ledger is None or not self.ledger.enabled:
            return None
        record = LedgerRecord.build(
            collection=self.collection,
            document_id=snapshot.snapshot_id,
            payload=body,
            tags=("horizon_broker", "recovery", f"job:{snapshot.job_id}"),
        )
        try:
            receipt = await self.ledger.commit(record)
        except LedgerUnavailableError as exc:
            LOGGER.warning(
                "ledger commit failed, snapshot kept in local journal only: %s", exc
            )
            return None
        snapshot.ledger_document_id = receipt.document_id
        return receipt

    async def latest_for_job(self, job_id: str) -> FaultRecoverySnapshot | None:
        """Return the newest snapshot for ``job_id``."""
        if self.ledger is not None and self.ledger.enabled:
            try:
                record = await self.ledger.latest_for_job(job_id)
            except LedgerUnavailableError as exc:
                LOGGER.warning("ledger query failed, falling back to journal: %s", exc)
            else:
                snapshot = self._snapshot_from_record(record) if record else None
                if snapshot is not None:
                    return snapshot

        raw = await self.journal.latest(SNAPSHOT_KIND, job_id)
        if not raw:
            return None
        body = raw.get("snapshot")
        if not isinstance(body, dict):
            return None
        try:
            return FaultRecoverySnapshot.model_validate(body)
        except Exception as exc:  # noqa: BLE001 - a corrupt journal line must not kill recovery
            LOGGER.warning("ignoring corrupt snapshot for job %s: %s", job_id, exc)
            return None

    async def _maybe_trim(self) -> None:
        """Keep the journal from growing without bound."""
        self._commits += 1
        if self._commits % TRIM_EVERY != 0:
            return
        try:
            removed = await self.journal.trim(self.journal_max_bytes)
        except OSError as exc:
            LOGGER.warning("could not trim the journal: %s", exc)
            return
        if removed:
            LOGGER.info("trimmed %d bytes from the local journal", removed)

    async def store_job_payload(self, job: InferenceJob) -> None:
        """Persist the job envelope so a cold broker can rebuild it."""
        await self.journal.append(
            JOB_KIND,
            {"job_id": job.job_id, "job": job.model_dump(mode="json")},
        )
        await self._maybe_trim()

    async def load_job_payload(self, job_id: str) -> InferenceJob | None:
        """Rebuild a job envelope from the journal."""
        raw = await self.journal.latest(JOB_KIND, job_id)
        if not raw:
            return None
        body = raw.get("job")
        if not isinstance(body, dict):
            return None
        try:
            return InferenceJob.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ignoring corrupt journal payload for job %s: %s", job_id, exc)
            return None

    @staticmethod
    def _snapshot_from_record(record: LedgerRecord) -> FaultRecoverySnapshot | None:
        """Validate a ledger record into a snapshot, rejecting tampered payloads."""
        if not record.payload:
            return None
        try:
            snapshot = FaultRecoverySnapshot.model_validate(record.payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ledger record %s is not a valid snapshot: %s", record.document_id, exc)
            return None
        stored_hash = record.payload.get("content_hash")
        if stored_hash and stored_hash != snapshot.compute_content_hash():
            LOGGER.warning(
                "ledger record %s failed content hash verification", record.document_id
            )
            return None
        return snapshot
