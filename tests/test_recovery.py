"""State recovery: snapshots, the EdgeRecord port and the recovery coordinator."""

from __future__ import annotations

import pytest

from horizon_broker.config import LedgerConfig, RecoveryConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.ledger import LedgerCommitReceipt, LedgerRecord
from horizon_broker.contracts.recovery import (
    FaultRecoverySnapshot,
    PartialToken,
    ResumePoint,
)
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.recovery.ledger import EdgeRecordClient
from horizon_broker.recovery.recovery import RecoveryCoordinator
from horizon_broker.recovery.snapshot_store import SnapshotStore
from tests.conftest import make_job


class FakeLedger(EdgeRecordClient):
    """In-memory ledger used by the tests."""

    def __init__(self) -> None:
        super().__init__(LedgerConfig(enabled=True, base_url="http://fake"))
        self.records: dict[str, LedgerRecord] = {}

    async def commit(self, record: LedgerRecord) -> LedgerCommitReceipt:
        self.records[record.document_id] = record
        return LedgerCommitReceipt(
            collection=record.collection,
            document_id=record.document_id,
            content_hash=record.content_hash,
        )

    async def latest_for_job(self, job_id: str) -> LedgerRecord | None:
        for record in self.records.values():
            if record.payload.get("job_id") == job_id:
                return record
        return None


class UnavailableLedger(EdgeRecordClient):
    """A ledger that is always down."""

    def __init__(self) -> None:
        super().__init__(LedgerConfig(enabled=True, base_url="http://down"))

    async def commit(self, record: LedgerRecord) -> LedgerCommitReceipt:
        raise LedgerUnavailableError("ledger is down")

    async def latest_for_job(self, job_id: str) -> LedgerRecord | None:
        raise LedgerUnavailableError("ledger is down")


def snapshot_for(job, *, step: int = 12, prompt_hash: str | None = None) -> FaultRecoverySnapshot:
    """Build a snapshot that matches ``job``."""
    return FaultRecoverySnapshot(
        job_id=job.job_id,
        node_id="node-a",
        attempt=job.attempt,
        model=job.payload.model,
        step=step,
        partial_tokens=[PartialToken(index=0, text="Hel"), PartialToken(index=1, text="lo")],
        resume_point=ResumePoint(
            decode_step=step,
            generated_token_count=step,
            kv_cache_ref="edge-kv-1",
            prompt_hash=prompt_hash or job.payload.prompt_hash,
            model=job.payload.model,
        ),
    )


async def test_snapshot_is_committed_to_the_ledger_and_the_journal(tmp_path) -> None:
    ledger = FakeLedger()
    journal = LocalJournal(tmp_path / "journal.jsonl")
    store = SnapshotStore(ledger=ledger, journal=journal)
    job = make_job()
    snapshot = snapshot_for(job)

    receipt = await store.commit(snapshot)
    assert receipt is not None
    assert snapshot.ledger_document_id == receipt.document_id
    assert receipt.document_id in ledger.records

    restored = await store.latest_for_job(job.job_id)
    assert restored is not None
    assert restored.resume_point.kv_cache_ref == "edge-kv-1"
    assert [token.text for token in restored.partial_tokens] == ["Hel", "lo"]

    journalled = await journal.latest("snapshot", job.job_id)
    assert journalled is not None
    assert journalled["job_id"] == job.job_id


async def test_the_journal_covers_for_a_ledger_outage(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    store = SnapshotStore(ledger=UnavailableLedger(), journal=journal)
    job = make_job()
    await store.commit(snapshot_for(job))

    restored = await store.latest_for_job(job.job_id)
    assert restored is not None
    assert restored.job_id == job.job_id


async def test_job_payload_is_rebuildable_after_a_broker_restart(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    await store.store_job_payload(job)
    restored = await store.load_job_payload(job.job_id)
    assert restored is not None
    assert restored.job_id == job.job_id
    assert restored.payload.prompt == job.payload.prompt


async def test_recovery_resumes_from_the_last_snapshot(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    await store.commit(snapshot_for(job))

    requeued: list = []
    dispatched: list[tuple] = []

    async def dispatch(job, excluded):
        dispatched.append((job, excluded))

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3, recovery_budget_seconds=30.0),
        requeue=lambda job: requeued.append(job) or job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a", reason="power loss")

    assert outcome.resumed is True
    assert outcome.dispatched is True
    assert outcome.within_budget is True
    assert outcome.elapsed_seconds < 30.0
    assert "node-a" in outcome.excluded_node_ids

    recovered_job = dispatched[0][0]
    assert recovered_job.attempt == 1
    assert recovered_job.state is JobState.QUEUED
    assert recovered_job.kv_cache_state_id == "edge-kv-1"
    assert recovered_job.resume_point is not None
    assert recovered_job.resume_point.decode_step == 12
    assert dispatched[0][1] == frozenset({"node-a"})

    # A recovered job is dispatched directly, never double-queued.
    assert requeued == []


async def test_recovery_ignores_a_snapshot_for_a_different_prompt(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    await store.commit(snapshot_for(job, prompt_hash="a-different-prompt"))

    dispatched: list = []

    async def dispatch(job, excluded):
        dispatched.append(job)

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda job: job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a")
    assert outcome.resumed is False
    assert dispatched[0].resume_point is None
    assert dispatched[0].kv_cache_state_id is None


async def test_a_job_is_re_queued_when_no_node_can_take_it(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    requeued: list = []

    async def dispatch(job, excluded):
        raise RuntimeError("no capacity")

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda job: requeued.append(job) or job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a")
    assert outcome.dispatched is False
    assert len(requeued) == 1
    assert requeued[0].not_before is not None


async def test_recovery_gives_up_after_the_retry_budget(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job(attempt=2)

    dispatched: list = []

    async def dispatch(job, excluded):
        dispatched.append(job)

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda job: job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a")
    assert outcome.dispatched is False
    assert dispatched == []
    assert job.state is JobState.FAILED
    assert job.error is not None


async def test_recovery_without_any_snapshot_still_re_dispatches(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()

    async def dispatch(job, excluded):
        return None

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda job: job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a")
    assert outcome.resumed is False
    assert outcome.dispatched is True
    assert outcome.snapshot_id is None


@pytest.mark.parametrize("attempts", [0, 1, 2])
async def test_recovered_jobs_increment_the_attempt_counter(tmp_path, attempts: int) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job(attempt=attempts)

    async def dispatch(job, excluded):
        return None

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=5),
        requeue=lambda job: job,
        dispatch=dispatch,
    )
    await coordinator.recover(job, "node-a")
    assert job.attempt == attempts + 1
