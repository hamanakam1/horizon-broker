"""State recovery: snapshots, the EdgeRecord port and the recovery coordinator."""

from __future__ import annotations

import asyncio

import pytest

from horizon_broker.config import LedgerConfig, RecoveryConfig
from horizon_broker.contracts.errors import (
    LedgerUnavailableError,
    MaxAttemptsExceededError,
)
from horizon_broker.contracts.jobs import ExecutionPhase, JobState
from horizon_broker.contracts.ledger import LedgerCommitReceipt, LedgerRecord
from horizon_broker.contracts.recovery import (
    FaultRecoverySnapshot,
    PartialToken,
    ResumePoint,
)
from horizon_broker.observability.metrics import MetricsRegistry
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


async def test_dispatch_error_cannot_requeue_a_synchronously_completed_job(tmp_path) -> None:
    """A terminal callback result survives a contradictory dispatch error."""
    job = make_job()
    requeued: list[str] = []

    async def dispatch(recovered, _excluded: frozenset[str]) -> None:
        recovered.state = JobState.COMPLETED
        recovered.result_text = "completed-before-transport-error"
        raise RuntimeError("synthetic transport acknowledgement contradiction")

    coordinator = RecoveryCoordinator(
        store=SnapshotStore(
            ledger=UnavailableLedger(),
            journal=LocalJournal(tmp_path / "journal.jsonl"),
        ),
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: requeued.append(recovered.job_id) or recovered,
        dispatch=dispatch,
    )

    outcome = await coordinator.recover(job, "node-a")

    assert outcome.dispatched is True
    assert job.state is JobState.COMPLETED
    assert job.result_text == "completed-before-transport-error"
    assert requeued == []


async def test_terminal_jobs_release_recovery_placement_history(tmp_path) -> None:
    """Avoided-node state does not grow after a recovered job finishes."""
    coordinator = RecoveryCoordinator(
        store=SnapshotStore(
            ledger=UnavailableLedger(),
            journal=LocalJournal(tmp_path / "journal.jsonl"),
        ),
        config=RecoveryConfig(),
        requeue=lambda recovered: recovered,
        dispatch=lambda _job, _excluded: None,
    )

    coordinator.avoid_node("job-1", "node-a")
    assert coordinator.avoided_nodes("job-1") == frozenset({"node-a"})
    coordinator.forget("job-1")
    assert coordinator.avoided_nodes("job-1") == frozenset()


def test_a_snapshot_with_a_tampered_ledger_envelope_hash_is_rejected() -> None:
    job = make_job()
    snapshot = snapshot_for(job)
    body = snapshot.to_ledger_body()
    record = LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id=snapshot.snapshot_id,
        payload=body,
    ).model_copy(update={"content_hash": "tampered"})

    assert SnapshotStore._snapshot_from_record(record) is None


def test_a_snapshot_with_a_mismatched_document_identity_is_rejected() -> None:
    job = make_job()
    snapshot = snapshot_for(job)
    record = LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id=snapshot.snapshot_id,
        payload=snapshot.to_ledger_body(),
    ).model_copy(update={"document_id": "foreign-document"})

    assert SnapshotStore._snapshot_from_record(record) is None


def test_a_snapshot_without_an_embedded_content_hash_is_rejected() -> None:
    job = make_job()
    snapshot = snapshot_for(job)
    body = snapshot.to_ledger_body()
    body.pop("content_hash")
    record = LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id=snapshot.snapshot_id,
        payload=body,
    )

    assert SnapshotStore._snapshot_from_record(record) is None


async def test_the_journal_covers_for_a_ledger_outage(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    metrics = MetricsRegistry()
    store = SnapshotStore(ledger=UnavailableLedger(), journal=journal, metrics=metrics)
    job = make_job()
    await store.commit(snapshot_for(job))

    restored = await store.latest_for_job(job.job_id)
    assert restored is not None
    assert restored.job_id == job.job_id
    assert metrics.counter("horizon_ledger_fallback_total", reason="commit") == 1
    assert metrics.counter("horizon_ledger_fallback_total", reason="query") == 1


async def test_a_journal_read_error_degrades_to_a_fresh_recovery(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")

    async def fail_latest(kind: str, key: str, field: str = "job_id") -> None:
        raise OSError("synthetic journal read failure")

    journal.latest = fail_latest  # type: ignore[method-assign]
    job = make_job()
    dispatched: list[str] = []

    async def dispatch(recovered, _excluded: frozenset[str]) -> None:
        dispatched.append(recovered.job_id)

    coordinator = RecoveryCoordinator(
        store=SnapshotStore(ledger=UnavailableLedger(), journal=journal),
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: recovered,
        dispatch=dispatch,
    )

    outcome = await coordinator.recover(job, "node-a")

    assert outcome.dispatched is True
    assert outcome.resumed is False
    assert dispatched == [job.job_id]


async def test_a_journal_snapshot_with_a_tampered_content_hash_is_rejected(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    store = SnapshotStore(ledger=None, journal=journal)
    job = make_job()
    snapshot = snapshot_for(job)
    body = snapshot.to_ledger_body()
    body["step"] = snapshot.step + 1
    await journal.append(
        "snapshot",
        {
            "job_id": job.job_id,
            "node_id": snapshot.node_id,
            "attempt": snapshot.attempt,
            "created_at": snapshot.created_at.isoformat(),
            "snapshot": body,
        },
    )

    assert await store.latest_for_job(job.job_id) is None


async def test_invalid_ledger_snapshot_falls_back_and_is_counted(tmp_path) -> None:
    ledger = FakeLedger()
    metrics = MetricsRegistry()
    store = SnapshotStore(
        ledger=ledger,
        journal=LocalJournal(tmp_path / "journal.jsonl"),
        metrics=metrics,
    )
    job = make_job()
    snapshot = snapshot_for(job)
    await store.commit(snapshot)
    ledger.records[snapshot.snapshot_id] = ledger.records[snapshot.snapshot_id].model_copy(
        update={"content_hash": "tampered"}
    )

    restored = await store.latest_for_job(job.job_id)

    assert restored is not None
    assert restored.job_id == job.job_id
    assert metrics.counter("horizon_ledger_fallback_total", reason="invalid") == 1


async def test_ledger_snapshot_identity_must_match_the_requested_job(tmp_path) -> None:
    requested = make_job(prompt="requested")
    foreign = make_job(prompt="foreign")
    record = LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id="foreign-snapshot",
        payload=snapshot_for(foreign).to_ledger_body(),
    )
    ledger = FakeLedger()

    async def return_foreign(_job_id: str) -> LedgerRecord:
        return record

    ledger.latest_for_job = return_foreign  # type: ignore[method-assign]
    store = SnapshotStore(ledger=ledger, journal=LocalJournal(tmp_path / "journal.jsonl"))

    assert await store.latest_for_job(requested.job_id) is None


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
    assert recovered_job.phase is ExecutionPhase.MONOLITHIC
    assert recovered_job.kv_cache_state_id == "edge-kv-1"
    assert recovered_job.resume_point is not None
    assert recovered_job.resume_point.decode_step == 12
    assert dispatched[0][1] == frozenset({"node-a"})

    # A recovered job is dispatched directly, never double-queued.
    assert requeued == []


async def test_recovery_ignores_a_snapshot_for_a_different_prompt(tmp_path) -> None:
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    await store.commit(snapshot_for(job, prompt_hash="f" * 64))

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


async def test_journal_snapshot_identity_must_match_the_requested_job(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    requested = make_job(prompt="requested")
    foreign = make_job(prompt="foreign")
    snapshot = snapshot_for(foreign)
    await journal.append(
        "snapshot",
        {
            "job_id": requested.job_id,
            "node_id": snapshot.node_id,
            "attempt": snapshot.attempt,
            "created_at": snapshot.created_at.isoformat(),
            "snapshot": snapshot.to_ledger_body(),
        },
    )
    store = SnapshotStore(ledger=None, journal=journal)

    assert await store.latest_for_job(requested.job_id) is None


async def test_journal_job_payload_identity_must_match_its_record_key(tmp_path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    requested = make_job(prompt="requested")
    foreign = make_job(prompt="foreign")
    await journal.append(
        "job_payload",
        {"job_id": requested.job_id, "job": foreign.model_dump(mode="json")},
    )
    store = SnapshotStore(ledger=None, journal=journal)

    assert await store.load_job_payload(requested.job_id) is None
    assert await store.load_jobs() == []


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


async def test_replaying_a_successful_recovery_does_not_dispatch_twice(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    dispatches: list[int] = []

    async def dispatch(recovered, _excluded: frozenset[str]) -> None:
        dispatches.append(recovered.attempt)

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: recovered,
        dispatch=dispatch,
    )

    first = await coordinator.recover(job, "node-a")
    second = await coordinator.recover(job, "node-a")

    assert first.dispatched is True
    assert second == first
    assert dispatches == [1]
    assert job.attempt == 1


async def test_concurrent_recovery_notifications_share_one_attempt(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    dispatches: list[int] = []

    async def dispatch(recovered, _excluded: frozenset[str]) -> None:
        await asyncio.sleep(0.01)
        dispatches.append(recovered.attempt)

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: recovered,
        dispatch=dispatch,
    )

    first, second = await asyncio.gather(
        coordinator.recover(job, "node-a"),
        coordinator.recover(job, "node-a"),
    )

    assert first == second
    assert dispatches == [1]
    assert job.attempt == 1


async def test_a_later_node_loss_starts_a_new_recovery_attempt(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    dispatches: list[int] = []

    async def dispatch(recovered, _excluded: frozenset[str]) -> None:
        dispatches.append(recovered.attempt)
        recovered.state = JobState.DISPATCHED

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: recovered,
        dispatch=dispatch,
    )

    await coordinator.recover(job, "node-a")
    job.state = JobState.RECOVERING
    await coordinator.recover(job, "node-b")

    assert dispatches == [1, 2]
    assert job.attempt == 2


async def test_cancelled_recovery_requeues_the_incremented_attempt(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()
    requeued: list[str] = []

    async def dispatch(_recovered, _excluded: frozenset[str]) -> None:
        raise asyncio.CancelledError

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda recovered: requeued.append(recovered.job_id) or recovered,
        dispatch=dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await coordinator.recover(job, "node-a")

    assert job.state is JobState.QUEUED
    assert job.attempt == 1
    assert job.not_before is not None
    assert requeued == [job.job_id]


async def test_cancelled_recovery_preserves_cancellation_if_requeue_fails(tmp_path) -> None:
    store = SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job()

    async def dispatch(_recovered, _excluded: frozenset[str]) -> None:
        raise asyncio.CancelledError

    def broken_requeue(_recovered):
        raise RuntimeError("synthetic requeue failure")

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=broken_requeue,
        dispatch=dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await coordinator.recover(job, "node-a")

    assert job.state is JobState.QUEUED
    assert job.not_before is not None


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


async def test_a_spent_attempt_budget_is_a_typed_failure(tmp_path) -> None:
    """The failures the taxonomy describes must be raised, not written as prose."""
    store = SnapshotStore(ledger=FakeLedger(), journal=LocalJournal(tmp_path / "journal.jsonl"))
    job = make_job(attempt=2)

    async def dispatch(job, excluded):
        return None

    coordinator = RecoveryCoordinator(
        store=store,
        config=RecoveryConfig(max_attempts=3),
        requeue=lambda job: job,
        dispatch=dispatch,
    )
    outcome = await coordinator.recover(job, "node-a")

    assert outcome.dispatched is False
    assert isinstance(outcome.failure, MaxAttemptsExceededError)
    assert job.state is JobState.FAILED
    assert job.phase is ExecutionPhase.MONOLITHIC
    assert job.assigned_node_id is None
    assert "max attempts exceeded" in (job.error or "")
    assert outcome.as_dict()["failure"] == "MaxAttemptsExceededError"
