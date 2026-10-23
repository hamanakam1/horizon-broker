"""Queue gates cannot extend a caller's latency budget."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import InferenceJob, JobState, PriorityTier
from horizon_broker.scheduler.engine import BrokerEngine
from horizon_broker.scheduler.priority_queue import JobPriorityQueue
from horizon_broker.scheduler.window import WindowPolicy
from tests.conftest import make_job

NOW = datetime(2030, 1, 1, tzinfo=UTC)
POLICY = WindowPolicy(window_seconds=100, min_batch_size=4, max_batch_size=8)


def queued_job(
    tier: PriorityTier, deadline_seconds: int, *, deferred: bool = False
) -> InferenceJob:
    return make_job(
        tier=tier,
        model="synthetic-model",
        submitted_at=NOW - timedelta(seconds=10),
        deadline_at=NOW + timedelta(seconds=deadline_seconds),
        not_before=NOW + timedelta(seconds=60) if deferred else None,
        max_latency_ms=600_000,
    )


@pytest.mark.parametrize("power_ok", [True, False])
@pytest.mark.parametrize("deferred", [True, False])
def test_flex_expiry_precedes_power_and_deferral_gates(power_ok: bool, deferred: bool) -> None:
    queue = JobPriorityQueue()
    expired = queued_job(PriorityTier.FLEX, 0, deferred=deferred)
    valid = queued_job(PriorityTier.FLEX, 120)
    queue.submit(expired)
    queue.submit(valid)

    assert queue.plan_flex(NOW.timestamp(), POLICY, power_ok=power_ok) == []

    assert expired.state is JobState.FAILED
    assert expired.error == "latency budget expired before dispatch"
    assert not queue.contains(expired.job_id)
    assert queue.pending_flex() == [valid]
    assert valid.state is JobState.WINDOWING
    assert queue.stats().flex == 1
    assert queue.stats().open_windows == int(power_ok)
    assert queue.drain_expired() == [expired]
    queue.plan_flex(NOW.timestamp(), POLICY, power_ok=power_ok)
    assert queue.drain_expired() == []


def test_flex_batches_exclude_expired_jobs_and_retain_deferred_jobs() -> None:
    queue = JobPriorityQueue()
    expired = queued_job(PriorityTier.FLEX, -1)
    ready = queued_job(PriorityTier.FLEX, 120)
    deferred = queued_job(PriorityTier.FLEX, 120, deferred=True)
    for job in (expired, ready, deferred):
        queue.submit(job)
    policy = WindowPolicy(min_batch_size=1, max_batch_size=8, min_dwell_seconds=0)

    batches = queue.plan_flex(NOW.timestamp(), policy)

    assert len(batches) == 1
    assert batches[0].jobs == (ready,)
    assert queue.pending_flex() == [deferred]
    assert queue.drain_expired() == [expired]


@pytest.mark.parametrize("expire", [True, False], ids=["expiry", "removal"])
def test_an_empty_flex_window_does_not_apply_to_future_work(expire: bool) -> None:
    queue = JobPriorityQueue()
    first = queued_job(PriorityTier.FLEX, 5)
    queue.submit(first)
    assert queue.plan_flex(NOW.timestamp(), POLICY) == []
    assert queue.stats().open_windows == 1

    if expire:
        queue.plan_flex(NOW.timestamp() + 5, POLICY, power_ok=False)
        assert queue.drain_expired() == [first]
    else:
        assert queue.remove(first.job_id) is first
    assert queue.stats().open_windows == 0
    assert queue.pending_flex() == []

    fresh = queued_job(PriorityTier.FLEX, 400)
    queue.submit(fresh)
    assert queue.plan_flex(NOW.timestamp() + 150, POLICY) == []
    assert queue.stats().open_windows == 1
    batches = queue.plan_flex(NOW.timestamp() + 250, POLICY)
    assert len(batches) == 1
    assert batches[0].jobs == (fresh,)
    assert batches[0].opened_at == NOW.timestamp() + 150
    assert queue.stats().open_windows == 0


def test_interactive_expiry_precedes_the_deferral_gate() -> None:
    queue = JobPriorityQueue()
    expired = queued_job(PriorityTier.INTERACTIVE, 0, deferred=True)
    ready = queued_job(PriorityTier.INTERACTIVE, 120)
    deferred = queued_job(PriorityTier.INTERACTIVE, 180, deferred=True)
    for job in (expired, ready, deferred):
        queue.submit(job)

    assert queue.pop_interactive(NOW.timestamp()) is ready
    assert queue.drain_expired() == [expired]
    assert expired.state is JobState.FAILED
    assert not queue.contains(expired.job_id)
    assert queue.contains(deferred.job_id)
    assert queue.stats().interactive == 1
    assert queue.pop_interactive(NOW.timestamp()) is None
    assert queue.drain_expired() == []


@pytest.mark.parametrize("tier", [PriorityTier.INTERACTIVE, PriorityTier.FLEX])
async def test_engine_expiry_finishes_callers_and_journals_failure_without_nodes(
    config: BrokerConfig, tier: PriorityTier
) -> None:
    engine = BrokerEngine(config)
    job = make_job(
        tier=tier,
        model="synthetic-model",
        max_latency_ms=600_000,
        not_before=datetime(2099, 1, 1, tzinfo=UTC),
    )
    waiter: asyncio.Task[str | None] | None = None
    await engine.start()
    try:
        await engine.submit(job)
        waiting = asyncio.Event()

        async def wait_for_result() -> str | None:
            waiting.set()
            return await engine.wait_for_result(job.job_id)

        waiter = asyncio.create_task(wait_for_result())
        stream = engine.subscribe(job.job_id)
        await waiting.wait()
        # Move the deadline explicitly; expiration never depends on a timed sleep.
        job.deadline_at = datetime(2000, 1, 1, tzinfo=UTC)
        engine.queue.wakeup.set()

        assert await asyncio.wait_for(waiter, timeout=1.0) is None
        assert await asyncio.wait_for(stream.get(), timeout=1.0) is None
        assert engine.require_status(job.job_id)["state"] == "failed"
        assert job.error == "latency budget expired before dispatch"
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
        assert engine.queue.stats().interactive == 0
        assert engine.queue.stats().flex == 0
        assert engine.supervisor.active_jobs() == []
    finally:
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await engine.stop(drain=False)

    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None
    assert stored.state is JobState.FAILED
    assert stored.error == job.error
