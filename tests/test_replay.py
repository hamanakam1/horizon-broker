"""Crash replay: a submitted job survives the process that accepted it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from horizon_broker.contracts.jobs import ExecutionPhase, JobState, PriorityTier
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for
from tests.test_engine import start_fake_ollama, state_of


async def test_a_submitted_job_survives_a_cold_broker(config: Any, tmp_path) -> None:
    """The queue is rebuilt from the journal on the next start.

    The first broker accepted work and died before dispatching it; the
    second broker owes the same job to the same caller.
    """
    server, url = await start_fake_ollama(tokens=("Hello", " world"))
    node = make_node("node-a", ollama_url=url)
    journal = str(tmp_path / "journal.jsonl")
    doomed = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    first = make_job(tier=PriorityTier.INTERACTIVE)
    second = make_job(tier=PriorityTier.INTERACTIVE)
    await doomed.submit(first)
    await doomed.submit(second)

    revived = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    await revived.start()
    try:
        assert revived.metrics.counter("horizon_jobs_restored_total") == 2
        assert await wait_for(lambda: state_of(revived, first.job_id) == "completed")
        assert await wait_for(lambda: state_of(revived, second.job_id) == "completed")
        assert revived.status(first.job_id)["result_text"] == "Hello world"
    finally:
        await revived.stop(drain=False)
        await server.close()


async def test_replay_clears_a_stale_split_phase(config: Any, tmp_path) -> None:
    """A crash during split execution must not restore a phantom phase."""
    journal = str(tmp_path / "journal.jsonl")
    first = BrokerEngine(replace(config, journal_path=journal))
    job = make_job(tier=PriorityTier.INTERACTIVE)
    await first.submit(job)
    job.state = JobState.DISPATCHED
    job.phase = ExecutionPhase.PREFILL
    job.assigned_node_id = "synthetic-prefill"
    await first._journal_state(job)

    second = BrokerEngine(replace(config, journal_path=journal))
    assert await second._replay_journal() == 1
    restored = second._jobs[job.job_id]
    assert restored.state is JobState.QUEUED
    assert restored.phase is ExecutionPhase.MONOLITHIC
    assert restored.assigned_node_id is None


async def test_finished_jobs_are_not_replayed_on_a_restart(config: Any, tmp_path) -> None:
    """A completed job stays retired; only unfinished work owes a restart."""
    server, url = await start_fake_ollama(tokens=("done",))
    node = make_node("node-a", ollama_url=url)
    journal = str(tmp_path / "journal.jsonl")
    first = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    await first.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await first.submit(job)
        assert await wait_for(lambda: state_of(first, job.job_id) == "completed")
    finally:
        await first.stop(drain=False)
        await server.close()

    second = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    try:
        await second.start()
        assert second.metrics.counter("horizon_jobs_restored_total") == 0
        assert second.stats()["tracked_jobs"] == 0
    finally:
        await second.stop(drain=False)


async def test_a_cancelled_job_is_not_replayed_on_a_restart(config: Any, tmp_path) -> None:
    """Cancellation is a decision, not a crash: it survives the restart too."""
    server, url = await start_fake_ollama(tokens=("x",))
    node = make_node("node-a", ollama_url=url)
    journal = str(tmp_path / "journal.jsonl")
    first = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    await first.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await first.submit(job)
        await first.cancel(job.job_id)
        assert state_of(first, job.job_id) == "cancelled"
    finally:
        await first.stop(drain=False)
        await server.close()

    second = BrokerEngine(replace(config, static_nodes=(node,), journal_path=journal))
    try:
        await second.start()
        assert second.metrics.counter("horizon_jobs_restored_total") == 0
    finally:
        await second.stop(drain=False)


async def test_replay_persists_and_exposes_an_exhausted_job(config: Any, tmp_path) -> None:
    """A spent recovery budget becomes a durable terminal answer on replay."""
    journal = str(tmp_path / "journal.jsonl")
    first = BrokerEngine(replace(config, journal_path=journal))
    job = make_job(attempt=config.recovery.max_attempts)
    job.state = JobState.DISPATCHED
    first._jobs[job.job_id] = job
    await first._journal_state(job)

    second = BrokerEngine(replace(config, journal_path=journal))
    assert await second._replay_journal() == 0
    assert second.status(job.job_id)["state"] == JobState.FAILED.value
    assert second.status(job.job_id)["error"] == "recovery attempts exhausted before the restart"
    stored = await second.store.load_job_payload(job.job_id)
    assert stored is not None
    assert stored.state is JobState.FAILED
