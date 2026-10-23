"""A spent recovery budget must reach callers and remain spent after restart."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for


async def test_exhausted_recovery_finishes_callers_and_survives_restart(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node("node-a", execution_mode=ExecutionMode.AGENT_CONTAINER)
    broker_config = replace(config, static_nodes=(node,))
    engine = BrokerEngine(broker_config)
    assigned = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assigned.set()

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job(attempt=config.recovery.max_attempts - 1, max_latency_ms=60_000)
    waiter: asyncio.Task[str | None] | None = None
    await engine.start()
    try:
        await engine.submit(job)
        await asyncio.wait_for(assigned.wait(), timeout=1.0)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        stream = engine.subscribe(job.job_id)
        # Register the caller before the loss, rather than taking the terminal fast path.
        await asyncio.sleep(0)

        await engine.supervisor.on_node_lost(node.node_id, "synthetic node loss")

        assert job.state is JobState.FAILED
        assert "max attempts exceeded" in (job.error or "")
        assert await wait_for(waiter.done, timeout=0.2)
        assert await waiter is None
        assert await asyncio.wait_for(stream.get(), timeout=0.2) is None
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on(node.node_id) == ()

        # Repeated loss reports must not publish a second terminal outcome.
        await engine.supervisor.on_node_lost(node.node_id, "repeated synthetic node loss")
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
        assert stream.empty()
    finally:
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await engine.stop(drain=False)

    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None
    assert stored.state is JobState.FAILED
    assert stored.error == job.error

    restarted = BrokerEngine(broker_config)
    try:
        await restarted.start()
        assert restarted.metrics.counter("horizon_jobs_restored_total") == 0
        assert restarted.stats()["tracked_jobs"] == 0
    finally:
        await restarted.stop(drain=False)
