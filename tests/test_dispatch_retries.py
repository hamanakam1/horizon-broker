"""Deferred jobs must keep their place and their recovery exclusions."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import JobState, PriorityTier, utcnow
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for


@pytest.mark.parametrize("tier", [PriorityTier.INTERACTIVE, PriorityTier.FLEX])
async def test_recovery_fallback_completes_on_an_alternate_node(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, tier: PriorityTier
) -> None:
    node_a = make_node("node-a", execution_mode=ExecutionMode.AGENT_CONTAINER)
    node_b = make_node(
        "node-b", execution_mode=ExecutionMode.AGENT_CONTAINER, max_batch=1, current_batch=1
    )
    engine = BrokerEngine(replace(config, static_nodes=(node_a, node_b)))
    assigned = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if frame["job"]["attempt"] == 0:
            assigned.set()
        else:
            await engine.supervisor.handle_event(
                NodeEvent(
                    node_id=node_id,
                    type=NodeEventType.COMPLETE,
                    data={"job_id": frame["job"]["job_id"], "text": f"result from {node_id}"},
                )
            )

    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        job = make_job(tier=tier, max_latency_ms=60_000)
        await engine.submit(job)
        await asyncio.wait_for(assigned.wait(), timeout=1.0)
        await engine.supervisor.on_node_lost(node_a.node_id, "synthetic loss")
        assert engine.queue.contains(job.job_id)

        # Capacity returns after the direct recovery attempt has fallen back
        # to the ordinary scheduler. The original node is also still eligible.
        engine.registry.upsert(node_b.model_copy(update={"current_batch_size": 0}))
        job.not_before = utcnow()
        engine.queue.wakeup.set()

        assert await asyncio.wait_for(engine.wait_for_result(job.job_id), timeout=2.0) == (
            "result from node-b"
        )
        assert job.state is JobState.COMPLETED
        assert job.assigned_node_id == node_b.node_id
        assert job.attempt == 1
        assert not engine.queue.contains(job.job_id)
    finally:
        await engine.stop(drain=False)


async def test_unroutable_member_of_a_flex_batch_is_retried(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node("node-a", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": frame["job"]["job_id"], "text": "completed"},
            )
        )

    monkeypatch.setattr(engine.manager, "assign", assign)
    easy = make_job(tier=PriorityTier.FLEX, max_latency_ms=60_000)
    # The same model can have a much larger KV footprint for a longer prompt.
    large = make_job(tier=PriorityTier.FLEX, prompt_tokens=1_000_000, max_latency_ms=60_000)
    await engine.submit(easy)
    await engine.submit(large)
    await engine.start()
    try:
        assert await wait_for(lambda: easy.state is JobState.COMPLETED)
        assert engine.queue.contains(large.job_id)
        assert large.not_before is not None
        assert engine.metrics.counter("horizon_jobs_deferred_total") == 1

        engine.registry.upsert(
            node.model_copy(
                update={"available_vram_bytes": 1_000_000_000_000, "total_vram_bytes": 1_000_000_000_000}
            )
        )
        large.not_before = utcnow()
        engine.queue.wakeup.set()

        assert await asyncio.wait_for(engine.wait_for_result(large.job_id), timeout=2.0) == "completed"
        assert large.state is JobState.COMPLETED
        assert not engine.queue.contains(large.job_id)
        assert engine.metrics.counter("horizon_jobs_completed_total") == 2
    finally:
        await engine.stop(drain=False)
