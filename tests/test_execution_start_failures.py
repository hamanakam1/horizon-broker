"""Failed execution startup must release callers and node reservations."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig, RoutingConfig
from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.nodes import ExecutionMode, NodeCapability, NodeRole
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.ollama.client import OllamaClient
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for


@pytest.mark.parametrize("failure", ["assignment", "missing-callback"])
async def test_failed_assignment_releases_capacity_and_cannot_be_resurrected(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    node = make_node("node-a", execution_mode=ExecutionMode.AGENT_CONTAINER, max_batch=1)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def fail_assignment(node_id: str, frame: dict[str, Any]) -> None:
        raise OSError("synthetic assignment failure")

    monkeypatch.setattr(engine.manager, "assign", fail_assignment)
    if failure == "missing-callback":
        engine.supervisor._assign = None
    job = make_job(max_latency_ms=60_000)
    try:
        await engine.submit(job)
        engine.queue.remove(job.job_id)
        stream = engine.subscribe(job.job_id)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        await asyncio.sleep(0)
        await engine._execute(job)

        assert job.state is JobState.FAILED
        assert await asyncio.wait_for(waiter, 0.2) is None
        assert await asyncio.wait_for(stream.get(), 0.2) is None
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1

        await engine.supervisor.handle_event(NodeEvent(
            node_id=node.node_id, type=NodeEventType.COMPLETE,
            data={"job_id": job.job_id, "text": "late answer"},
        ))
        assert job.state is JobState.FAILED
        assert job.result_text is None

        async def accept_assignment(node_id: str, frame: dict[str, Any]) -> None:
            return None

        monkeypatch.setattr(engine.manager, "assign", accept_assignment)
        engine.supervisor._assign = engine._assign
        next_job = make_job(max_latency_ms=60_000)
        await engine.submit(next_job)
        engine.queue.remove(next_job.job_id)
        await engine._execute(next_job)
        await engine.supervisor.handle_event(NodeEvent(
            node_id=node.node_id, type=NodeEventType.COMPLETE,
            data={"job_id": next_job.job_id, "text": "next answer"},
        ))
        assert await engine.wait_for_result(next_job.job_id) == "next answer"
    finally:
        await engine.stop(drain=False)
    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None and stored.state is JobState.FAILED


@pytest.mark.parametrize("outcome", ["recovered", "split-handoff", "completed", "cancelled"])
async def test_late_assignment_failure_cannot_invalidate_a_new_execution_or_result(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    split = outcome == "split-handoff"
    nodes = (
        make_node("node-a", execution_mode=ExecutionMode.AGENT_CONTAINER, max_batch=1,
                  role=NodeRole.PREFILL if split else NodeRole.GENERAL, prefill=split),
        make_node("node-b", execution_mode=ExecutionMode.AGENT_CONTAINER, max_batch=1,
                  role=NodeRole.DECODE if split else NodeRole.GENERAL, decode=split),
    )
    engine = BrokerEngine(replace(
        config, static_nodes=nodes,
        routing=RoutingConfig(kv_handoff_models=("qwen2.5:7b",) if split else ()),
    ))
    assigning = asyncio.Event()
    fail_old_send = asyncio.Event()

    async def hold_assignment(node_id: str, frame: dict[str, Any]) -> None:
        if node_id == "node-a" and frame["type"] == "assign":
            assigning.set()
            await fail_old_send.wait()
            raise OSError("synthetic late assignment failure")

    monkeypatch.setattr(engine.manager, "assign", hold_assignment)
    job = make_job(
        max_latency_ms=60_000, prompt_tokens=32_768 if split else None,
        output_tokens=4_096 if split else 64,
    )
    try:
        await engine.submit(job)
        engine.queue.remove(job.job_id)
        stream = engine.subscribe(job.job_id)
        dispatch = asyncio.create_task(engine._execute(job))
        await asyncio.wait_for(assigning.wait(), 0.2)
        if outcome == "recovered":
            await engine.supervisor.on_node_lost("node-a", "synthetic loss")
        elif split:
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-a", type=NodeEventType.COMPLETE,
                data={"job_id": job.job_id, "kv_cache_ref": "synthetic-kv"},
            ))
        elif outcome == "cancelled":
            assert await engine.cancel(job.job_id)
        else:
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-a", type=NodeEventType.COMPLETE,
                data={"job_id": job.job_id, "text": "current answer"},
            ))
        fail_old_send.set()
        await asyncio.wait_for(dispatch, 0.2)
        if outcome in {"recovered", "split-handoff"}:
            assert job.state is JobState.DISPATCHED
            assert engine.registry.jobs_on("node-b") == (job.job_id,)
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-b", type=NodeEventType.COMPLETE,
                data={"job_id": job.job_id, "text": "current answer"},
            ))
        expected_state = JobState.CANCELLED if outcome == "cancelled" else JobState.COMPLETED
        assert job.state is expected_state
        assert await engine.wait_for_result(job.job_id) == (
            None if outcome == "cancelled" else "current answer"
        )
        if outcome != "cancelled":
            assert await asyncio.wait_for(stream.get(), 0.2) == "current answer"
        assert await asyncio.wait_for(stream.get(), 0.2) is None
        assert stream.empty()
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on("node-a") == ()
        assert engine.registry.jobs_on("node-b") == ()
        assert engine.metrics.counter("horizon_jobs_failed_total") == 0
        assert engine.metrics.counter("horizon_jobs_completed_total") == (outcome != "cancelled")
    finally:
        fail_old_send.set()
        await engine.stop(drain=False)
    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None and stored.state is expected_state


@pytest.mark.parametrize("kv_ref", [None, "synthetic-kv"])
@pytest.mark.parametrize("failure", ["assignment", "missing-node"])
async def test_failed_decode_handoff_finalizes_the_job(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, kv_ref: str | None, failure: str,
) -> None:
    prefill = make_node(
        "node-prefill", role=NodeRole.PREFILL, prefill=True,
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    decode = make_node(
        "node-decode", role=NodeRole.DECODE, decode=True,
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    engine = BrokerEngine(replace(
        config, static_nodes=(prefill, decode),
        routing=RoutingConfig(kv_handoff_models=("qwen2.5:7b",)),
    ))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if node_id == decode.node_id:
            raise OSError("synthetic decode assignment failure")

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job(prompt_tokens=32_768, output_tokens=4_096, max_latency_ms=60_000)
    try:
        await engine.submit(job)
        engine.queue.remove(job.job_id)
        await engine._execute(job)
        if failure == "missing-node":
            engine.registry.remove(decode.node_id)
        stream = engine.subscribe(job.job_id)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        await asyncio.sleep(0)
        await engine.supervisor.handle_event(NodeEvent(
            node_id=prefill.node_id, type=NodeEventType.COMPLETE,
            data={"job_id": job.job_id, "kv_cache_ref": kv_ref},
        ))
        assert job.state is JobState.FAILED
        assert await asyncio.wait_for(waiter, 0.2) is None
        assert await asyncio.wait_for(stream.get(), 0.2) is None
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on(prefill.node_id) == ()
        assert engine.registry.jobs_on(decode.node_id) == ()
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
    finally:
        await engine.stop(drain=False)
    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None and stored.state is JobState.FAILED


async def test_client_factory_failure_reaches_waiter_and_journal(config: BrokerConfig) -> None:
    node = make_node("node-a")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    def fail_factory(node: NodeCapability) -> OllamaClient:
        raise ValueError("synthetic client setup failure")

    engine.supervisor._ollama_factory = fail_factory
    job = make_job(max_latency_ms=60_000)
    try:
        await engine.submit(job)
        engine.queue.remove(job.job_id)
        stream = engine.subscribe(job.job_id)
        await engine._execute(job)
        assert await wait_for(lambda: job.state is JobState.FAILED, timeout=0.2)
        assert await engine.wait_for_result(job.job_id) is None
        assert await asyncio.wait_for(stream.get(), 0.2) is None
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
    finally:
        await engine.stop(drain=False)
    stored = await engine.store.load_job_payload(job.job_id)
    assert stored is not None and stored.state is JobState.FAILED
