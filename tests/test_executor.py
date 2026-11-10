"""The supervisor: node events that must not lie about a job's state."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import ExecutionPhase, JobState
from horizon_broker.contracts.nodes import ExecutionMode, NodeHealth
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.routing.pd_dag import PDDagRouter
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node
from tests.test_engine import start_fake_ollama


def build_engine(config: Any, url: str | None = None) -> BrokerEngine:
    node = make_node("m3-max", ollama_url=url)
    return BrokerEngine(replace(config, static_nodes=(node,)))


async def test_a_draining_frame_stores_a_node_health(config: Any) -> None:
    """A health verdict is an enum, not a string.

    The watchdog branches on identity, so a frame that stores the raw text
    ``"draining"`` makes the drain-and-migrate path unreachable.
    """
    engine = build_engine(config)
    await engine.supervisor.handle_event(
        NodeEvent(node_id="m3-max", type=NodeEventType.DRAINING, data={"health": "draining"})
    )

    node = engine.registry.get("m3-max")
    assert node is not None
    assert node.health is NodeHealth.DRAINING


async def test_a_transient_socket_error_does_not_fail_the_running_job(config: Any) -> None:
    """A dropped socket is the watchdog's business, not the job's.

    The connection retries with a backoff; failing the job on the first
    blip throws away work that the node may still be running.
    """
    server, url = await start_fake_ollama(tokens=("slow",), delay=30.0)
    engine = build_engine(config, url)
    try:
        job = make_job()
        decision = PDDagRouter().route(job, engine.registry.all())
        await engine.supervisor.start(job, decision)
        assert job.state is JobState.DISPATCHED

        await engine.supervisor.handle_event(
            NodeEvent(
                node_id="m3-max",
                type=NodeEventType.ERROR,
                data={"kind": "unreachable", "error": "connection reset by peer"},
            )
        )
        assert job.state is not JobState.FAILED
        assert job.error is None
    finally:
        await engine.stop(drain=False)
        await server.close()


async def test_a_malformed_checkpoint_does_not_mutate_the_running_job(config: Any) -> None:
    """A bad checkpoint frame must not corrupt state or break event handling."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())

    async def assign(_node_id: str, _frame: dict[str, Any]) -> None:
        return None

    engine.supervisor._assign = assign
    await engine.supervisor.start(job, decision)
    context = engine.supervisor.context(job.job_id)
    assert context is not None
    context.tokens_generated = 4
    context.chunks = ["stable"]

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={
                "job_id": job.job_id,
                "tokens_generated": "not-an-integer",
                "partial_tokens": ["corrupt"],
                "kv_cache_ref": "corrupt-kv",
            },
        )
    )

    assert context.tokens_generated == 4
    assert context.chunks == ["stable"]
    assert context.kv_cache_state_id is None
    assert job.state is JobState.DISPATCHED
    assert engine.supervisor.context(job.job_id) is context

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={
                "job_id": job.job_id,
                "tokens_generated": 5,
                "partial_tokens": [{"index": True, "text": "corrupt"}],
            },
        )
    )

    assert context.tokens_generated == 4
    assert context.chunks == ["stable"]
    assert context.kv_cache_state_id is None

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={
                "job_id": job.job_id,
                "tokens_generated": 5,
                "partial_tokens": [
                    {"index": 0, "text": "first"},
                    {"index": 2, "text": "gap"},
                ],
            },
        )
    )

    assert context.tokens_generated == 4
    assert context.chunks == ["stable"]
    assert context.kv_cache_state_id is None

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={
                "job_id": job.job_id,
                "tokens_generated": 5,
                "kv_cache_ref": 123,
                "kv_cache_state_id": "stable-kv",
            },
        )
    )

    assert context.tokens_generated == 5
    assert context.chunks == ["stable"]
    assert context.kv_cache_state_id == "stable-kv"

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={
                "job_id": job.job_id,
                "tokens_generated": 6,
                "kv_cache_ref": "cache-a",
                "kv_cache_state_id": "cache-b",
            },
        )
    )

    assert context.tokens_generated == 5
    assert context.kv_cache_state_id == "stable-kv"

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.CHECKPOINT,
            data={"job_id": job.job_id, "tokens_generated": 3},
        )
    )

    assert context.tokens_generated == 5
    assert context.chunks == ["stable"]
    await engine.stop(drain=False)


async def test_a_completion_without_text_does_not_succeed_the_job(config: Any) -> None:
    """A malformed terminal frame cannot turn an active job into an empty success."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        return None

    engine.supervisor._assign = assign
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())
    await engine.supervisor.start(job, decision)

    await engine.supervisor.handle_event(
        NodeEvent(node_id=node.node_id, type=NodeEventType.COMPLETE, data={"job_id": job.job_id})
    )

    assert job.state is JobState.DISPATCHED
    assert job.result_text is None
    await engine.stop(drain=False)


async def test_a_malformed_heartbeat_cannot_poison_node_capacity(config: Any) -> None:
    """Heartbeat values are untrusted and must remain inside routing bounds."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.HEARTBEAT,
            data={
                "available_vram_bytes": -1,
                "total_vram_bytes": 10,
                "current_batch_size": -2,
                "max_batch_size": 0,
                "tokens_per_second": 10**1000,
                "power_headroom": 10**1000,
            },
        )
    )

    refreshed = engine.registry.get(node.node_id)
    assert refreshed is not None
    assert refreshed.available_vram_bytes == 10
    assert refreshed.total_vram_bytes == 10
    assert refreshed.current_batch_size == 0
    assert refreshed.max_batch_size == node.max_batch_size
    assert refreshed.tokens_per_second == node.tokens_per_second
    assert refreshed.power_headroom == 1.0


async def test_a_partial_heartbeat_does_not_promote_local_occupancy_to_remote(
    config: Any,
) -> None:
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    engine.registry.assign_job(node.node_id, "local-job")

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.HEARTBEAT,
            data={"available_vram_bytes": node.available_vram_bytes - 1},
        )
    )
    engine.registry.release_job(node.node_id, "local-job")

    refreshed = engine.registry.get(node.node_id)
    assert refreshed is not None
    assert refreshed.current_batch_size == 0


async def test_a_full_heartbeat_replaces_the_remote_occupancy_baseline(
    config: Any,
) -> None:
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    engine.registry.assign_job(node.node_id, "local-job")

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.HEARTBEAT,
            data={
                "current_batch_size": 2,
                "max_batch_size": node.max_batch_size,
            },
        )
    )
    engine.registry.release_job(node.node_id, "local-job")

    refreshed = engine.registry.get(node.node_id)
    assert refreshed is not None
    assert refreshed.current_batch_size == 2


async def test_a_malformed_progress_count_cannot_rewind_execution(config: Any) -> None:
    """Progress counters cannot move a running job backwards or become booleans."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())

    async def assign(_node_id: str, _frame: dict[str, Any]) -> None:
        return None

    engine.supervisor._assign = assign
    await engine.supervisor.start(job, decision)
    context = engine.supervisor.context(job.job_id)
    assert context is not None
    context.tokens_generated = 4

    for invalid_count in (-1, True):
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=node.node_id,
                type=NodeEventType.PROGRESS,
                data={"job_id": job.job_id, "tokens_generated": invalid_count},
            )
        )

    assert context.tokens_generated == 4
    assert job.state is JobState.DISPATCHED

    await engine.supervisor.handle_event(
        NodeEvent(
            node_id=node.node_id,
            type=NodeEventType.PROGRESS,
            data={"job_id": job.job_id, "tokens_generated": 2},
        )
    )

    assert context.tokens_generated == 4
    assert job.state is JobState.RUNNING
    await engine.stop(drain=False)


async def test_direct_start_rechecks_node_capacity_before_reserving(config: Any) -> None:
    """A stale routing decision cannot overfill a node at admission time."""
    engine = build_engine(config)
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())
    node = engine.registry.get("m3-max")
    assert node is not None
    engine.registry.upsert(node.model_copy(update={"current_batch_size": node.max_batch_size}))
    with pytest.raises(NoCapacityError, match="no longer accepts"):
        await engine.supervisor.start(job, decision)
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on("m3-max") == ()


async def test_cancelled_direct_assignment_rolls_back_ownership(config: Any) -> None:
    """Cancellation during a control send must release the provisional admission."""
    node = make_node("m3-max", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())

    async def cancel_assignment(node_id: str, frame: dict[str, Any]) -> None:
        raise asyncio.CancelledError

    engine.supervisor._assign = cancel_assignment
    with pytest.raises(asyncio.CancelledError):
        await engine.supervisor.start(job, decision)
    assert job.state is JobState.QUEUED
    assert job.assigned_node_id is None
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on("m3-max") == ()


async def test_recovery_error_cannot_overwrite_a_synchronous_completion(config: Any) -> None:
    """A recovery callback's terminal result survives a later callback error."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())

    async def assign(_node_id: str, _frame: dict[str, Any]) -> None:
        return None

    async def recover(recovered, _node_id: str, _reason: str):
        recovered.state = JobState.COMPLETED
        recovered.result_text = "completed-during-recovery"
        raise RuntimeError("synthetic recovery acknowledgement contradiction")

    engine.supervisor._assign = assign
    engine.supervisor._recover = recover
    await engine.supervisor.start(job, decision)
    await engine.supervisor.on_node_lost(node.node_id, "synthetic loss")

    assert job.state is JobState.COMPLETED
    assert job.result_text == "completed-during-recovery"


async def test_cancel_cannot_overwrite_a_synchronous_completion(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completion received while cancel is in flight remains authoritative."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if frame["type"] == "cancel":
            await engine.supervisor.handle_event(
                NodeEvent(
                    node_id=node_id,
                    type=NodeEventType.COMPLETE,
                    data={"job_id": frame["job_id"], "text": "completed-before-cancel-return"},
                )
            )

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job()
    engine._jobs[job.job_id] = job
    decision = PDDagRouter().route(job, engine.registry.all())
    await engine.supervisor.start(job, decision)

    assert await engine.cancel(job.job_id) is False
    assert job.state is JobState.COMPLETED
    assert job.result_text == "completed-before-cancel-return"


async def test_a_late_ollama_completion_cannot_overwrite_cancellation(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task finishing after cancellation cannot publish a stale result."""
    node = make_node("ollama-node", ollama_url="http://synthetic-ollama")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    engine._jobs[job.job_id] = job
    decision = PDDagRouter().route(job, engine.registry.all())
    context = await engine.supervisor.start(job, decision)

    assert await engine.supervisor.cancel(job.job_id) is True
    await engine.supervisor._complete(context, "late-result")

    assert job.state is JobState.CANCELLED
    assert job.result_text is None


async def test_cancelled_cancel_frame_still_releases_local_ownership(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled outbound cancel send cannot strand the local execution."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def cancel_frame(node_id: str, frame: dict[str, Any]) -> None:
        if frame["type"] == "cancel":
            raise asyncio.CancelledError

    monkeypatch.setattr(engine.manager, "assign", cancel_frame)
    job = make_job()
    engine._jobs[job.job_id] = job
    decision = PDDagRouter().route(job, engine.registry.all())
    await engine.supervisor.start(job, decision)

    with pytest.raises(asyncio.CancelledError):
        await engine.supervisor.cancel(job.job_id)
    assert job.state is JobState.CANCELLED
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on(node.node_id) == ()


async def test_node_loss_without_recovery_callback_fails_and_releases_job(
    config: Any,
) -> None:
    """A supervisor without recovery authority must fail lost work explicitly."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        return None

    engine.supervisor._assign = assign
    engine.supervisor._recover = None
    job = make_job()
    engine._jobs[job.job_id] = job
    decision = PDDagRouter().route(job, engine.registry.all())
    await engine.supervisor.start(job, decision)

    await engine.supervisor.on_node_lost(node.node_id, "synthetic loss")

    assert job.state is JobState.FAILED
    assert "no recovery coordinator" in (job.error or "")
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on(node.node_id) == ()


async def test_direct_start_rechecks_node_health_before_reserving(config: Any) -> None:
    """A stale route cannot dispatch onto a node that became unavailable."""
    engine = build_engine(config)
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())
    engine.registry.mark_health("m3-max", NodeHealth.LOST, "synthetic stale route")
    with pytest.raises(NoCapacityError, match="no longer accepts"):
        await engine.supervisor.start(job, decision)
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on("m3-max") == ()


async def test_failed_direct_assignment_rolls_back_ownership(config: Any) -> None:
    """A failed control send must not strand a dispatched job or reservation."""
    node = make_node("m3-max", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    job = make_job()
    decision = PDDagRouter().route(job, engine.registry.all())

    async def reject_assignment(node_id: str, frame: dict[str, Any]) -> None:
        raise RuntimeError("synthetic assignment failure")

    engine.supervisor._assign = reject_assignment
    with pytest.raises(RuntimeError, match="assignment failure"):
        await engine.supervisor.start(job, decision)
    assert job.state is JobState.QUEUED
    assert job.phase is ExecutionPhase.MONOLITHIC
    assert job.assigned_node_id is None
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on("m3-max") == ()
