"""Primary placement must respect occupied slots until a job actually finishes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import JobState, PriorityTier, utcnow
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.routing.pd_dag import PDDagRouter
from horizon_broker.scheduler.batcher import BatchPlanner
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for

MODEL = "synthetic-model"


async def complete(engine: BrokerEngine, node_id: str, job_id: str) -> None:
    await engine.manager.handle_event(
        NodeEvent(
            node_id=node_id,
            type=NodeEventType.COMPLETE,
            data={"job_id": job_id, "text": f"result from {node_id}"},
        )
    )


def test_planner_omits_overflow_without_mutating_the_fleet_snapshot() -> None:
    nodes = [make_node(name, max_batch=1, models=(MODEL,)) for name in ("node-a", "node-b")]
    jobs = [make_job(tier=PriorityTier.FLEX, model=MODEL) for _ in range(3)]

    plans = BatchPlanner(PDDagRouter()).plan(jobs, nodes)

    assert [(plan.node_id, plan.size) for plan in plans] == [("node-a", 1), ("node-b", 1)]
    assert {job.job_id for plan in plans for job in plan.jobs} == {
        jobs[0].job_id,
        jobs[1].job_id,
    }
    assert all(node.current_batch_size == 0 and not node.assigned_job_ids for node in nodes)


@pytest.mark.parametrize("exclude_alternate", [False, True])
async def test_dispatch_refreshes_stale_capacity_without_losing_exclusions(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, exclude_alternate: bool
) -> None:
    nodes = tuple(
        make_node(name, max_batch=1, models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER)
        for name in ("node-a", "node-b", "node-c")
    )
    engine = BrokerEngine(replace(config, static_nodes=nodes))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr(engine.manager, "assign", assign)
    first = make_job(model=MODEL)
    second = make_job(model=MODEL)
    stale = engine.dispatcher.route(second, excluded=frozenset({"node-c"}))
    assert stale.primary_node_id == "node-a"
    try:
        await engine.dispatcher.dispatch(first)
        if exclude_alternate:
            with pytest.raises(NoCapacityError):
                await engine.dispatcher.dispatch(second, stale, excluded=frozenset({"node-b"}))
            assert second.assigned_node_id is None
            assert engine.supervisor.active_jobs() == [first.job_id]
        else:
            refreshed = await engine.dispatcher.dispatch(second, stale)
            assert refreshed.primary_node_id == "node-b"
            assert second.assigned_node_id == "node-b"
            await complete(engine, "node-b", second.job_id)
            assert second.state is JobState.COMPLETED
            assert second.result_text == "result from node-b"
        await complete(engine, "node-a", first.job_id)
        assert first.state is JobState.COMPLETED
    finally:
        await engine.stop(drain=False)


def test_registry_preserves_reported_load_and_ignores_duplicate_releases() -> None:
    registry = NodeRegistry()
    registry.upsert(make_node("node-a", max_batch=2, current_batch=1))

    registry.assign_job("node-a", "synthetic-job")
    assert registry.snapshot()[0]["current_batch_size"] == 2
    registry.assign_job("node-a", "synthetic-job")
    registry.release_job("node-a", "unknown-job")
    assert registry.snapshot()[0]["current_batch_size"] == 2

    registry.release_job("node-a", "synthetic-job")
    assert registry.snapshot()[0]["current_batch_size"] == 1
    registry.release_job("node-a", "synthetic-job")
    assert registry.snapshot()[0]["current_batch_size"] == 1


def test_delayed_load_reports_cannot_hide_active_assignments() -> None:
    registry = NodeRegistry()
    registry.upsert(make_node("node-a", max_batch=1))
    registry.assign_job("node-a", "synthetic-job")
    node = registry.get("node-a")
    assert node is not None

    registry.upsert(node.model_copy(update={"current_batch_size": 0}))

    current = registry.get("node-a")
    assert current is not None
    assert current.batch_headroom == 0
    assert registry.jobs_on("node-a") == ("synthetic-job",)


@pytest.mark.parametrize("reported_jobs", [(), ("unowned-job",)])
def test_full_capability_updates_preserve_broker_owned_assignments(
    reported_jobs: tuple[str, ...],
) -> None:
    registry = NodeRegistry()
    registry.upsert(make_node("node-a", max_batch=1))
    registry.assign_job("node-a", "synthetic-job")
    incoming = make_node("node-a", max_batch=1, current_batch=0).model_copy(
        update={"assigned_job_ids": reported_jobs}
    )

    current = registry.upsert(incoming)

    assert current.assigned_job_ids == ("synthetic-job",)
    assert current.batch_headroom == 0
    registry.release_job("node-a", "synthetic-job")
    assert registry.snapshot()[0]["current_batch_size"] == 0
    assert registry.jobs_on("node-a") == ()


async def test_reported_load_defers_work_until_a_broker_slot_is_released(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node(
        "node-a", max_batch=2, current_batch=1, models=(MODEL,),
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    assignments: asyncio.Queue[str] = asyncio.Queue()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assignments.put_nowait(frame["job"]["job_id"])

    monkeypatch.setattr(engine.manager, "assign", assign)
    jobs = [make_job(model=MODEL, max_latency_ms=60_000) for _ in range(2)]
    await engine.start()
    try:
        await engine.submit(jobs[0])
        assert await asyncio.wait_for(assignments.get(), timeout=1.0) == jobs[0].job_id
        await engine.submit(jobs[1])
        assert await wait_for(
            lambda: engine.metrics.counter("horizon_jobs_deferred_total") == 1,
            timeout=0.2,
        )
        assert assignments.empty()
        assert engine.queue.contains(jobs[1].job_id)
        assert engine.registry.snapshot()[0]["current_batch_size"] == 2

        await complete(engine, node.node_id, jobs[0].job_id)
        assert engine.registry.snapshot()[0]["current_batch_size"] == 1
        jobs[1].not_before = utcnow()
        engine.queue.wakeup.set()
        assert await asyncio.wait_for(assignments.get(), timeout=1.0) == jobs[1].job_id
        await complete(engine, node.node_id, jobs[1].job_id)

        assert all(job.state is JobState.COMPLETED for job in jobs)
        assert all(job.result_text == "result from node-a" for job in jobs)
        assert engine.registry.snapshot()[0]["current_batch_size"] == 1
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.metrics.counter("horizon_jobs_completed_total") == 2
    finally:
        await engine.stop(drain=False)


async def test_flex_overflow_completes_without_exceeding_node_capacity(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = tuple(
        make_node(name, max_batch=1, models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER)
        for name in ("node-a", "node-b")
    )
    engine = BrokerEngine(
        replace(
            config,
            static_nodes=nodes,
            window=replace(
                config.window, min_batch_size=4, max_batch_size=4,
                window_seconds=60, min_dwell_seconds=0, idle_flush=False,
            ),
        )
    )
    assignments: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    peaks: dict[str, int] = {}

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        peaks[node_id] = max(peaks.get(node_id, 0), len(engine.registry.jobs_on(node_id)))
        assignments.put_nowait((node_id, frame["job"]["job_id"]))

    monkeypatch.setattr(engine.manager, "assign", assign)
    jobs = [make_job(tier=PriorityTier.FLEX, model=MODEL, max_latency_ms=60_000) for _ in range(4)]
    await engine.start()
    try:
        for job in jobs:
            await engine.submit(job)
        assert await wait_for(lambda: assignments.qsize() >= 2)
        assert peaks == {"node-a": 1, "node-b": 1}
        assert assignments.qsize() == 2
        assert engine.queue.stats().flex == 2
        for _ in range(2):
            node_id, job_id = assignments.get_nowait()
            await complete(engine, node_id, job_id)

        # Release deferred work explicitly; the test does not wait through retry backoff.
        engine.window_policy = replace(engine.window_policy, min_batch_size=1)
        for job in engine.queue.pending_flex():
            job.not_before = utcnow()
        engine.queue.wakeup.set()
        assert await wait_for(lambda: assignments.qsize() >= 2)
        for _ in range(2):
            node_id, job_id = assignments.get_nowait()
            await complete(engine, node_id, job_id)

        assert all(job.state is JobState.COMPLETED for job in jobs)
        assert all(job.result_text == f"result from {job.assigned_node_id}" for job in jobs)
        assert peaks == {"node-a": 1, "node-b": 1}
        assert engine.queue.stats().flex == 0
        assert engine.supervisor.active_jobs() == []
        assert engine.metrics.counter("horizon_jobs_completed_total") == 4
    finally:
        await engine.stop(drain=False)
