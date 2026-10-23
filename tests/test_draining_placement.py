"""A draining node can finish accepted work but cannot receive new placement."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import JobState, PriorityTier
from horizon_broker.contracts.nodes import ExecutionMode, NodeHealth
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.routing.pd_dag import PDDagRouter
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node

MODEL = "synthetic-model"


@pytest.mark.parametrize("tier", [PriorityTier.INTERACTIVE, PriorityTier.FLEX])
def test_routing_excludes_draining_nodes_in_both_lanes(tier: PriorityTier) -> None:
    draining = make_node("node-a", models=(MODEL,)).model_copy(
        update={"health": NodeHealth.DRAINING}
    )
    healthy = make_node("node-b", models=(MODEL,))
    job = make_job(tier=tier, model=MODEL)
    router = PDDagRouter()

    assert router.route(job, [draining, healthy]).primary_node_id == healthy.node_id
    with pytest.raises(NoCapacityError):
        router.route(job, [draining])
    assert draining.health is NodeHealth.DRAINING
    assert draining.current_batch_size == 0


@pytest.mark.parametrize("tier", [PriorityTier.INTERACTIVE, PriorityTier.FLEX])
@pytest.mark.parametrize("recover", [False, True], ids=["queued", "recovered"])
async def test_work_completes_on_an_alternate_node_that_is_not_draining(
    config: BrokerConfig,
    monkeypatch: pytest.MonkeyPatch,
    tier: PriorityTier,
    recover: bool,
) -> None:
    nodes = [
        make_node(name, models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER)
        for name in (("node-a", "node-b", "node-c") if recover else ("node-a", "node-b"))
    ]
    draining_index = 1 if recover else 0
    nodes[draining_index] = nodes[draining_index].model_copy(update={"health": NodeHealth.DRAINING})
    expected_node = "node-c" if recover else "node-b"
    engine = BrokerEngine(replace(config, static_nodes=tuple(nodes)))
    initially_assigned = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if recover and frame["job"]["attempt"] == 0:
            initially_assigned.set()
            return
        await engine.manager.handle_event(
            NodeEvent(
                node_id=node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": frame["job"]["job_id"], "text": f"result from {node_id}"},
            )
        )

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job(tier=tier, model=MODEL, max_latency_ms=60_000)
    await engine.start()
    try:
        await engine.submit(job)
        if recover:
            await asyncio.wait_for(initially_assigned.wait(), timeout=1.0)
            assert job.assigned_node_id == "node-a"
            await engine.supervisor.on_node_lost("node-a", "synthetic node loss")

        result = await asyncio.wait_for(engine.wait_for_result(job.job_id), timeout=1.0)
        assert result == f"result from {expected_node}"
        assert job.state is JobState.COMPLETED
        assert job.assigned_node_id == expected_node
        assert job.attempt == int(recover)
        assert engine.registry.jobs_on(nodes[draining_index].node_id) == ()
        assert engine.metrics.counter("horizon_jobs_completed_total") == 1
        assert engine.metrics.counter("horizon_jobs_failed_total") == 0
    finally:
        await engine.stop(drain=False)


async def test_placement_does_not_revoke_work_already_accepted_before_draining(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node("node-a", models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr(engine.manager, "assign", assign)
    accepted = make_job(model=MODEL, max_latency_ms=60_000)
    try:
        await engine.submit(accepted)
        engine.queue.remove(accepted.job_id)
        await engine.dispatcher.dispatch(accepted)
        engine.registry.mark_health(node.node_id, NodeHealth.DRAINING)

        with pytest.raises(NoCapacityError):
            engine.dispatcher.route(make_job(model=MODEL))
        assert accepted.state is JobState.DISPATCHED
        assert engine.registry.jobs_on(node.node_id) == (accepted.job_id,)

        await engine.manager.handle_event(
            NodeEvent(
                node_id=node.node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": accepted.job_id, "text": "accepted work completed"},
            )
        )
        assert await engine.wait_for_result(accepted.job_id) == "accepted work completed"
        assert accepted.state is JobState.COMPLETED
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.metrics.counter("horizon_jobs_completed_total") == 1
    finally:
        await engine.stop(drain=False)
