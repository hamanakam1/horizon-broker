"""The supervisor: node events that must not lie about a job's state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.nodes import NodeHealth
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
