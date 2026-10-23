"""A previous node cannot mutate a job after ownership moves elsewhere."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.nodes import ExecutionMode, NodeRole
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for


@pytest.mark.parametrize("split", [False, True], ids=["recovery", "split-handoff"])
@pytest.mark.parametrize(
    "event_type",
    [NodeEventType.PROGRESS, NodeEventType.CHECKPOINT, NodeEventType.ERROR, NodeEventType.COMPLETE],
    ids=["progress", "checkpoint", "error", "complete"],
)
async def test_old_node_frames_cannot_corrupt_the_new_execution(
    config: BrokerConfig,
    monkeypatch: pytest.MonkeyPatch,
    split: bool,
    event_type: NodeEventType,
) -> None:
    model = "synthetic-model"
    nodes = (
        make_node(
            "node-a",
            role=NodeRole.PREFILL if split else NodeRole.GENERAL,
            prefill=split,
            models=(model,),
            execution_mode=ExecutionMode.AGENT_CONTAINER,
        ),
        make_node(
            "node-b",
            role=NodeRole.DECODE if split else NodeRole.GENERAL,
            decode=split,
            models=(model,),
            execution_mode=ExecutionMode.AGENT_CONTAINER,
        ),
    )
    engine = BrokerEngine(
        replace(
            config,
            static_nodes=nodes,
            routing=replace(config.routing, kv_handoff_models=(model,) if split else ()),
        )
    )

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job(
        model=model,
        prompt_tokens=32_768 if split else None,
        output_tokens=4_096 if split else 64,
        max_latency_ms=60_000,
    )
    await engine.start()
    try:
        await engine.submit(job)
        assert await wait_for(lambda: job.assigned_node_id == "node-a")
        if split:
            await engine.manager.handle_event(
                NodeEvent(
                    node_id="node-a",
                    type=NodeEventType.COMPLETE,
                    data={"job_id": job.job_id, "kv_cache_ref": "current-kv"},
                )
            )
        else:
            await engine.supervisor.on_node_lost("node-a", "synthetic node loss")

        assert job.assigned_node_id == "node-b"
        stream = engine.subscribe(job.job_id)
        await engine.manager.handle_event(
            NodeEvent(
                node_id="node-a",
                type=event_type,
                data={
                    "job_id": job.job_id,
                    "text": "stale answer",
                    "tokens_generated": 123,
                    "partial_tokens": ["stale answer"],
                    "kv_cache_ref": "stale-kv",
                    "error": "stale error",
                },
            )
        )

        assert job.state is JobState.DISPATCHED
        assert job.kv_cache_state_id == ("current-kv" if split else None)
        assert job.error is None
        assert job.result_text is None
        assert stream.empty()
        assert engine.registry.jobs_on("node-b") == (job.job_id,)

        await engine.manager.handle_event(
            NodeEvent(
                node_id="node-b",
                type=NodeEventType.PROGRESS,
                data={"job_id": job.job_id, "text": "valid ", "tokens_generated": 1},
            )
        )
        await engine.manager.handle_event(
            NodeEvent(
                node_id="node-b",
                type=NodeEventType.COMPLETE,
                data={"job_id": job.job_id, "text": "answer"},
            )
        )

        assert job.state is JobState.COMPLETED
        assert job.assigned_node_id == "node-b"
        assert await engine.wait_for_result(job.job_id) == "valid answer"
        assert [stream.get_nowait(), stream.get_nowait(), stream.get_nowait()] == [
            "valid ",
            "answer",
            None,
        ]
        assert engine.metrics.counter("horizon_jobs_completed_total") == 1
        assert engine.metrics.counter("horizon_jobs_failed_total") == 0
        assert engine.supervisor.active_jobs() == []
    finally:
        await engine.stop(drain=False)
