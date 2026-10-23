"""A clean recovery must not forward state left over from an older attempt."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import JobState
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.contracts.recovery import FaultRecoverySnapshot, ResumePoint
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node


@pytest.mark.parametrize(
    "snapshot_case",
    ["missing", "prompt", "model", "resume_model", "job_id", "valid", "legacy_model"],
)
async def test_replacement_node_receives_only_validated_resume_state_and_completes(
    config: BrokerConfig,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_case: str,
) -> None:
    nodes = tuple(
        make_node(node_id, execution_mode=ExecutionMode.AGENT_CONTAINER)
        for node_id in ("node-a", "node-b")
    )
    engine = BrokerEngine(replace(config, static_nodes=nodes))
    assignments: list[tuple[str, dict[str, Any]]] = []
    assigned = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assignments.append((node_id, frame))
        assigned.set()

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = make_job(attempt=1, max_latency_ms=60_000, kv_cache_state_id="prior-cache")
    job.resume_point = ResumePoint(
        decode_step=2,
        generated_token_count=2,
        kv_cache_ref="prior-cache",
        prompt_hash=job.payload.prompt_hash,
        model=job.payload.model,
    )
    snapshot = FaultRecoverySnapshot(
        job_id=job.job_id,
        node_id="node-a",
        attempt=job.attempt,
        model=job.payload.model,
        step=8,
        resume_point=ResumePoint(
            decode_step=8,
            generated_token_count=8,
            kv_cache_ref="latest-cache",
            prompt_hash=job.payload.prompt_hash,
            model=job.payload.model,
        ),
    )
    if snapshot_case == "prompt":
        snapshot.resume_point.prompt_hash = "different-prompt-hash"
    elif snapshot_case == "model":
        snapshot.model = "different-model"
    elif snapshot_case == "resume_model":
        snapshot.resume_point.model = "different-model"
    elif snapshot_case == "job_id":
        snapshot.job_id = "different-job"
    elif snapshot_case == "legacy_model":
        snapshot.model = ""
        snapshot.resume_point.model = ""

    async def latest_for_job(job_id: str) -> FaultRecoverySnapshot | None:
        assert job_id == job.job_id
        return None if snapshot_case == "missing" else snapshot

    await engine.start()
    try:
        await engine.submit(job)
        await asyncio.wait_for(assigned.wait(), timeout=1.0)
        failed_node_id, first_frame = assignments[0]
        assert first_frame["kv_cache_state_id"] == "prior-cache"
        monkeypatch.setattr(engine.store, "latest_for_job", latest_for_job)

        await engine.supervisor.on_node_lost(failed_node_id, "synthetic node loss")

        assert len(assignments) == 2
        replacement_node_id, replacement_frame = assignments[1]
        assert replacement_node_id != failed_node_id
        assert replacement_node_id == job.assigned_node_id
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=replacement_node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": job.job_id, "text": "completed on the replacement node"},
            )
        )
        assert await engine.wait_for_result(job.job_id, timeout=1.0) == (
            "completed on the replacement node"
        )
        assert job.state is JobState.COMPLETED
        assert engine.registry.jobs_on(failed_node_id) == ()
        assert engine.registry.jobs_on(replacement_node_id) == ()
        assert job.attempt == 2

        valid_snapshot = snapshot_case in {"valid", "legacy_model"}
        expected_resume = snapshot.resume_point.model_dump(mode="json") if valid_snapshot else None
        expected_cache = "latest-cache" if valid_snapshot else None
        assert replacement_frame["resume_point"] == expected_resume
        assert replacement_frame["kv_cache_state_id"] == expected_cache
        assert replacement_frame["job"]["resume_point"] == expected_resume
        assert replacement_frame["job"]["kv_cache_state_id"] == expected_cache
        assert job.is_resumed is valid_snapshot
    finally:
        await engine.stop(drain=False)
