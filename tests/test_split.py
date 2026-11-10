"""Prefill/decode disaggregation: the two-phase execution path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from horizon_broker.config import RoutingConfig
from horizon_broker.contracts.jobs import ExecutionPhase, JobState
from horizon_broker.contracts.nodes import ExecutionMode, NodeRole
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.nodes.executor import JobSupervisor
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.observability.metrics import MetricsRegistry
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.recovery.snapshot_store import SnapshotStore
from horizon_broker.routing.pd_dag import PDDagRouter
from tests.conftest import make_job, make_node

MODEL = "qwen2.5:7b"


def build(tmp_path: Path) -> tuple[JobSupervisor, NodeRegistry, list[tuple[str, dict[str, Any]]]]:
    """Build a supervisor over a prefill node and a decode node."""
    registry = NodeRegistry()
    prefill = make_node(
        "m3-max",
        role=NodeRole.PREFILL,
        prefill=True,
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    decode = make_node(
        "pi-decode",
        role=NodeRole.DECODE,
        decode=True,
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    registry.upsert(prefill)
    registry.upsert(decode)
    frames: list[tuple[str, dict[str, Any]]] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        frames.append((node_id, frame))

    supervisor = JobSupervisor(
        registry=registry,
        store=SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl")),
        assign=assign,
        metrics=MetricsRegistry(),
    )
    return supervisor, registry, frames


def split_decision(job: Any, registry: NodeRegistry) -> Any:
    router = PDDagRouter(config=RoutingConfig(kv_handoff_models=(MODEL,)))
    return router.route(job, registry.all())


async def test_a_split_job_prefills_on_one_node_and_decodes_on_another(tmp_path) -> None:
    supervisor, registry, frames = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096, max_latency_ms=600_000)
    decision = split_decision(job, registry)
    assert decision.is_split

    await supervisor.start(job, decision)
    assert frames[0][0] == "m3-max"
    assert frames[0][1]["phase"] == ExecutionPhase.PREFILL.value
    assert job.phase is ExecutionPhase.PREFILL

    # The prefill node finishes and hands over its KV cache.
    await supervisor.handle_event(
        NodeEvent(
            node_id="m3-max",
            type=NodeEventType.COMPLETE,
            data={"job_id": job.job_id, "kv_cache_ref": "kv-cache-1"},
        )
    )
    assert len(frames) == 2
    assert frames[1][0] == "pi-decode"
    assert frames[1][1]["phase"] == ExecutionPhase.DECODE.value
    assert frames[1][1]["kv_cache_state_id"] == "kv-cache-1"
    assert job.kv_cache_state_id == "kv-cache-1"

    await supervisor.handle_event(
        NodeEvent(node_id="pi-decode", type=NodeEventType.COMPLETE, data={"text": "done"})
    )
    assert job.state is JobState.COMPLETED
    assert job.result_text == "done"
    assert job.phase is ExecutionPhase.DECODE


async def test_failed_split_prefill_releases_the_initial_reservation(tmp_path) -> None:
    supervisor, registry, _ = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)

    async def reject_prefill(node_id: str, frame: dict[str, Any]) -> None:
        raise RuntimeError("synthetic prefill assignment failure")

    supervisor._assign = reject_prefill
    with pytest.raises(RuntimeError, match="prefill assignment failure"):
        await supervisor.start(job, decision)
    assert job.state is JobState.QUEUED
    assert job.phase is ExecutionPhase.MONOLITHIC
    assert job.assigned_node_id is None
    assert supervisor.active_jobs() == []
    assert registry.jobs_on("m3-max") == ()
    assert registry.jobs_on("pi-decode") == ()


async def test_failed_split_decode_handoff_finishes_the_job(tmp_path) -> None:
    supervisor, registry, frames = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)

    async def reject_decode(node_id: str, frame: dict[str, Any]) -> None:
        frames.append((node_id, frame))
        if frame["phase"] == ExecutionPhase.DECODE.value:
            raise RuntimeError("synthetic decode assignment failure")

    supervisor._assign = reject_decode
    await supervisor.start(job, decision)
    await supervisor.handle_event(
        NodeEvent(
            node_id="m3-max",
            type=NodeEventType.COMPLETE,
            data={"job_id": job.job_id, "kv_cache_ref": "kv-cache-1"},
        )
    )
    assert job.state is JobState.FAILED
    assert "decode assignment failed" in (job.error or "")
    assert supervisor.active_jobs() == []
    assert registry.jobs_on("m3-max") == ()
    assert registry.jobs_on("pi-decode") == ()


async def test_a_frame_without_a_job_id_is_attributed_to_the_only_active_job(tmp_path) -> None:
    supervisor, registry, frames = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)
    await supervisor.start(job, decision)

    # No job id in the frame: the node owns exactly one job, so it is safe.
    await supervisor.handle_event(
        NodeEvent(node_id="m3-max", type=NodeEventType.COMPLETE, data={"kv_cache_ref": "kv-9"})
    )
    assert len(frames) == 2
    assert frames[1][1]["kv_cache_state_id"] == "kv-9"


async def test_a_split_without_a_kv_cache_falls_back_to_a_whole_run(tmp_path) -> None:
    supervisor, registry, frames = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)

    await supervisor.start(job, decision)
    await supervisor.handle_event(
        NodeEvent(node_id="m3-max", type=NodeEventType.COMPLETE, data={})
    )

    # The job is re-run whole on the decode node instead of failing.
    assert len(frames) == 2
    assert frames[1][0] == "pi-decode"
    assert frames[1][1]["phase"] == ExecutionPhase.MONOLITHIC.value
    assert supervisor.metrics.counter("horizon_splits_degraded_total") == 1


async def test_failed_degraded_decode_handoff_finishes_the_job(tmp_path) -> None:
    supervisor, registry, _ = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)

    async def reject_decode(node_id: str, frame: dict[str, Any]) -> None:
        if frame["phase"] == ExecutionPhase.MONOLITHIC.value:
            raise RuntimeError("synthetic degraded assignment failure")

    supervisor._assign = reject_decode
    await supervisor.start(job, decision)
    await supervisor.handle_event(
        NodeEvent(node_id="m3-max", type=NodeEventType.COMPLETE, data={})
    )
    assert job.state is JobState.FAILED
    assert "degraded decode assignment failed" in (job.error or "")
    assert supervisor.active_jobs() == []
    assert registry.jobs_on("m3-max") == ()
    assert registry.jobs_on("pi-decode") == ()


async def test_decode_rollback_preserves_a_synchronous_completion(tmp_path) -> None:
    supervisor, registry, _ = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)

    async def complete_then_reject(node_id: str, frame: dict[str, Any]) -> None:
        if frame["phase"] == ExecutionPhase.DECODE.value:
            await supervisor.handle_event(
                NodeEvent(
                    node_id=node_id,
                    type=NodeEventType.COMPLETE,
                    data={"job_id": job.job_id, "text": "completed-at-decode"},
                )
            )
            raise RuntimeError("synthetic decode acknowledgement contradiction")

    supervisor._assign = complete_then_reject
    await supervisor.start(job, decision)
    await supervisor.handle_event(
        NodeEvent(
            node_id="m3-max",
            type=NodeEventType.COMPLETE,
            data={"job_id": job.job_id, "kv_cache_ref": "kv-cache-2"},
        )
    )

    assert job.state is JobState.COMPLETED
    assert job.result_text == "completed-at-decode"
    assert supervisor.active_jobs() == []
    assert registry.jobs_on("m3-max") == ()
    assert registry.jobs_on("pi-decode") == ()


async def test_a_direct_ollama_node_cannot_run_a_split(tmp_path) -> None:
    registry = NodeRegistry()
    prefill = make_node(
        "m3-max",
        role=NodeRole.PREFILL,
        prefill=True,
        execution_mode=ExecutionMode.OLLAMA_DIRECT,
        ollama_url="http://127.0.0.1:1",
    )
    decode = make_node(
        "pi-decode",
        role=NodeRole.DECODE,
        decode=True,
        execution_mode=ExecutionMode.AGENT_CONTAINER,
    )
    registry.upsert(prefill)
    registry.upsert(decode)

    frames: list[tuple[str, dict[str, Any]]] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        frames.append((node_id, frame))

    supervisor = JobSupervisor(
        registry=registry,
        store=SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl")),
        assign=assign,
    )
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    decision = split_decision(job, registry)
    assert decision.is_split

    await supervisor.start(job, decision)
    # The split is not executable without a KV export, so the job runs whole.
    assert job.phase is ExecutionPhase.MONOLITHIC
    assert supervisor.metrics.counter("horizon_splits_degraded_total") == 1


async def test_without_kv_handoff_support_the_router_never_splits(tmp_path) -> None:
    supervisor, registry, _ = build(tmp_path)
    job = make_job(model=MODEL, prompt_tokens=32_768, output_tokens=4_096)
    router = PDDagRouter()
    decision = router.route(job, registry.all())
    assert not decision.is_split
