"""A split owns its future decode slot before its first assignment can suspend."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import (
    ExecutionPhase,
    InferenceJob,
    JobState,
    PriorityTier,
    utcnow,
)
from horizon_broker.contracts.nodes import ExecutionMode, NodeCapability, NodeHealth, NodeRole
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.ollama.client import OllamaClient
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for

MODEL = "synthetic-model"


def split_nodes() -> tuple[NodeCapability, NodeCapability]:
    return (
        make_node(
            "node-a", role=NodeRole.PREFILL, prefill=True, max_batch=1,
            models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER,
        ),
        make_node(
            "node-b", role=NodeRole.DECODE, decode=True, max_batch=1,
            models=(MODEL,), execution_mode=ExecutionMode.AGENT_CONTAINER,
        ),
    )


def split_engine(config: BrokerConfig) -> BrokerEngine:
    return BrokerEngine(replace(
        config, static_nodes=split_nodes(),
        routing=replace(config.routing, kv_handoff_models=(MODEL,)),
        window=replace(config.window, min_batch_size=1, min_dwell_seconds=0),
    ))


def split_job(tier: PriorityTier = PriorityTier.INTERACTIVE) -> InferenceJob:
    return make_job(
        tier=tier, model=MODEL, prompt_tokens=32_768,
        output_tokens=4_096, max_latency_ms=60_000,
    )


async def prepare(engine: BrokerEngine, job: InferenceJob) -> None:
    await engine.submit(job)
    engine.queue.remove(job.job_id)


async def complete(
    engine: BrokerEngine, node_id: str, job: InferenceJob, **data: Any,
) -> None:
    await engine.supervisor.handle_event(NodeEvent(
        node_id=node_id, type=NodeEventType.COMPLETE,
        data={"job_id": job.job_id, **data},
    ))


def assert_released(engine: BrokerEngine) -> None:
    assert engine.supervisor.active_jobs() == []
    for node in engine.registry.all():
        assert node.assigned_job_ids == ()
        assert node.current_batch_size == 0


@pytest.mark.parametrize("cleanup", ["release", "remove"])
def test_pending_reservation_cleanup_cannot_return_as_phantom_load(cleanup: str) -> None:
    registry = NodeRegistry()
    reported = make_node("node-a", max_batch=2, current_batch=1)
    registry.upsert(reported)
    registry.reserve_job("node-a", "synthetic-future-job")
    registry.reserve_job("node-a", "synthetic-future-job")
    current = registry.upsert(reported)
    assert current.current_batch_size == 2
    if cleanup == "remove":
        registry.remove("node-a")
    else:
        registry.release_job("node-a", "synthetic-future-job")
        registry.release_job("node-a", "synthetic-future-job")
    current = registry.upsert(reported)
    assert current.current_batch_size == 1
    assert current.assigned_job_ids == ()


async def test_split_reserves_atomically_before_send_and_preserves_slots_across_heartbeats(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = split_engine(config)
    sending = asyncio.Event()
    finish_send = asyncio.Event()

    async def fail_send(node_id: str, frame: dict[str, Any]) -> None:
        sending.set()
        await finish_send.wait()
        raise OSError("synthetic prefill assignment failure")

    monkeypatch.setattr(engine.manager, "assign", fail_send)
    first, second = split_job(), split_job()
    stale_decision = engine.dispatcher.route(second)
    assert stale_decision.is_split
    await prepare(engine, first)
    pending = asyncio.create_task(engine._execute(first))
    try:
        await asyncio.wait_for(sending.wait(), 0.5)
        for node in engine.registry.all():
            assert node.assigned_job_ids == (first.job_id,)
            assert node.current_batch_size == node.max_batch_size == 1

        # A future decoder reports no running work; the broker still owns its slot.
        await engine.supervisor.handle_event(NodeEvent(
            node_id="node-b", type=NodeEventType.HEARTBEAT,
            data={"current_batch_size": 0},
        ))
        engine.registry.upsert(split_nodes()[1])
        assert engine.registry.jobs_on("node-b") == (first.job_id,)
        reserved_decoder = engine.registry.get("node-b")
        assert reserved_decoder is not None and reserved_decoder.current_batch_size == 1

        before = second.model_dump()
        with pytest.raises(NoCapacityError):
            await engine.supervisor.start(second, stale_decision)
        assert second.model_dump() == before
        assert engine.supervisor.context(second.job_id) is None
        assert engine.registry.jobs_on("node-a") == (first.job_id,)
        assert engine.registry.jobs_on("node-b") == (first.job_id,)

        finish_send.set()
        await asyncio.wait_for(pending, 0.5)
        assert first.state is JobState.FAILED
        assert await engine.wait_for_result(first.job_id) is None
        assert_released(engine)
    finally:
        finish_send.set()
        await asyncio.gather(pending, return_exceptions=True)
        await engine.stop(drain=False)
    saved = await engine.store.load_job_payload(first.job_id)
    assert saved is not None and saved.state is JobState.FAILED


@pytest.mark.parametrize("unavailable_node", ["node-a", "node-b"])
async def test_stale_split_with_one_full_node_cannot_partially_reserve_the_other(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, unavailable_node: str,
) -> None:
    engine = split_engine(config)

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = split_job()
    decision = engine.dispatcher.route(job)
    engine.registry.assign_job(unavailable_node, "synthetic-existing-job")
    before = job.model_dump()
    before_nodes = engine.registry.snapshot()
    try:
        with pytest.raises(NoCapacityError):
            await engine.supervisor.start(job, decision)
        assert engine.registry.snapshot() == before_nodes
        assert job.model_dump() == before
        assert engine.supervisor.context(job.job_id) is None

        engine.registry.release_job(unavailable_node, "synthetic-existing-job")
        await prepare(engine, job)
        await engine._execute(job)
        await complete(engine, "node-a", job, kv_cache_ref="synthetic-kv")
        await complete(engine, "node-b", job, text="result after capacity returned")
        assert await engine.wait_for_result(job.job_id) == "result after capacity returned"
        assert_released(engine)
    finally:
        await engine.stop(drain=False)


async def test_nonexecutable_split_reserves_only_its_whole_job_node(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = split_engine(config)
    decoder = split_nodes()[1].model_copy(update={"execution_mode": ExecutionMode.OLLAMA_DIRECT})
    engine.registry.upsert(decoder)

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assert frame["phase"] == "monolithic"

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = split_job()
    try:
        await prepare(engine, job)
        assert engine.dispatcher.route(job).is_split
        await engine._execute(job)
        assert engine.registry.jobs_on("node-a") == (job.job_id,)
        assert engine.registry.jobs_on("node-b") == ()
        assert engine.metrics.counter("horizon_splits_degraded_total") == 1
        await complete(engine, "node-a", job, text="whole result")
        assert await engine.wait_for_result(job.job_id) == "whole result"
        assert_released(engine)
    finally:
        await engine.stop(drain=False)


async def test_running_telemetry_cannot_spend_a_pending_decode_reservation(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = split_engine(config)
    decoder = split_nodes()[1].model_copy(update={"max_batch_size": 2, "current_batch_size": 1})
    engine.registry.upsert(decoder)
    first = split_job()
    second = make_job(model=MODEL, prompt_tokens=1, output_tokens=8_192, max_latency_ms=60_000)
    external_running = 1
    active_decode_jobs: set[str] = set()
    measured_peaks: list[int] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if node_id == "node-b":
            active_decode_jobs.add(frame["job"]["job_id"])
        pending_decode = first.phase is ExecutionPhase.PREFILL
        actual_and_reserved = external_running + len(active_decode_jobs) + int(pending_decode)
        measured_peaks.append(actual_and_reserved)
        assert actual_and_reserved <= 2

    monkeypatch.setattr(engine.manager, "assign", assign)
    try:
        await prepare(engine, first)
        await engine._execute(first)
        for _ in range(2):
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-b", type=NodeEventType.HEARTBEAT,
                data={"current_batch_size": external_running},
            ))
            current = engine.registry.get("node-b")
            assert current is not None and current.current_batch_size == 2
            engine.registry.upsert(decoder)
            current = engine.registry.get("node-b")
            assert current is not None and current.current_batch_size == 2
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-b", type=NodeEventType.HEARTBEAT,
                data={"tokens_per_second": 10.0},
            ))
            current = engine.registry.get("node-b")
            assert current is not None and current.current_batch_size == 2

        await prepare(engine, second)
        await engine._execute(second)
        assert engine.queue.contains(second.job_id)
        assert engine.supervisor.context(second.job_id) is None
        assert active_decode_jobs == set()

        external_running = 0
        await engine.supervisor.handle_event(NodeEvent(
            node_id="node-b", type=NodeEventType.HEARTBEAT,
            data={"current_batch_size": external_running},
        ))
        engine.queue.remove(second.job_id)
        await engine._execute(second)
        assert second.assigned_node_id == "node-b"
        await complete(engine, "node-a", first, kv_cache_ref="synthetic-kv")
        assert active_decode_jobs == {first.job_id, second.job_id}
        # Once activated, telemetry includes both jobs; it must not add decode twice.
        await engine.supervisor.handle_event(NodeEvent(
            node_id="node-b", type=NodeEventType.HEARTBEAT,
            data={"current_batch_size": 2},
        ))
        current = engine.registry.get("node-b")
        assert current is not None and current.current_batch_size == 2
        await complete(engine, "node-b", first, text="split result")
        await complete(engine, "node-b", second, text="competing result")
        assert await engine.wait_for_result(first.job_id) == "split result"
        assert await engine.wait_for_result(second.job_id) == "competing result"
        assert max(measured_peaks) == 2
        assert_released(engine)
    finally:
        await engine.stop(drain=False)


async def test_heartbeat_during_decode_send_cannot_spend_the_pending_slot(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = split_engine(config)
    decoder = split_nodes()[1].model_copy(update={"max_batch_size": 2, "current_batch_size": 1})
    engine.registry.upsert(decoder)
    first, second = split_job(), split_job()
    sending = asyncio.Event()
    finish_send = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if node_id == "node-b" and frame["job"]["job_id"] == first.job_id:
            sending.set()
            await finish_send.wait()

    monkeypatch.setattr(engine.manager, "assign", assign)
    await prepare(engine, first)
    await engine._execute(first)
    handoff = asyncio.create_task(complete(engine, "node-a", first, kv_cache_ref="synthetic-kv"))
    try:
        await asyncio.wait_for(sending.wait(), 0.5)
        await engine.supervisor.handle_event(NodeEvent(
            node_id="node-b", type=NodeEventType.HEARTBEAT,
            data={"current_batch_size": 1},
        ))
        current = engine.registry.get("node-b")
        assert current is not None and current.current_batch_size == 2
        # This job may only use the decoder; the freed prefill slot is excluded.
        engine.recovery.avoid_node(second.job_id, "node-a")
        await prepare(engine, second)
        await engine._execute(second)
        assert engine.queue.contains(second.job_id)
        assert engine.supervisor.context(second.job_id) is None

        finish_send.set()
        await asyncio.wait_for(handoff, 0.5)
        await complete(engine, "node-b", first, text="first result")
        current = engine.registry.get("node-b")
        assert current is not None and current.current_batch_size == 1
        # The external job finishes; dispatch can now reuse the released slot.
        await engine.supervisor.handle_event(NodeEvent(
            node_id="node-b", type=NodeEventType.HEARTBEAT,
            data={"current_batch_size": 0},
        ))
        engine.queue.remove(second.job_id)
        await engine._execute(second)
        await complete(engine, "node-b", second, text="second result")
        assert await engine.wait_for_result(first.job_id) == "first result"
        assert await engine.wait_for_result(second.job_id) == "second result"
        assert_released(engine)
    finally:
        finish_send.set()
        await asyncio.gather(handoff, return_exceptions=True)
        await engine.stop(drain=False)


@pytest.mark.parametrize("tier", [PriorityTier.INTERACTIVE, PriorityTier.FLEX])
@pytest.mark.parametrize("kv_ref", [None, "synthetic-kv"])
async def test_split_and_queued_overflow_complete_without_overbooking(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
    tier: PriorityTier, kv_ref: str | None,
) -> None:
    engine = split_engine(config)
    assignments: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
    peaks: dict[str, int] = {}

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        for node in engine.registry.all():
            peaks[node.node_id] = max(peaks.get(node.node_id, 0), node.current_batch_size)
            assert node.current_batch_size <= node.max_batch_size
        assignments.put_nowait((node_id, frame["job"]["job_id"], frame["phase"]))

    monkeypatch.setattr(engine.manager, "assign", assign)
    first = split_job(tier)
    second = make_job(tier=tier, model=MODEL, max_latency_ms=60_000)
    await engine.start()
    try:
        await engine.submit(first)
        assert await asyncio.wait_for(assignments.get(), 0.5) == (
            "node-a", first.job_id, "prefill",
        )
        assert all(node.assigned_job_ids == (first.job_id,) for node in engine.registry.all())
        await engine.submit(second)
        if tier is PriorityTier.INTERACTIVE:
            assert await wait_for(lambda: second.not_before is not None, timeout=0.5)
        # Flex's fleet gate retains queued jobs while every slot is occupied;
        # only the interactive lane attempts placement and adds retry backoff.
        assert assignments.empty()
        assert engine.queue.contains(second.job_id)

        first_stream = engine.subscribe(first.job_id)
        second_stream = engine.subscribe(second.job_id)
        await complete(engine, "node-a", first, kv_cache_ref=kv_ref)
        assert await asyncio.wait_for(assignments.get(), 0.5) == (
            "node-b", first.job_id, "decode" if kv_ref else "monolithic",
        )
        assert engine.registry.jobs_on("node-a") == ()
        assert engine.registry.jobs_on("node-b") == (first.job_id,)

        # Explicitly release retry backoff, so this test only waits for scheduling.
        second.not_before = utcnow()
        engine.queue.wakeup.set()
        assert await asyncio.wait_for(assignments.get(), 0.5) == (
            "node-a", second.job_id, "monolithic",
        )
        await complete(engine, "node-a", second, text="overflow result")
        await complete(engine, "node-b", first, text="split result")
        assert await engine.wait_for_result(first.job_id) == "split result"
        assert await engine.wait_for_result(second.job_id) == "overflow result"
        for stream, text in ((first_stream, "split result"), (second_stream, "overflow result")):
            assert await asyncio.wait_for(stream.get(), 0.5) == text
            assert await asyncio.wait_for(stream.get(), 0.5) is None
        assert peaks == {"node-a": 1, "node-b": 1}
        assert_released(engine)
    finally:
        await engine.stop(drain=False)
    for job in (first, second):
        saved = await engine.store.load_job_payload(job.job_id)
        assert saved is not None and saved.state is JobState.COMPLETED


@pytest.mark.parametrize("outcome", ["prefill-error", "cancel", "cancel-send-error", "decode-send-error"])
async def test_split_terminal_paths_release_every_reservation(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    engine = split_engine(config)

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if outcome == "cancel-send-error" and frame["type"] == "cancel":
            raise OSError("synthetic cancel failure")
        if outcome == "decode-send-error" and node_id == "node-b":
            raise OSError("synthetic decode failure")

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = split_job()
    try:
        await prepare(engine, job)
        await engine._execute(job)
        assert all(engine.registry.jobs_on(node) == (job.job_id,) for node in ("node-a", "node-b"))
        stream = engine.subscribe(job.job_id)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        await asyncio.sleep(0)
        expected = JobState.FAILED
        if outcome.startswith("cancel"):
            assert await engine.cancel(job.job_id)
            expected = JobState.CANCELLED
        elif outcome == "prefill-error":
            await engine.supervisor.handle_event(NodeEvent(
                node_id="node-a", type=NodeEventType.ERROR,
                data={"job_id": job.job_id, "error": "synthetic prefill failure"},
            ))
        else:
            await complete(engine, "node-a", job, kv_cache_ref="synthetic-kv")
        assert job.state is expected
        assert await asyncio.wait_for(waiter, 0.5) is None
        assert await asyncio.wait_for(stream.get(), 0.5) is None
        assert_released(engine)
    finally:
        await engine.stop(drain=False)
    saved = await engine.store.load_job_payload(job.job_id)
    assert saved is not None and saved.state is expected


@pytest.mark.parametrize("kv_ref", [None, "synthetic-kv"])
@pytest.mark.parametrize("unavailable", ["lost", "draining", "quarantined", "removed", "model", "mode"])
async def test_unavailable_reserved_decoder_fails_at_handoff_without_restarting_prefill(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch,
    unavailable: str, kv_ref: str | None,
) -> None:
    engine = split_engine(config)
    assignments: list[tuple[str, str]] = []

    def reject_direct_execution(node: NodeCapability) -> OllamaClient:
        raise RuntimeError("synthetic changed-mode node cannot execute a reserved phase")

    engine.supervisor._ollama_factory = reject_direct_execution

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assignments.append((node_id, frame["phase"]))

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = split_job()
    try:
        await prepare(engine, job)
        await engine._execute(job)
        decoder = engine.registry.get("node-b")
        assert decoder is not None
        if unavailable == "lost":
            engine.registry.mark_health("node-b", NodeHealth.LOST)
            await engine.supervisor.on_node_lost("node-b", "synthetic future decoder loss")
        elif unavailable == "draining":
            engine.registry.mark_health("node-b", NodeHealth.DRAINING)
        elif unavailable == "quarantined":
            engine.registry.quarantine("node-b", 60)
        elif unavailable == "removed":
            engine.registry.remove("node-b")
        else:
            update = {"model_whitelist": ("different-model",)} if unavailable == "model" else {
                "execution_mode": ExecutionMode.OLLAMA_DIRECT,
            }
            engine.registry.upsert(decoder.model_copy(update=update), occupancy_reported=False)

        assert job.phase is ExecutionPhase.PREFILL
        assert job.attempt == 0
        assert assignments == [("node-a", "prefill")]
        stream = engine.subscribe(job.job_id)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        await asyncio.sleep(0)
        await complete(engine, "node-a", job, kv_cache_ref=kv_ref)
        assert job.state is JobState.FAILED
        assert await asyncio.wait_for(waiter, 0.5) is None
        assert await asyncio.wait_for(stream.get(), 0.5) is None
        assert assignments == [("node-a", "prefill")]
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
        assert_released(engine)
    finally:
        await engine.stop(drain=False)
    saved = await engine.store.load_job_payload(job.job_id)
    assert saved is not None and saved.state is JobState.FAILED


@pytest.mark.parametrize("transition", ["recovery", "handoff"])
@pytest.mark.parametrize("send_fails", [False, True])
async def test_late_prefill_send_cannot_change_recovered_or_handed_off_reservations(
    config: BrokerConfig, monkeypatch: pytest.MonkeyPatch, transition: str, send_fails: bool,
) -> None:
    engine = split_engine(config)
    sending = asyncio.Event()
    fail_old_send = asyncio.Event()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if node_id == "node-a":
            sending.set()
            await fail_old_send.wait()
            if send_fails:
                raise OSError("synthetic late prefill send failure")

    monkeypatch.setattr(engine.manager, "assign", assign)
    job = split_job()
    await prepare(engine, job)
    dispatch = asyncio.create_task(engine._execute(job))
    try:
        await asyncio.wait_for(sending.wait(), 0.5)
        assert engine.registry.jobs_on("node-b") == (job.job_id,)
        if transition == "recovery":
            engine.registry.mark_health("node-a", NodeHealth.LOST)
            await engine.supervisor.on_node_lost("node-a", "synthetic active prefill loss")
            assert job.attempt == 1
        else:
            await complete(engine, "node-a", job, kv_cache_ref="synthetic-kv")
            assert job.attempt == 0
        assert job.assigned_node_id == "node-b"
        assert engine.registry.jobs_on("node-a") == ()
        assert engine.registry.jobs_on("node-b") == (job.job_id,)

        fail_old_send.set()
        await asyncio.wait_for(dispatch, 0.5)
        assert job.state is JobState.DISPATCHED
        assert engine.registry.jobs_on("node-b") == (job.job_id,)
        await complete(engine, "node-b", job, text="recovered result")
        assert await engine.wait_for_result(job.job_id) == "recovered result"
        assert engine.metrics.counter("horizon_jobs_failed_total") == 0
        assert_released(engine)
    finally:
        fail_old_send.set()
        await asyncio.gather(dispatch, return_exceptions=True)
        await engine.stop(drain=False)
    saved = await engine.store.load_job_payload(job.job_id)
    assert saved is not None and saved.state is JobState.COMPLETED
