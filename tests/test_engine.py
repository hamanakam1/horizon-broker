"""Engine integration: dispatch, windowing and the fault path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.contracts.errors import (
    BrokerDrainingError,
    NoCapacityError,
    QueueFullError,
    UnknownJobError,
)
from horizon_broker.contracts.jobs import JobState, PriorityTier
from horizon_broker.contracts.nodes import ExecutionMode, PowerClass
from horizon_broker.contracts.recovery import utcnow
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.scheduler.engine import BrokerEngine
from horizon_broker.scheduler.window import FlexBatch
from tests.conftest import make_job, make_node, wait_for


async def start_fake_ollama(
    tokens: tuple[str, ...] = ("Hello", " world"),
    delay: float = 0.0,
) -> tuple[TestServer, str]:
    """Start a fake Ollama server that streams ``tokens``."""

    async def stream(request: web.Request, chat: bool) -> web.StreamResponse:
        response = web.StreamResponse()
        response.content_type = "application/x-ndjson"
        await response.prepare(request)
        if delay:
            await asyncio.sleep(delay)
        for token in tokens:
            if chat:
                body = json.dumps(
                    {"message": {"role": "assistant", "content": token}, "done": False}
                )
            else:
                body = json.dumps({"response": token, "done": False})
            await response.write((body + "\n").encode("utf-8"))
        final = json.dumps(
            {"response": "", "done": True, "eval_count": len(tokens), "prompt_eval_count": 8}
        )
        await response.write((final + "\n").encode("utf-8"))
        await response.write_eof()
        return response

    async def generate(request: web.Request) -> web.StreamResponse:
        return await stream(request, chat=False)

    async def chat(request: web.Request) -> web.StreamResponse:
        return await stream(request, chat=True)

    app = web.Application()
    app.router.add_post("/api/generate", generate)
    app.router.add_post("/api/chat", chat)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


async def test_engine_forgets_owned_session_when_close_fails(config: Any) -> None:
    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    engine = BrokerEngine(config)
    engine._session = Session()  # type: ignore[assignment]
    engine._owns_session = True
    engine.manager._session = None
    engine.probe._session = None
    engine.ledger._session = None

    await engine.stop(drain=False)

    assert engine._session is None


def state_of(engine: BrokerEngine, job_id: str) -> str | None:
    status = engine.status(job_id)
    return status["state"] if status else None


async def test_engine_start_failure_closes_partial_runtime(config: Any, monkeypatch) -> None:
    """A failed replay must not leave a broker-owned session or running flag behind."""
    engine = BrokerEngine(config)

    async def fail_replay() -> int:
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(engine, "_replay_journal", fail_replay)

    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        await engine.start()

    assert engine._running is False
    assert engine._session is None
    assert engine._tasks == []


async def test_tracked_execution_task_observes_unexpected_failures(config: Any, caplog) -> None:
    """A child-task failure is logged and removed instead of becoming unobserved asyncio state."""
    engine = BrokerEngine(config)

    async def fail() -> None:
        raise RuntimeError("synthetic tracked-task failure")

    with caplog.at_level("ERROR"):
        engine._track_execution(fail())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert not engine._execution_tasks
    assert "tracked scheduler execution task failed: synthetic tracked-task failure" in caplog.text
    failure_records = [
        record
        for record in caplog.records
        if record.message == "tracked scheduler execution task failed: synthetic tracked-task failure"
    ]
    assert failure_records and failure_records[-1].exc_info is not None


def test_terminal_journal_coroutine_is_closed_without_a_running_loop(config: Any, monkeypatch) -> None:
    """The synchronous completion fallback must not leak an unawaited journal coroutine."""
    engine = BrokerEngine(config)
    job = make_job()

    captured: list[Any] = []

    def fail_create_task(coroutine):
        captured.append(coroutine)
        raise RuntimeError("no running loop")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    engine._record_state_async(job)
    assert not engine._journal_tasks
    assert captured and captured[0].cr_frame is None


async def test_journal_replay_failure_is_counted_and_returns_no_jobs(config: Any, monkeypatch) -> None:
    engine = BrokerEngine(config)

    async def fail_load() -> list[Any]:
        raise OSError("synthetic journal read failure")

    monkeypatch.setattr(engine.store, "load_jobs", fail_load)

    assert await engine._replay_journal() == 0
    assert engine.metrics.counter("horizon_journal_replay_failures_total") == 1


async def test_engine_task_creation_failure_cleans_up_prior_tasks(config: Any, monkeypatch) -> None:
    """A partial scheduler launch must cancel tasks created before the failure."""
    engine = BrokerEngine(config)
    real_create_task = asyncio.create_task
    calls = 0

    def fail_on_second(coro, *, name=None, context=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            coro.close()
            raise RuntimeError("synthetic task creation failure")
        return real_create_task(coro, name=name, context=context)

    monkeypatch.setattr(asyncio, "create_task", fail_on_second)

    with pytest.raises(RuntimeError, match="synthetic task creation failure"):
        await engine.start()

    assert calls == 2
    assert engine._running is False
    assert engine._tasks == []


async def test_engine_start_cancellation_closes_partial_runtime(config: Any, monkeypatch) -> None:
    """Operator cancellation during replay must still unwind the shared runtime."""
    engine = BrokerEngine(config)
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def block_replay() -> int:
        replay_started.set()
        await release_replay.wait()
        return 0

    monkeypatch.setattr(engine, "_replay_journal", block_replay)
    startup = asyncio.create_task(engine.start())
    await replay_started.wait()
    startup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await startup

    assert engine._running is False
    assert engine._session is None
    assert engine._tasks == []


async def test_engine_can_restart_after_a_clean_stop(config: Any) -> None:
    """A stop must not permanently poison a reusable engine instance."""
    engine = BrokerEngine(config)

    await engine.start()
    await engine.stop(drain=False)
    await engine.start()
    try:
        assert engine._running is True
        assert engine._draining is False
    finally:
        await engine.stop(drain=False)


async def test_engine_stop_cancellation_still_closes_runtime(config: Any, monkeypatch) -> None:
    """Cancelling a bounded drain must not leave the broker half-running."""
    engine = BrokerEngine(config)
    await engine.start()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()

    async def block_drain(timeout: float) -> bool:
        drain_started.set()
        await release_drain.wait()
        return True

    monkeypatch.setattr(engine, "drain", block_drain)
    shutdown = asyncio.create_task(engine.stop())
    await drain_started.wait()
    shutdown.cancel()

    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assert engine._running is False
    assert engine._session is None
    assert engine._tasks == []


async def test_engine_stop_attempts_later_cleanup_after_a_close_failure(
    config: Any,
    monkeypatch,
) -> None:
    """One broken client close must not strand the remaining broker resources."""
    engine = BrokerEngine(config)
    await engine.start()
    probe_close = AsyncMock(wraps=engine.probe.close)
    ledger_close = AsyncMock(wraps=engine.ledger.close)
    monkeypatch.setattr(engine.probe, "close", probe_close)
    monkeypatch.setattr(engine.ledger, "close", ledger_close)

    async def fail_manager_close() -> None:
        raise OSError("synthetic manager close failure")

    monkeypatch.setattr(engine.manager, "close", fail_manager_close)

    await engine.stop(drain=False)

    probe_close.assert_called_once_with()
    ledger_close.assert_called_once_with()
    assert engine._running is False
    assert engine._session is None


async def test_ollama_client_cleanup_detaches_cache_and_closes_all_clients(config: Any) -> None:
    engine = BrokerEngine(config)
    closed: list[str] = []

    class Client:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise OSError(f"synthetic close failure: {self.name}")

    engine._ollama_clients = {
        "one": Client("one", fail=True),  # type: ignore[assignment]
        "two": Client("two"),  # type: ignore[assignment]
    }

    with pytest.raises(OSError, match="one"):
        await engine._close_ollama_clients()

    assert closed == ["one", "two"]
    assert engine._ollama_clients == {}


async def test_cancelling_a_job_releases_its_waiter(config: Any) -> None:
    """A cancelled job answers its waiter instead of hanging it.

    The OpenAI ingress waits on that future; an unresolved one leaves the
    HTTP client hanging until its own timeout.
    """
    engine = BrokerEngine(config)
    job = make_job(tier=PriorityTier.INTERACTIVE)
    await engine.submit(job)
    waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
    await asyncio.sleep(0)

    assert await engine.cancel(job.job_id) is True
    assert await asyncio.wait_for(waiter, timeout=1.0) is None
    assert state_of(engine, job.job_id) == "cancelled"


async def test_an_expired_job_releases_its_waiter(config: Any) -> None:
    """A job that ran out of latency budget is a terminal answer, not silence."""
    engine = BrokerEngine(config)
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE, max_latency_ms=1)
        # The budget elapsed before the scheduler ever looked at the job.
        job.deadline_at = utcnow() - timedelta(seconds=1)
        await engine.submit(job)
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        assert await asyncio.wait_for(waiter, timeout=2.0) is None
        assert state_of(engine, job.job_id) == "failed"
    finally:
        await engine.stop(drain=False)


async def test_the_completion_counter_moves(config: Any) -> None:
    """``horizon_jobs_completed_total`` must not be a metric that stays at zero."""
    server, url = await start_fake_ollama()
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: state_of(engine, job.job_id) == "completed")
        assert engine.metrics.counter("horizon_jobs_completed_total") == 1
    finally:
        await engine.stop()
        await server.close()


async def test_the_failure_counter_moves(config: Any) -> None:
    """A job that cannot reach its node ends in a failure the operator can see."""
    node = make_node("node-a", ollama_url="http://127.0.0.1:1")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: state_of(engine, job.job_id) == "failed")
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
    finally:
        await engine.stop(drain=False)


async def test_an_interactive_job_starts_without_waiting_for_the_poll_tick(config: Any) -> None:
    """The queue raises its wakeup event; the lane must not sit on a timer.

    An interactive caller pays for every millisecond of poll latency, and
    the poll interval is a fallback, not a schedule.
    """
    slow_poll = replace(
        config,
        scheduling=replace(config.scheduling, poll_interval_seconds=5.0),
    )
    server, url = await start_fake_ollama()
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(slow_poll, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(
            lambda: state_of(engine, job.job_id) == "completed", timeout=2.0
        )
    finally:
        await engine.stop()
        await server.close()


async def test_an_unknown_job_raises_the_typed_error(config: Any) -> None:
    """The taxonomy is used, not string-matched."""
    engine = BrokerEngine(config)
    with pytest.raises(UnknownJobError):
        engine.require_status("never-existed")


async def test_a_finished_job_cannot_be_cancelled(config: Any) -> None:
    """A completed job must stay completed; DELETE is only a queued/running verb."""
    server, url = await start_fake_ollama()
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: state_of(engine, job.job_id) == "completed")
        assert await engine.cancel(job.job_id) is False
        assert state_of(engine, job.job_id) == "completed"
    finally:
        await engine.stop()
        await server.close()


async def test_a_second_waiter_survives_the_first_timeout(config: Any) -> None:
    """One caller hitting its timeout must not strand the other waiters.

    The first waiter resolves (and times out) while the job is still
    running; the second one still receives the result when the job finishes.
    """
    server, url = await start_fake_ollama(delay=0.3)
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        impatient = asyncio.create_task(engine.wait_for_result(job.job_id, timeout=0.02))
        patient = asyncio.create_task(engine.wait_for_result(job.job_id))
        assert await impatient is None  # times out while the job runs
        assert state_of(engine, job.job_id) in ("queued", "dispatched", "running")
        assert await patient == "Hello world"
        assert state_of(engine, job.job_id) == "completed"
    finally:
        await engine.stop()
        await server.close()


async def test_an_interactive_job_is_dispatched_immediately(config: Any) -> None:
    server, url = await start_fake_ollama()
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: state_of(engine, job.job_id) == "completed")
        assert engine.status(job.job_id)["result_text"] == "Hello world"
    finally:
        await engine.stop()
        await server.close()


async def test_flex_jobs_wait_for_the_window_and_then_run_as_one_batch(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("tick",))
    node = make_node("node-a", ollama_url=url)
    # The idle flush is a different rule with its own test; here the window
    # is the only thing that may release the batch, so the assertion about
    # "windowing" does not race the flush.
    windowed = replace(
        config,
        window=replace(
            config.window,
            window_seconds=1.0,
            min_dwell_seconds=0.2,
            idle_flush=False,
        ),
    )
    engine = BrokerEngine(replace(windowed, static_nodes=(node,)))
    await engine.start()
    try:
        jobs = [make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000) for _ in range(2)]
        for job in jobs:
            await engine.submit(job)

        # Nothing runs before the window closes.
        await asyncio.sleep(0.05)
        assert state_of(engine, jobs[0].job_id) == "windowing"

        for job in jobs:
            assert await wait_for(lambda job_id=job.job_id: state_of(engine, job_id) == "completed")
        assert engine.metrics.counter("horizon_flex_batches_total") >= 1
    finally:
        await engine.stop()
        await server.close()


async def test_direct_nodes_never_receive_a_batch_assignment(config: Any, monkeypatch) -> None:
    """A grouped planner result still uses individual requests on direct Ollama nodes."""
    server, url = await start_fake_ollama(tokens=("direct",))
    node = make_node("direct-node", ollama_url=url)
    windowed = replace(
        config,
        window=replace(
            config.window,
            window_seconds=0.05,
            min_dwell_seconds=0.0,
            idle_flush=False,
        ),
    )
    engine = BrokerEngine(replace(windowed, static_nodes=(node,)))
    start_batch = AsyncMock(side_effect=AssertionError("direct node was batch-assigned"))
    monkeypatch.setattr(engine.supervisor, "start_batch", start_batch)
    await engine.start()
    try:
        jobs = [
            make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000, prompt=f"direct-{index}")
            for index in range(2)
        ]
        for job in jobs:
            await engine.submit(job)
        for job in jobs:
            assert await wait_for(lambda job_id=job.job_id: state_of(engine, job_id) == "completed")
        start_batch.assert_not_awaited()
    finally:
        await engine.stop()
        await server.close()


async def test_a_power_gated_node_holds_flex_work_back(config: Any) -> None:
    server, url = await start_fake_ollama()
    node = make_node(
        "trailer-01",
        ollama_url=url,
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.05,
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000)
        await engine.submit(job)
        await asyncio.sleep(0.3)
        assert state_of(engine, job.job_id) == "windowing"
    finally:
        await engine.stop()
        await server.close()


async def test_a_lost_node_hands_the_job_to_another_node(config: Any) -> None:
    slow_server, slow_url = await start_fake_ollama(tokens=("slow",), delay=5.0)
    fast_server, fast_url = await start_fake_ollama(tokens=("recovered",))
    node_a = make_node("node-a", ollama_url=slow_url)
    node_b = make_node("node-b", ollama_url=fast_url)
    engine = BrokerEngine(replace(config, static_nodes=(node_a, node_b)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: bool(engine.registry.jobs_on("node-a")))

        await engine._on_node_lost("node-a", "simulated power loss")

        assert await wait_for(lambda: state_of(engine, job.job_id) == "completed")
        status = engine.status(job.job_id)
        assert status["result_text"] == "recovered"
        assert status["assigned_node_id"] == "node-b"
        assert engine.metrics.counter("horizon_jobs_recovered_total") >= 1
    finally:
        await engine.stop()
        await slow_server.close()
        await fast_server.close()


async def test_the_queue_reports_its_occupancy(config: Any) -> None:
    node = make_node("node-a")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        await engine.submit(make_job(tier=PriorityTier.FLEX))
        stats = engine.stats()
        assert stats["queue"]["flex"] == 1
        assert stats["nodes"]["total"] == 1
        assert stats["nodes"]["healthy"] == 1
    finally:
        await engine.stop()


async def test_an_idle_fleet_flushes_the_window_early(config: Any) -> None:
    """A single Flex job must not wait for a batch that will never arrive."""
    server, url = await start_fake_ollama(tokens=("tick",))
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000)
        await engine.submit(job)
        assert await wait_for(lambda: state_of(engine, job.job_id) == "completed")
        assert engine.metrics.counter("horizon_flex_batches_total", reason="fleet_idle") >= 1
    finally:
        await engine.stop()
        await server.close()


async def test_a_duplicate_submission_does_not_restart_the_work(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("tick",))
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000)
        first = await engine.submit(job)
        second = await engine.submit(job)
        assert first is second
        assert engine.metrics.counter("horizon_jobs_duplicate_total") == 1
        assert engine.queue.stats().flex <= 1
    finally:
        await engine.stop()
        await server.close()


async def test_the_queue_depth_is_enforced(config: Any) -> None:
    node = make_node("node-a")
    limited = replace(
        config,
        static_nodes=(node,),
        scheduling=replace(config.scheduling, max_queue_depth=1),
    )
    engine = BrokerEngine(limited)
    await engine.start()
    try:
        await engine.submit(make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000))
        with pytest.raises(QueueFullError):
            await engine.submit(make_job(tier=PriorityTier.FLEX, max_latency_ms=600_000))
    finally:
        await engine.stop()


async def test_a_draining_broker_refuses_new_work(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("tick",), delay=0.4)
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        draining = asyncio.create_task(engine.drain(timeout=5.0))
        await asyncio.sleep(0.05)
        assert engine.stats()["draining"] is True
        with pytest.raises(BrokerDrainingError):
            await engine.submit(make_job(tier=PriorityTier.INTERACTIVE))
        assert await draining
    finally:
        await engine.stop()
        await server.close()


async def test_a_drain_waits_for_the_running_job(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("slow",), delay=0.3)
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: bool(engine.registry.jobs_on("node-a")))
        assert await engine.drain(timeout=5.0)
        assert state_of(engine, job.job_id) == "completed"
    finally:
        await engine.stop()
        await server.close()


async def test_a_drain_waits_for_inflight_flex_batches(config: Any) -> None:
    """A drain cannot report success while a Flex dispatch task is active."""
    engine = BrokerEngine(config)
    engine._inflight_flex = 1
    try:
        assert await engine.drain(timeout=0.01) is False
    finally:
        engine._inflight_flex = 0


async def test_a_timed_out_drain_cancels_before_clients_close(config: Any) -> None:
    """A forced shutdown must not leave a generation task using a closed session."""
    server, url = await start_fake_ollama(tokens=("slow",), delay=1.0)
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: bool(engine.registry.jobs_on("node-a")))
        assert not await engine.drain(timeout=0.01)
        assert job.state is JobState.CANCELLED
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on("node-a") == ()
    finally:
        await engine.stop(drain=False)
        await server.close()


async def test_immediate_stop_cancels_active_jobs_before_closing_resources(config: Any) -> None:
    """Non-draining shutdown still releases execution ownership and waiters."""
    server, url = await start_fake_ollama(tokens=("slow",), delay=1.0)
    node = make_node("node-a", ollama_url=url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.INTERACTIVE)
        await engine.submit(job)
        assert await wait_for(lambda: bool(engine.registry.jobs_on("node-a")))
        waiter = asyncio.create_task(engine.wait_for_result(job.job_id))
        await engine.stop(drain=False)
        assert job.state is JobState.CANCELLED
        assert engine.supervisor.active_jobs() == []
        assert engine.registry.jobs_on("node-a") == ()
        assert await asyncio.wait_for(waiter, timeout=1.0) is None
    finally:
        await server.close()


async def test_opt_in_agent_node_receives_one_batch_assignment(config: Any) -> None:
    """Batch-capable agent nodes receive one admission frame per planned batch."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    frames: list[dict[str, Any]] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        frames.append(frame)
        if frame["type"] == "batch_assign":
            for item in frame["jobs"]:
                await engine.supervisor.handle_event(
                    NodeEvent(
                        node_id=node_id,
                        type=NodeEventType.COMPLETE,
                        data={"job_id": item["job"]["job_id"], "text": "batched"},
                    )
                )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"batch-{index}") for index in range(2)]
        for job in jobs:
            await engine.submit(job)
        for job in jobs:
            assert await wait_for(lambda job=job: job.state is JobState.COMPLETED)
        batch_frames = [frame for frame in frames if frame["type"] == "batch_assign"]
        assert len(batch_frames) == 1
        assert len(batch_frames[0]["jobs"]) == 2
        assert engine.metrics.counter("horizon_batches_admitted_total") == 1
        assert engine.metrics.counter("horizon_batch_members_total", outcome="admitted") == 2
        assert engine.metrics.gauge("horizon_batch_active_members") == 0
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)


async def test_batch_admission_failure_rolls_back_every_member(config: Any) -> None:
    """A rejected batch leaves all jobs queued and releases node reservations."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def reject_batch(node_id: str, frame: dict[str, Any]) -> None:
        assert frame["type"] == "batch_assign"
        raise RuntimeError("synthetic batch admission rejection")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine.manager, "assign", reject_batch)
    await engine.start()
    try:
        jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"reject-{index}") for index in range(2)]
        for job in jobs:
            await engine.submit(job)
        assert await wait_for(
            lambda: engine.metrics.counter("horizon_jobs_deferred_total") >= 1
            and all(engine.queue.contains(job.job_id) for job in jobs)
        )
        assert all(job.state in {JobState.QUEUED, JobState.WINDOWING} for job in jobs)
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.supervisor.active_jobs() == []
        assert engine.metrics.counter("horizon_batches_rejected_total") >= 1
        assert engine.metrics.counter("horizon_batch_members_total", outcome="rejected") >= 2
        assert engine.metrics.gauge("horizon_batch_active_members") == 0
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)


async def test_cancelled_batch_admission_rolls_back_every_member(config: Any) -> None:
    """Cancellation during a batch control send releases every provisional reservation."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"cancel-{index}") for index in range(2)]
    decisions = tuple(engine.dispatcher.route(job) for job in jobs)

    async def cancel_batch(node_id: str, frame: dict[str, Any]) -> None:
        raise asyncio.CancelledError

    engine.supervisor._assign = cancel_batch
    with pytest.raises(asyncio.CancelledError):
        await engine.supervisor.start_batch(tuple(jobs), decisions)
    assert all(job.state is JobState.QUEUED for job in jobs)
    assert all(job.assigned_node_id is None for job in jobs)
    assert engine.registry.jobs_on(node.node_id) == ()
    assert engine.supervisor.active_jobs() == []


async def test_batch_duplicate_job_id_does_not_overwrite_active_context(config: Any) -> None:
    """A duplicate batch member is rejected before any reservation changes."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    frames: list[dict[str, Any]] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        frames.append(frame)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        job = make_job(tier=PriorityTier.FLEX, prompt="duplicate-context")
        decision = engine.dispatcher.route(job)
        await engine.supervisor.start(job, decision)
        context = engine.supervisor.context(job.job_id)
        assert context is not None
        with pytest.raises(ValueError, match="already has an active execution"):
            await engine.supervisor.start(job, decision)
        with pytest.raises(ValueError, match="unique, inactive"):
            await engine.supervisor.start_batch((job, job), (context.decision, context.decision))
        assert engine.supervisor.context(job.job_id) is context
        assert engine.registry.jobs_on(node.node_id) == (job.job_id,)
        assert all(frame["type"] != "batch_assign" for frame in frames)
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)


async def test_batch_start_rechecks_total_capacity_before_reserving(config: Any) -> None:
    """A stale batch plan cannot partially reserve a node with new occupancy."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}, "max_batch_size": 2}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    jobs = tuple(make_job(tier=PriorityTier.FLEX, prompt=f"capacity-{index}") for index in range(2))
    decisions = tuple(engine.dispatcher.route(job) for job in jobs)
    current = engine.registry.get(node.node_id)
    assert current is not None
    engine.registry.upsert(current.model_copy(update={"current_batch_size": 1}))
    with pytest.raises(NoCapacityError, match="capacity for 1 of 2"):
        await engine.supervisor.start_batch(jobs, decisions)
    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on(node.node_id) == ()
    assert engine.metrics.counter("horizon_batches_rejected_total") == 1
    assert engine.metrics.counter("horizon_batch_members_total", outcome="rejected") == 2


async def test_batch_start_rejects_an_agent_without_explicit_batch_opt_in(config: Any) -> None:
    """The supervisor enforces the same batch capability contract as the planner."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    jobs = tuple(make_job(tier=PriorityTier.FLEX, prompt=f"opt-in-{index}") for index in range(2))
    decisions = tuple(engine.dispatcher.route(job) for job in jobs)

    with pytest.raises(ValueError, match="agent-container node"):
        await engine.supervisor.start_batch(jobs, decisions)

    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on(node.node_id) == ()


async def test_batch_start_rejects_mixed_models_before_reserving(config: Any) -> None:
    """One assignment frame must never mix model weights or stale decisions."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    jobs = (
        make_job(tier=PriorityTier.FLEX, prompt="model-a"),
        make_job(tier=PriorityTier.FLEX, prompt="model-b"),
    )
    decisions = tuple(engine.dispatcher.route(job) for job in jobs)
    jobs = (
        jobs[0],
        jobs[1].model_copy(
            update={"payload": jobs[1].payload.model_copy(update={"model": "qwen2.5:14b"})}
        ),
    )

    with pytest.raises(ValueError, match="one model"):
        await engine.supervisor.start_batch(jobs, decisions)

    assert engine.supervisor.active_jobs() == []
    assert engine.registry.jobs_on(node.node_id) == ()


async def test_batch_rollback_preserves_member_completed_before_send_failure(config: Any) -> None:
    """A contradictory send failure cannot resurrect a synchronously completed member."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    monkeypatch = pytest.MonkeyPatch()

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assert frame["type"] == "batch_assign"
        first = frame["jobs"][0]["job"]["job_id"]
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": first, "text": "completed-before-rejection"},
            )
        )
        raise RuntimeError("synthetic acknowledgement contradiction")

    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"contradiction-{index}") for index in range(2)]
        decisions = tuple(engine.dispatcher.route(job) for job in jobs)
        with pytest.raises(RuntimeError, match="contradiction"):
            await engine.supervisor.start_batch(tuple(jobs), decisions)
        assert jobs[0].state is JobState.COMPLETED
        assert jobs[0].result_text == "completed-before-rejection"
        assert jobs[1].state is JobState.QUEUED
        assert engine.supervisor.context(jobs[0].job_id) is None
        assert engine.supervisor.context(jobs[1].job_id) is None
        assert engine.registry.jobs_on(node.node_id) == ()
        assert engine.metrics.counter("horizon_batch_members_total", outcome="admitted") == 1
        assert engine.metrics.counter("horizon_batch_members_total", outcome="rejected") == 1
        assert engine.metrics.gauge("horizon_batch_active_members") == 0
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)


async def test_batch_loop_does_not_requeue_a_member_completed_before_send_failure(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine retry loop preserves a terminal member of a failed batch."""
    node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    jobs = tuple(
        make_job(tier=PriorityTier.FLEX, prompt=f"batch-loop-{index}") for index in range(2)
    )
    completed = jobs[0]

    async def fail_batch(_jobs, _decisions) -> None:
        completed.state = JobState.COMPLETED
        completed.result_text = "completed-before-loop-retry"
        raise RuntimeError("synthetic batch transport contradiction")

    async def assign(_node_id: str, _frame: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(engine.manager, "assign", assign)
    monkeypatch.setattr(engine.supervisor, "start_batch", fail_batch)
    engine.queue.submit(jobs[1])
    await engine._run_flex(
        FlexBatch(
            model=jobs[0].payload.model,
            jobs=jobs,
            reason="synthetic",
            opened_at=0.0,
            closed_at=1.0,
        )
    )

    assert completed.state is JobState.COMPLETED
    assert not engine.queue.contains(completed.job_id)
    assert engine.queue.contains(jobs[1].job_id)


async def test_flex_loop_requeues_jobs_when_batch_planning_raises(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected planner failure cannot silently lose a ready batch."""
    engine = BrokerEngine(config)
    jobs = tuple(make_job(tier=PriorityTier.FLEX, prompt=f"planner-{index}") for index in range(2))
    monkeypatch.setattr(
        engine.planner,
        "plan",
        lambda _jobs, _nodes: (_ for _ in ()).throw(RuntimeError("synthetic planner failure")),
    )

    await engine._run_flex(
        FlexBatch(
            model=jobs[0].payload.model,
            jobs=jobs,
            reason="synthetic",
            opened_at=0.0,
            closed_at=1.0,
        )
    )

    assert all(engine.queue.contains(job.job_id) for job in jobs)
    assert all(job.state is JobState.WINDOWING for job in jobs)


async def test_recovery_budget_overrun_does_not_mark_a_dispatched_job_failed(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow successful recovery remains observable as active work."""
    engine = BrokerEngine(config)
    job = make_job()
    job.state = JobState.DISPATCHED
    calls: list[str] = []

    async def recover(_job, _node_id: str, *, reason: str):
        calls.append(reason)
        return type(
            "Outcome",
            (),
            {
                "failure": RuntimeError("synthetic recovery budget exceeded"),
                "dispatched": True,
                "resumed": False,
                "elapsed_seconds": 31.0,
            },
        )()

    monkeypatch.setattr(engine.recovery, "recover", recover)
    outcome = await engine._recover(job, "node-a", "synthetic loss")

    assert calls == ["synthetic loss"]
    assert outcome.dispatched is True
    assert job.state is JobState.DISPATCHED
    assert job.error is None


async def test_recovery_budget_overrun_does_not_mark_a_requeued_job_failed(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed over-budget dispatch remains retryable when requeued."""
    engine = BrokerEngine(config)
    job = make_job()
    job.state = JobState.QUEUED

    async def recover(_job, _node_id: str, *, reason: str):
        assert reason == "synthetic loss"
        return type(
            "Outcome",
            (),
            {
                "failure": RuntimeError("synthetic recovery budget exceeded"),
                "dispatched": False,
                "resumed": False,
                "elapsed_seconds": 31.0,
            },
        )()

    monkeypatch.setattr(engine.recovery, "recover", recover)
    outcome = await engine._recover(job, "node-a", "synthetic loss")

    assert outcome.dispatched is False
    assert job.state is JobState.QUEUED
    assert job.error is None


async def test_direct_rollback_preserves_a_synchronously_completed_job(config: Any) -> None:
    """A direct assignment error cannot resurrect a job completed by its callback."""
    node = make_node("agent-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        assert frame["type"] == "assign"
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": frame["job"]["job_id"], "text": "completed-before-error"},
            )
        )
        raise RuntimeError("synthetic acknowledgement contradiction")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        job = make_job()
        decision = engine.dispatcher.route(job)
        with pytest.raises(RuntimeError, match="contradiction"):
            await engine.supervisor.start(job, decision)
        assert job.state is JobState.COMPLETED
        assert job.result_text == "completed-before-error"
        assert engine.supervisor.context(job.job_id) is None
        assert engine.registry.jobs_on(node.node_id) == ()
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)


async def test_batch_member_recovery_does_not_replay_completed_sibling(config: Any) -> None:
    """A node loss recovers only the batch member that was still active."""
    batch_node = make_node("batch-node", execution_mode=ExecutionMode.AGENT_CONTAINER).model_copy(
        update={"metadata": {"supports_batch": True}}
    )
    fallback = make_node("fallback-node", execution_mode=ExecutionMode.AGENT_CONTAINER)
    engine = BrokerEngine(replace(config, static_nodes=(batch_node, fallback)))
    completed_on_batch: list[str] = []
    assignments: list[tuple[str, str]] = []

    async def assign(node_id: str, frame: dict[str, Any]) -> None:
        if frame["type"] == "batch_assign":
            first_job_id = frame["jobs"][0]["job"]["job_id"]
            completed_on_batch.append(first_job_id)
            await engine.supervisor.handle_event(
                NodeEvent(
                    node_id=node_id,
                    type=NodeEventType.COMPLETE,
                    data={"job_id": first_job_id, "text": "completed-before-loss"},
                )
            )
            return
        job_id = frame["job"]["job_id"]
        assignments.append((node_id, job_id))
        await engine.supervisor.handle_event(
            NodeEvent(
                node_id=node_id,
                type=NodeEventType.COMPLETE,
                data={"job_id": job_id, "text": "recovered-member"},
            )
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine.manager, "assign", assign)
    await engine.start()
    try:
        jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"partial-{index}") for index in range(2)]
        for job in jobs:
            await engine.submit(job)
        assert await wait_for(lambda: len(completed_on_batch) == 1)
        unfinished = next(job for job in jobs if job.job_id not in completed_on_batch)
        await engine.supervisor.on_node_lost(batch_node.node_id, "synthetic batch node loss")
        assert await asyncio.wait_for(engine.wait_for_result(unfinished.job_id), timeout=2.0) == (
            "recovered-member"
        )
        assert jobs[0].result_text == "completed-before-loss" or jobs[1].result_text == (
            "completed-before-loss"
        )
        assert assignments == [(fallback.node_id, unfinished.job_id)]
        assert engine.metrics.counter("horizon_jobs_recovered_total") == 1
    finally:
        monkeypatch.undo()
        await engine.stop(drain=False)
