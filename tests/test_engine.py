"""Engine integration: dispatch, windowing and the fault path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.contracts.errors import (
    BrokerDrainingError,
    QueueFullError,
    UnknownJobError,
)
from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.contracts.nodes import PowerClass
from horizon_broker.contracts.recovery import utcnow
from horizon_broker.scheduler.engine import BrokerEngine
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


def state_of(engine: BrokerEngine, job_id: str) -> str | None:
    status = engine.status(job_id)
    return status["state"] if status else None


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
    windowed = replace(config, window=replace(config.window, idle_flush=False))
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
