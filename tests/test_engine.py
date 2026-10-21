"""Engine integration: dispatch, windowing and the fault path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.contracts.nodes import PowerClass
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for


async def start_fake_ollama(
    tokens: tuple[str, ...] = ("Hello", " world"),
    delay: float = 0.0,
) -> tuple[TestServer, str]:
    """Start a fake Ollama server that streams ``tokens``."""

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        response.content_type = "application/x-ndjson"
        await response.prepare(request)
        if delay:
            await asyncio.sleep(delay)
        for token in tokens:
            body = json.dumps({"response": token, "done": False}) + "\n"
            await response.write(body.encode("utf-8"))
        final = json.dumps({"response": "", "done": True, "eval_count": len(tokens)}) + "\n"
        await response.write(final.encode("utf-8"))
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/api/generate", handler)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


def state_of(engine: BrokerEngine, job_id: str) -> str | None:
    status = engine.status(job_id)
    return status["state"] if status else None


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
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
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
