"""A flex batch must reach the engine as a wave, not as a queue."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node, wait_for

MODEL = "synthetic-model"


class WaveRecorder:
    """Records how many requests the engine had in flight at once.

    Every request is held briefly, so the peak tells the truth about
    whether the broker sent the batch as a wave or one at a time.
    """

    def __init__(self, hold_seconds: float = 0.05) -> None:
        self.hold_seconds = hold_seconds
        self.in_flight = 0
        self.peak = 0
        self.requests = 0

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        response = web.StreamResponse()
        response.content_type = "application/x-ndjson"
        await response.prepare(request)
        try:
            await asyncio.sleep(self.hold_seconds)
            body = json.dumps({"response": "tick", "done": False})
            await response.write((body + "\n").encode("utf-8"))
            final = json.dumps({"response": "", "done": True, "eval_count": 1})
            await response.write((final + "\n").encode("utf-8"))
            await response.write_eof()
        finally:
            self.in_flight -= 1
        return response


def state_of(engine: BrokerEngine, job_id: str) -> str | None:
    status = engine.status(job_id)
    return status["state"] if status else None


async def test_a_flex_batch_reaches_the_engine_as_one_wave(config: BrokerConfig) -> None:
    """The jobs of one group must overlap.

    A batch that runs one prompt after another buys the fleet nothing that
    single prompts would not buy too: the weight load is only amortised
    when the requests are in flight together.
    """
    recorder = WaveRecorder()

    async def generate(request: web.Request) -> web.StreamResponse:
        return await recorder.handle(request)

    app = web.Application()
    app.router.add_post("/api/generate", generate)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url("")).rstrip("/")

    node = make_node("node-a", ollama_url=url, max_batch=4, models=(MODEL,))
    engine = BrokerEngine(
        replace(
            config,
            static_nodes=(node,),
            window=replace(
                config.window,
                min_batch_size=2,
                max_batch_size=4,
                window_seconds=30,
                min_dwell_seconds=0,
                idle_flush=False,
            ),
        )
    )
    await engine.start()
    try:
        # Two jobs are enough to show the wave: the third would wait in the
        # window for the next one, not join this batch.
        jobs = [
            make_job(tier=PriorityTier.FLEX, model=MODEL, max_latency_ms=60_000)
            for _ in range(2)
        ]
        for job in jobs:
            await engine.submit(job)
        for job in jobs:
            assert await wait_for(
                lambda job_id=job.job_id: state_of(engine, job_id) == "completed",
                timeout=15.0,
            )
        # The wave size is timing-dependent (the window may flush before the
        # last job arrives); the overlap is the property that matters.
        assert recorder.requests >= 2
        assert recorder.peak >= 2, (
            "the batch did not overlap: the weight load is not amortised across it"
        )
    finally:
        await engine.stop(drain=False)
        await server.close()
