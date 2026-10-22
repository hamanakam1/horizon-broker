"""Ingress API tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from horizon_broker.ingress.app import create_app
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_node


async def make_client(config: Any) -> tuple[TestClient, BrokerEngine]:
    """Start an ingress server in front of a broker engine."""
    node = make_node("node-a")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    app = create_app(engine, config)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, engine


async def test_a_job_is_accepted(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post(
            "/v1/jobs",
            json={
                "payload": {"model": "qwen2.5:7b", "prompt": "scan the repo"},
                "max_acceptable_latency_ms": 1_000,
            },
        )
        assert response.status == 202
        body = await response.json()
        assert body["priority_tier"] == "interactive"
        assert body["job_id"]

        status = await client.get(f"/v1/jobs/{body['job_id']}")
        assert status.status == 200
        payload = await status.json()
        assert payload["payload"]["model"] == "qwen2.5:7b"
    finally:
        await client.close()


async def test_latency_budget_categorises_the_tier(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post(
            "/v1/jobs",
            json={
                "payload": {"model": "qwen2.5:7b", "prompt": "deep scan"},
                "max_acceptable_latency_ms": 3_600_000,
            },
        )
        assert response.status == 202
        body = await response.json()
        assert body["priority_tier"] == "flex"
    finally:
        await client.close()


async def test_an_explicit_tier_wins(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post(
            "/v1/jobs",
            json={
                "priority_tier": "interactive",
                "payload": {"model": "qwen2.5:7b", "prompt": "deep scan"},
                "max_acceptable_latency_ms": 3_600_000,
            },
        )
        body = await response.json()
        assert body["priority_tier"] == "interactive"
    finally:
        await client.close()


async def test_an_invalid_job_is_rejected(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post("/v1/jobs", json={"payload": {"prompt": "no model"}})
        assert response.status == 422
    finally:
        await client.close()


async def test_an_unknown_job_is_a_404(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.get("/v1/jobs/does-not-exist")
        assert response.status == 404
    finally:
        await client.close()


async def test_the_fleet_and_queue_are_exposed(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        nodes = await client.get("/v1/nodes")
        assert nodes.status == 200
        body = await nodes.json()
        assert body["nodes"][0]["node_id"] == "node-a"

        queue = await client.get("/v1/queue")
        assert queue.status == 200
        assert "queue" in await queue.json()
    finally:
        await client.close()


async def test_health_and_metrics(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        assert (await client.get("/healthz")).status == 200
        assert (await client.get("/readyz")).status == 200
        metrics = await client.get("/metrics")
        assert metrics.status == 200
        assert "horizon_jobs_submitted_total" in await metrics.text()
    finally:
        await client.close()


async def test_a_node_can_announce_itself_over_http(config: Any) -> None:
    client, engine = await make_client(config)
    try:
        response = await client.post(
            "/v1/nodes/pi-01/heartbeat",
            json={
                "role": "decode",
                "available_vram_bytes": 4_000_000_000,
                "total_vram_bytes": 8_000_000_000,
                "max_batch_size": 2,
                "is_decode_optimized": True,
            },
        )
        assert response.status == 200
        node = engine.registry.get("pi-01")
        assert node is not None
        assert node.is_decode_optimized is True
        assert node.last_heartbeat_at is not None
    finally:
        await client.close()


async def test_a_full_queue_returns_429(config: Any) -> None:
    limited = replace(
        config,
        scheduling=replace(config.scheduling, max_queue_depth=1),
    )
    client, _ = await make_client(limited)
    try:
        body = {
            "payload": {"model": "qwen2.5:7b", "prompt": "scan"},
            "max_acceptable_latency_ms": 600_000,
        }
        first = await client.post("/v1/jobs", json=body)
        assert first.status == 202
        second = await client.post("/v1/jobs", json=body)
        assert second.status == 429
        assert (await second.json())["error"] == "queue full"
    finally:
        await client.close()


async def test_a_draining_broker_returns_503(config: Any) -> None:
    client, engine = await make_client(config)
    try:
        await engine.drain(timeout=0.1)
        response = await client.post(
            "/v1/jobs",
            json={
                "payload": {"model": "qwen2.5:7b", "prompt": "scan"},
                "max_acceptable_latency_ms": 600_000,
            },
        )
        assert response.status == 503
        assert (await response.json())["error"] == "broker draining"

        ready = await client.get("/readyz")
        assert ready.status == 503
        assert (await ready.json())["status"] == "draining"
    finally:
        await client.close()


async def test_a_duplicate_job_id_returns_the_original_job(config: Any) -> None:
    client, engine = await make_client(config)
    try:
        body = {
            "job_id": "fixed-job-id",
            "payload": {"model": "qwen2.5:7b", "prompt": "scan"},
            "max_acceptable_latency_ms": 600_000,
        }
        first = await client.post("/v1/jobs", json=body)
        second = await client.post("/v1/jobs", json=body)
        assert first.status == 202
        assert second.status == 202
        assert (await first.json())["job_id"] == (await second.json())["job_id"]
        assert engine.metrics.counter("horizon_jobs_duplicate_total") == 1
    finally:
        await client.close()
