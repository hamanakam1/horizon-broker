"""Ingress API tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from horizon_broker.ingress.app import JSON, _job_list_limit, _stream_completion, create_app
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 100),
        ("0", 0),
        ("1001", 1000),
        ("-1", 100),
        ("+1", 100),
        ("1_0", 100),
        ("９", 100),
        ("9" * 21, 100),
        ("nope", 100),
    ],
)
def test_job_list_limit_is_bounded_and_canonical(raw: str, expected: int) -> None:
    assert _job_list_limit(raw) == expected


def test_api_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        JSON({"power_headroom": float("nan")})


async def make_client(config: Any) -> tuple[TestClient, BrokerEngine]:
    """Start an ingress server in front of a broker engine."""
    node = make_node("node-a")
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    app = create_app(engine, config)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, engine


async def test_stream_setup_failure_unsubscribes_the_job(config: Any, monkeypatch) -> None:
    """A disconnect while preparing SSE must not leave a subscriber behind."""
    engine = BrokerEngine(config)
    job = make_job()
    engine._jobs[job.job_id] = job

    class FailingResponse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def prepare(self, _request: Any) -> None:
            raise RuntimeError("synthetic response setup failure")

    monkeypatch.setattr("horizon_broker.ingress.app.web.StreamResponse", FailingResponse)

    with pytest.raises(RuntimeError, match="response setup failure"):
        await _stream_completion(object(), engine, job, timeout=None)

    assert job.job_id not in engine._chunk_subscribers


async def test_initial_stream_write_failure_unsubscribes_the_job(
    config: Any, monkeypatch
) -> None:
    """A failure writing the opening SSE frame also releases the subscriber."""
    engine = BrokerEngine(config)
    job = make_job()
    engine._jobs[job.job_id] = job

    class FailingResponse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def prepare(self, _request: Any) -> None:
            return None

        async def write(self, _payload: bytes) -> None:
            raise RuntimeError("synthetic initial stream write failure")

    monkeypatch.setattr("horizon_broker.ingress.app.web.StreamResponse", FailingResponse)

    with pytest.raises(RuntimeError, match="initial stream write failure"):
        await _stream_completion(object(), engine, job, timeout=None)

    assert job.job_id not in engine._chunk_subscribers


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


async def test_a_non_queued_submission_state_is_rejected(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post(
            "/v1/jobs",
            json={
                "state": "completed",
                "payload": {"model": "qwen2.5:7b", "prompt": "replay only"},
                "max_acceptable_latency_ms": 1_000,
            },
        )
        assert response.status == 422
        body = await response.json()
        assert "must start queued" in body["details"][0]
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


async def test_an_empty_model_identifier_is_rejected(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post("/v1/jobs", json={"payload": {"model": "   "}})
        assert response.status == 422
    finally:
        await client.close()


async def test_a_boolean_latency_budget_is_rejected(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.post(
            "/v1/jobs",
            json={
                "payload": {"model": "qwen2.5:7b", "prompt": "hello"},
                "max_acceptable_latency_ms": True,
            },
        )
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
        rendered = await metrics.text()
        assert "horizon_jobs_submitted_total" in rendered
        assert 'horizon_ledger_fallback_total{reason="query"}' in rendered
        assert "horizon_journal_replay_failures_total 0.0" in rendered
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


@pytest.mark.parametrize("field", [
    "available_vram_bytes",
    "total_vram_bytes",
    "current_batch_size",
    "max_batch_size",
    "tokens_per_second",
    "power_headroom",
])
async def test_a_heartbeat_rejects_boolean_numeric_telemetry(config: Any, field: str) -> None:
    """Boolean JSON values must not be coerced into trusted capacity telemetry."""
    client, engine = await make_client(config)
    try:
        response = await client.post(
            "/v1/nodes/pi-boolean/heartbeat",
            json={
                "available_vram_bytes": 4_000_000_000,
                "total_vram_bytes": 8_000_000_000,
                "max_batch_size": 2,
                field: True,
            },
        )
        assert response.status == 422
        assert engine.registry.get("pi-boolean") is None
    finally:
        await client.close()


@pytest.mark.parametrize("field", ["is_prefill_optimized", "is_decode_optimized"])
async def test_a_heartbeat_rejects_non_boolean_capability_flags(config: Any, field: str) -> None:
    """Capability flags must not accept numeric truthiness from untrusted nodes."""
    client, engine = await make_client(config)
    try:
        response = await client.post(
            "/v1/nodes/pi-flag/heartbeat",
            json={
                "available_vram_bytes": 4_000_000_000,
                "total_vram_bytes": 8_000_000_000,
                "max_batch_size": 2,
                field: 1,
            },
        )
        assert response.status == 422
        assert engine.registry.get("pi-flag") is None
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


async def test_the_job_list_is_exposed(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        for _ in range(2):
            await client.post(
                "/v1/jobs",
                json={
                    "payload": {"model": "qwen2.5:7b", "prompt": "scan"},
                    "max_acceptable_latency_ms": 600_000,
                },
            )
        listing = await client.get("/v1/jobs")
        assert listing.status == 200
        body = await listing.json()
        assert body["total"] == 2
        assert body["count"] == 2
        assert len(body["jobs"]) == 2

        truncated = await client.get("/v1/jobs", params={"limit": "1"})
        body = await truncated.json()
        assert body["total"] == 2
        assert body["count"] == 1

        malformed = await client.get("/v1/jobs", params={"limit": "²"})
        assert malformed.status == 200
        body = await malformed.json()
        assert body["count"] == 2

        empty = await client.get("/v1/jobs", params={"limit": "0"})
        assert empty.status == 200
        body = await empty.json()
        assert body["count"] == 0
    finally:
        await client.close()


async def test_cancelling_a_finished_job_is_a_409(config: Any) -> None:
    """The second DELETE finds the job the first DELETE already cancelled."""
    client, _ = await make_client(config)
    try:
        await client.post(
            "/v1/jobs",
            json={
                "job_id": "cancellable",
                "payload": {"model": "qwen2.5:7b", "prompt": "scan"},
                "max_acceptable_latency_ms": 600_000,
            },
        )
        first = await client.delete("/v1/jobs/cancellable")
        assert first.status == 200
        assert (await first.json())["cancelled"] is True

        second = await client.delete("/v1/jobs/cancellable")
        assert second.status == 409
        assert (await second.json())["error"] == "job already finished"
        assert (await second.json())["state"] == "cancelled"
    finally:
        await client.close()


async def test_cancelling_an_unknown_job_is_a_404(config: Any) -> None:
    client, _ = await make_client(config)
    try:
        response = await client.delete("/v1/jobs/never-existed")
        assert response.status == 404
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
