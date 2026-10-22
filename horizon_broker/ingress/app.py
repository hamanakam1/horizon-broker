"""API ingress.

The ingress is deliberately thin: it validates the envelope, categorises the
request into a scheduling tier and hands the job to the engine. It never
touches the queue internals, the routers or the recovery subsystem.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.errors import BrokerDrainingError, QueueFullError
from horizon_broker.contracts.jobs import InferenceJob, PriorityTier
from horizon_broker.contracts.nodes import NodeCapability
from horizon_broker.scheduler.engine import BrokerEngine

LOGGER = logging.getLogger(__name__)

JSON = web.json_response

ENGINE_KEY: web.AppKey[BrokerEngine] = web.AppKey("engine")
CONFIG_KEY: web.AppKey[BrokerConfig] = web.AppKey("config")


def _engine(request: web.Request) -> BrokerEngine:
    return request.app[ENGINE_KEY]


def _config(request: web.Request) -> BrokerConfig:
    return request.app[CONFIG_KEY]


def job_from_payload(data: dict[str, Any], config: BrokerConfig) -> InferenceJob:
    """Build an :class:`InferenceJob` from an ingress body.

    When the caller does not state a tier, the broker categorises the request
    from its latency budget: anything slower than
    ``scheduling.interactive_latency_threshold_ms`` is sleep-time compute.
    """
    body = dict(data)
    if "priority_tier" not in body:
        latency = body.get("max_acceptable_latency_ms")
        if isinstance(latency, int):
            threshold = config.scheduling.interactive_latency_threshold_ms
            body["priority_tier"] = (
                PriorityTier.INTERACTIVE.value
                if latency <= threshold
                else PriorityTier.FLEX.value
            )
    return InferenceJob.model_validate(body)


async def handle_submit(request: web.Request) -> web.Response:
    """``POST /v1/jobs`` -- accept work."""
    engine = _engine(request)
    config = _config(request)
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return JSON({"error": "body must be valid JSON"}, status=400)
    if not isinstance(data, dict):
        return JSON({"error": "body must be a JSON object"}, status=400)
    try:
        job = job_from_payload(data, config)
    except ValidationError as exc:
        return JSON(
            {"error": "invalid job", "details": json.loads(exc.json())}, status=422
        )
    try:
        await engine.submit(job)
    except QueueFullError as exc:
        LOGGER.warning("refusing job %s: %s", job.job_id, exc)
        return JSON({"error": "queue full", "detail": str(exc)}, status=429)
    except BrokerDrainingError as exc:
        LOGGER.info("refusing job %s: %s", job.job_id, exc)
        return JSON({"error": "broker draining", "detail": str(exc)}, status=503)
    return JSON(
        {
            "job_id": job.job_id,
            "priority_tier": job.priority_tier.value,
            "state": job.state.value,
            "deadline_at": job.deadline_at.isoformat() if job.deadline_at else None,
        },
        status=202,
    )


async def handle_status(request: web.Request) -> web.Response:
    """``GET /v1/jobs/{job_id}`` -- job status."""
    engine = _engine(request)
    job_id = request.match_info["job_id"]
    status = engine.status(job_id)
    if status is None:
        return JSON({"error": "unknown job", "job_id": job_id}, status=404)
    return JSON(status)


async def handle_cancel(request: web.Request) -> web.Response:
    """``DELETE /v1/jobs/{job_id}`` -- cancel queued or running work."""
    engine = _engine(request)
    job_id = request.match_info["job_id"]
    cancelled = await engine.cancel(job_id)
    if not cancelled:
        return JSON({"error": "unknown job", "job_id": job_id}, status=404)
    return JSON({"job_id": job_id, "cancelled": True})


async def handle_nodes(request: web.Request) -> web.Response:
    """``GET /v1/nodes`` -- the fleet view."""
    engine = _engine(request)
    return JSON({"nodes": engine.registry.snapshot()})


async def handle_heartbeat(request: web.Request) -> web.Response:
    """``POST /v1/nodes/{node_id}/heartbeat`` -- HTTP heartbeat fallback."""
    engine = _engine(request)
    node_id = request.match_info["node_id"]
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return JSON({"error": "body must be valid JSON"}, status=400)
    if not isinstance(data, dict):
        return JSON({"error": "body must be a JSON object"}, status=400)
    payload = {**data, "node_id": node_id}
    try:
        capability = NodeCapability.model_validate(payload)
    except ValidationError as exc:
        return JSON(
            {"error": "invalid capability", "details": json.loads(exc.json())}, status=422
        )
    engine.registry.upsert(capability)
    engine.registry.record_heartbeat(node_id)
    return JSON({"node_id": node_id, "accepted": True})


async def handle_queue(request: web.Request) -> web.Response:
    """``GET /v1/queue`` -- queue occupancy."""
    engine = _engine(request)
    return JSON(engine.stats())


async def handle_config(request: web.Request) -> web.Response:
    """``GET /v1/config`` -- the effective configuration."""
    from horizon_broker.config import dump_config

    config = _config(request)
    return web.Response(text=dump_config(config), content_type="application/json")


async def handle_healthz(request: web.Request) -> web.Response:
    """``GET /healthz`` -- liveness."""
    return JSON({"status": "ok"})


async def handle_readyz(request: web.Request) -> web.Response:
    """``GET /readyz`` -- readiness: at least one node is reachable."""
    engine = _engine(request)
    healthy = len(engine.registry.healthy_nodes())
    stats = engine.stats()
    draining = bool(stats.get("draining"))
    ready = healthy > 0 and not draining
    body = {
        "status": "ready" if ready else ("draining" if draining else "degraded"),
        "healthy_nodes": healthy,
        "draining": draining,
    }
    return JSON(body, status=200 if ready else 503)


async def handle_metrics(request: web.Request) -> web.Response:
    """``GET /metrics`` -- Prometheus text exposition."""
    engine = _engine(request)
    return web.Response(text=engine.metrics.render(), content_type="text/plain")


def create_app(engine: BrokerEngine, config: BrokerConfig) -> web.Application:
    """Build the aiohttp application that fronts the broker."""
    app = web.Application()
    app[ENGINE_KEY] = engine
    app[CONFIG_KEY] = config
    app.router.add_post("/v1/jobs", handle_submit)
    app.router.add_get("/v1/jobs/{job_id}", handle_status)
    app.router.add_delete("/v1/jobs/{job_id}", handle_cancel)
    app.router.add_get("/v1/nodes", handle_nodes)
    app.router.add_post("/v1/nodes/{node_id}/heartbeat", handle_heartbeat)
    app.router.add_get("/v1/queue", handle_queue)
    app.router.add_get("/v1/config", handle_config)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/readyz", handle_readyz)
    app.router.add_get("/metrics", handle_metrics)
    return app
