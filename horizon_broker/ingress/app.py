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
from horizon_broker.ingress.openai import (
    DONE,
    OpenAITranslationError,
    chat_request_to_job,
    chunk_object,
    completion_object,
    sse,
)
from horizon_broker.scheduler.engine import TERMINAL_STATES, BrokerEngine

LOGGER = logging.getLogger(__name__)

JSON = web.json_response

TERMINAL_STATE_NAMES = {state.value for state in TERMINAL_STATES}
MAX_JOB_LIST_LIMIT = 1_000
DEFAULT_JOB_LIST_LIMIT = 100

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


async def handle_jobs(request: web.Request) -> web.Response:
    """``GET /v1/jobs`` -- the tracked jobs, newest first.

    An operator greps this for stuck work; ``limit`` keeps one response
    small instead of shipping the whole (bounded) store.
    """
    engine = _engine(request)
    raw = request.query.get("limit", "")
    limit = DEFAULT_JOB_LIST_LIMIT
    if raw.isdigit():
        limit = min(int(raw), MAX_JOB_LIST_LIMIT)
    return JSON(engine.list_jobs(limit))


async def handle_cancel(request: web.Request) -> web.Response:
    """``DELETE /v1/jobs/{job_id}`` -- cancel queued or running work.

    A finished job is never rewritten: the caller gets a 409 so it can tell
    "lost the race with completion" apart from "never existed" (404).
    """
    engine = _engine(request)
    job_id = request.match_info["job_id"]
    status = engine.status(job_id)
    if status is None:
        return JSON({"error": "unknown job", "job_id": job_id}, status=404)
    if status["state"] in TERMINAL_STATE_NAMES:
        return JSON(
            {"error": "job already finished", "state": status["state"], "job_id": job_id},
            status=409,
        )
    cancelled = await engine.cancel(job_id)
    if not cancelled:
        return JSON({"error": "not cancellable", "job_id": job_id}, status=409)
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


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """``POST /v1/chat/completions`` -- the OpenAI-compatible ingress.

    An agent can point its OpenAI base URL at the broker: the request is
    queued like any other work and the answer comes back in the OpenAI
    shape, streamed or not.
    """
    engine = _engine(request)
    config = _config(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSON({"error": {"message": "body must be valid JSON"}}, status=400)
    try:
        job = chat_request_to_job(body)
    except OpenAITranslationError as exc:
        return JSON({"error": {"message": str(exc)}}, status=422)

    timeout = body.get("horizon_timeout_seconds")
    wait_for = float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else None
    try:
        await engine.submit(job)
    except QueueFullError as exc:
        return JSON({"error": {"message": str(exc), "type": "queue_full"}}, status=429)
    except BrokerDrainingError as exc:
        return JSON({"error": {"message": str(exc), "type": "broker_draining"}}, status=503)

    text = await engine.wait_for_result(job.job_id, timeout=wait_for)
    if text is None:
        status = engine.status(job.job_id) or {}
        return JSON(
            {
                "error": {
                    "message": status.get("error") or "the job did not produce a result",
                    "type": "job_failed",
                    "job_id": job.job_id,
                }
            },
            status=504 if status.get("state") not in ("failed", "cancelled") else 500,
        )

    prompt_tokens = job.payload.estimated_prompt_tokens
    completion_tokens = max(1, len(text) // 4)
    measured = engine.token_usage(job.job_id)
    if measured is not None:
        # The engine counted the tokens; an estimate would misreport the cost.
        prompt_tokens, completion_tokens = measured
    if body.get("stream"):
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        await response.write(sse(chunk_object(job, delta={"role": "assistant", "content": ""})))
        await response.write(sse(chunk_object(job, delta={"content": text})))
        await response.write(sse(chunk_object(job, delta={}, finish_reason="stop")))
        await response.write(DONE)
        await response.write_eof()
        return response

    payload = completion_object(
        job,
        text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if config.log_level.upper() == "DEBUG":
        LOGGER.debug("openai completion for job %s", job.job_id)
    return JSON(payload)


async def handle_models(request: web.Request) -> web.Response:
    """``GET /v1/models`` -- the models the mesh can serve."""
    engine = _engine(request)
    models: set[str] = set()
    for node in engine.registry.healthy_nodes():
        models.update(node.model_whitelist)
    return JSON(
        {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "horizon-broker"} for model in sorted(models)],
        }
    )


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
    app.router.add_get("/v1/jobs", handle_jobs)
    app.router.add_get("/v1/jobs/{job_id}", handle_status)
    app.router.add_delete("/v1/jobs/{job_id}", handle_cancel)
    app.router.add_get("/v1/nodes", handle_nodes)
    app.router.add_post("/v1/nodes/{node_id}/heartbeat", handle_heartbeat)
    app.router.add_get("/v1/queue", handle_queue)
    app.router.add_get("/v1/config", handle_config)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/readyz", handle_readyz)
    app.router.add_get("/metrics", handle_metrics)
    return app
