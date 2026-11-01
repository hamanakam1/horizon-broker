"""The health probe for direct-mode nodes.

Regression guard: a node that runs Ollama directly has no control channel to
heartbeat over. Without the probe the watchdog saw silence, declared the node
lost and quarantined it - the M3 Max disappeared from its own mesh.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.contracts.nodes import ExecutionMode, NodeHealth
from horizon_broker.contracts.recovery import utcnow
from horizon_broker.nodes.probe import NodeHealthProbe
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.nodes.watchdog import FaultWatchdog
from tests.conftest import make_node


async def start_engine() -> tuple[TestServer, str]:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"version": "0.5.0"})

    app = web.Application()
    app.router.add_get("/api/version", handler)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


async def test_a_direct_node_is_heartbeated_by_the_probe() -> None:
    server, url = await start_engine()
    registry = NodeRegistry()
    registry.upsert(
        make_node("m3-max", execution_mode=ExecutionMode.OLLAMA_DIRECT, ollama_url=url)
    )
    probe = NodeHealthProbe(registry)
    try:
        results = await probe.probe_once()
        assert [result.healthy for result in results] == [True]
        assert registry.get("m3-max").last_heartbeat_at is not None
    finally:
        await probe.close()
        await server.close()


async def test_a_200_probe_with_invalid_version_body_is_not_healthy() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"unexpected": "body"})

    app = web.Application()
    app.router.add_get("/api/version", handler)
    server = TestServer(app)
    await server.start_server()
    registry = NodeRegistry()
    registry.upsert(
        make_node(
            "m3-max",
            execution_mode=ExecutionMode.OLLAMA_DIRECT,
            ollama_url=str(server.make_url("")),
        )
    )
    probe = NodeHealthProbe(registry)
    try:
        results = await probe.probe_once()
        assert results[0].healthy is False
        assert "invalid version" in results[0].detail
        assert registry.get("m3-max").last_heartbeat_at is None
    finally:
        await probe.close()
        await server.close()


async def test_a_silent_direct_node_is_declared_lost() -> None:
    registry = NodeRegistry()
    registry.upsert(
        make_node(
            "m3-max",
            execution_mode=ExecutionMode.OLLAMA_DIRECT,
            ollama_url="http://127.0.0.1:1",
        )
    )
    probe = NodeHealthProbe(registry)
    lost: list[tuple[str, str]] = []

    async def on_lost(node_id: str, reason: str) -> None:
        lost.append((node_id, reason))

    watchdog = FaultWatchdog(registry, on_node_lost=on_lost)
    try:
        results = await probe.probe_once()
        assert [result.healthy for result in results] == [False]
        # The probe failed, so no fresh heartbeat was recorded. Backdate the
        # last one and the watchdog calls the node lost.
        registry.record_heartbeat("m3-max", utcnow() - timedelta(seconds=30))
        assert await watchdog.scan() == ["m3-max"]
        assert lost and lost[0][0] == "m3-max"
        assert registry.get("m3-max").health is NodeHealth.LOST
    finally:
        await probe.close()


async def test_an_agent_container_is_not_probed() -> None:
    registry = NodeRegistry()
    registry.upsert(
        make_node(
            "trailer-01",
            execution_mode=ExecutionMode.AGENT_CONTAINER,
            agent_ws_url="ws://127.0.0.1:1/broker",
            ollama_url="http://127.0.0.1:1",
        )
    )
    probe = NodeHealthProbe(registry)
    try:
        assert await probe.probe_once() == []
    finally:
        await probe.close()


async def test_a_probed_node_survives_the_watchdog() -> None:
    server, url = await start_engine()
    registry = NodeRegistry()
    registry.upsert(
        make_node("m3-max", execution_mode=ExecutionMode.OLLAMA_DIRECT, ollama_url=url)
    )
    probe = NodeHealthProbe(registry)
    watchdog = FaultWatchdog(registry)
    try:
        await probe.probe_once()
        assert await watchdog.scan() == []
        assert registry.get("m3-max").health is NodeHealth.HEALTHY
    finally:
        await probe.close()
        await server.close()


async def test_a_probe_uses_its_own_timeout_with_a_shared_session() -> None:
    async def slow_handler(request: web.Request) -> web.Response:
        await asyncio.sleep(0.2)
        return web.json_response({"version": "0.5.0"})

    app = web.Application()
    app.router.add_get("/api/version", slow_handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))
    registry = NodeRegistry()
    registry.upsert(
        make_node(
            "m3-max",
            execution_mode=ExecutionMode.OLLAMA_DIRECT,
            ollama_url=str(server.make_url("")),
        )
    )
    probe = NodeHealthProbe(registry, session=session, timeout_seconds=0.01)
    try:
        results = await probe.probe_once()
        assert results[0].healthy is False
        assert "timed out" in results[0].detail.lower()
    finally:
        await probe.close()
        await session.close()
        await server.close()


async def test_probe_forgets_owned_session_when_close_fails() -> None:
    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    registry = NodeRegistry()
    probe = NodeHealthProbe(registry)
    probe._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="session close failure"):
        await probe.close()
    assert probe._session is None
