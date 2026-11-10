"""HTTP server lifecycle rollback."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from horizon_broker.ingress.server import BrokerServer
from horizon_broker.scheduler.engine import BrokerEngine


async def test_server_start_rolls_back_when_port_binding_fails(config, monkeypatch) -> None:
    engine = BrokerEngine(config)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    server = BrokerServer(engine, config)

    async def fail_to_bind(site) -> None:
        raise OSError("synthetic bind failure")

    monkeypatch.setattr(web.TCPSite, "start", fail_to_bind)

    with pytest.raises(OSError, match="synthetic bind failure"):
        await server.start()

    engine.start.assert_awaited_once_with()
    engine.stop.assert_awaited_once_with(drain=False)
    assert server._runner is None
    assert server._site is None


async def test_server_start_rolls_back_when_engine_start_fails(config) -> None:
    engine = BrokerEngine(config)
    engine.start = AsyncMock(side_effect=RuntimeError("synthetic engine failure"))
    engine.stop = AsyncMock()
    server = BrokerServer(engine, config)

    with pytest.raises(RuntimeError, match="synthetic engine failure"):
        await server.start()

    engine.stop.assert_awaited_once_with(drain=False)
    assert server._runner is None
    assert server._site is None


async def test_server_start_stops_engine_when_runner_cleanup_fails(config, monkeypatch) -> None:
    engine = BrokerEngine(config)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    server = BrokerServer(engine, config)

    async def fail_to_bind(site) -> None:
        raise OSError("synthetic bind failure")

    async def fail_to_cleanup(runner) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(web.TCPSite, "start", fail_to_bind)
    monkeypatch.setattr(web.AppRunner, "cleanup", fail_to_cleanup)

    with pytest.raises(OSError, match="synthetic bind failure"):
        await server.start()

    engine.stop.assert_awaited_once_with(drain=False)
    assert server._runner is None
    assert server._site is None


async def test_server_stop_finishes_engine_shutdown_when_site_stop_fails(
    config,
    monkeypatch,
) -> None:
    engine = BrokerEngine(config)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    server = BrokerServer(engine, config)
    await server.start()

    async def fail_to_stop(site) -> None:
        raise OSError("synthetic site stop failure")

    monkeypatch.setattr(web.TCPSite, "stop", fail_to_stop)

    with pytest.raises(OSError, match="synthetic site stop failure"):
        await server.stop(drain=False)

    engine.stop.assert_awaited_once_with(drain=False)
    assert server._runner is None
    assert server._site is None
