"""Node connection manager.

Owns the WebSocket control channels of every agent-container node. Connections
are created lazily as nodes register and are re-created automatically when a
socket dies (the :class:`~horizon_broker.nodes.connection.NodeConnection`
handles its own reconnect loop).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from horizon_broker.config import HeartbeatConfig
from horizon_broker.contracts.nodes import ExecutionMode, NodeHealth
from horizon_broker.nodes.connection import NodeConnection, NodeEvent
from horizon_broker.nodes.executor import JobSupervisor
from horizon_broker.nodes.registry import NodeRegistry

LOGGER = logging.getLogger(__name__)


class NodeConnectionManager:
    """Keeps one control channel per agent-container node alive."""

    def __init__(
        self,
        registry: NodeRegistry,
        supervisor: JobSupervisor,
        *,
        config: HeartbeatConfig | None = None,
        session: aiohttp.ClientSession | None = None,
        sync_interval: float = 5.0,
    ) -> None:
        self.registry = registry
        self.supervisor = supervisor
        self.config = config or HeartbeatConfig()
        self._session = session
        self._owns_session = session is None
        self._connections: dict[str, NodeConnection] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sync_interval = sync_interval
        self._closing = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def _sync(self) -> None:
        """Create connections for new nodes and drop connections for gone nodes."""
        wanted = {
            node.node_id
            for node in self.registry.all()
            if node.execution_mode is ExecutionMode.AGENT_CONTAINER
            and node.agent_ws_url
            and node.health not in (NodeHealth.LOST, NodeHealth.OFFLINE)
        }
        for node_id in list(self._connections):
            if node_id not in wanted:
                self._drop(node_id)
        for node_id in wanted:
            if node_id in self._connections:
                continue
            node = self.registry.get(node_id)
            if node is None or not node.agent_ws_url:
                continue
            connection = NodeConnection(
                node_id,
                node.agent_ws_url,
                on_event=self.handle_event,
                config=self.config,
                session=self._session,
            )
            self._connections[node_id] = connection
            self._tasks[node_id] = asyncio.create_task(connection.run())
            LOGGER.info("opening control channel to node %s", node_id)

    def _drop(self, node_id: str) -> None:
        task = self._tasks.pop(node_id, None)
        if task is not None:
            task.cancel()
        self._connections.pop(node_id, None)

    async def run(self) -> None:
        """Periodically reconcile the set of connections."""
        while not self._closing:
            try:
                self._sync()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("connection sync failed: %s", exc)
            await asyncio.sleep(self._sync_interval)

    async def assign(self, node_id: str, frame: dict[str, Any]) -> None:
        """Send an assignment frame to a node."""
        connection = self._connections.get(node_id)
        if connection is None:
            raise RuntimeError(f"no control channel for node {node_id}")
        await connection.send(frame)

    async def handle_event(self, event: NodeEvent) -> None:
        """Forward a node frame to the supervisor."""
        await self.supervisor.handle_event(event)

    async def close(self) -> None:
        """Tear down every control channel."""
        self._closing = True
        for node_id in list(self._connections):
            self._drop(node_id)
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None
