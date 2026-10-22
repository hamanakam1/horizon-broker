"""Health probe for nodes without a control channel.

A node that runs Ollama directly has no WebSocket to heartbeat over. Without
a probe the watchdog would see silence, declare the node lost and quarantine
it - which is exactly how the M3 Max disappeared from its own mesh.

The probe asks the engine a cheap question (``GET /api/version``) and records
the answer as a heartbeat. A node that stops answering then falls silent in
the registry like any other node, and the watchdog handles it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

from horizon_broker.config import HeartbeatConfig
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.nodes.registry import NodeRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one probe."""

    node_id: str
    healthy: bool
    detail: str = ""


class NodeHealthProbe:
    """Probes the nodes that have no control channel."""

    def __init__(
        self,
        registry: NodeRegistry,
        *,
        config: HeartbeatConfig | None = None,
        session: aiohttp.ClientSession | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.registry = registry
        self.config = config or HeartbeatConfig()
        self._session = session
        self._owns_session = session is None
        self.timeout_seconds = timeout_seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session when this client owns it."""
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None

    def _targets(self) -> list[str]:
        """Nodes that need a probe: direct execution without a control channel."""
        return [
            node.node_id
            for node in self.registry.all()
            if node.execution_mode is ExecutionMode.OLLAMA_DIRECT
            and not node.agent_ws_url
            and bool(node.ollama_url)
        ]

    async def probe_once(self) -> list[ProbeResult]:
        """Probe every direct-mode node once."""
        session = await self._get_session()
        results: list[ProbeResult] = []
        for node_id in self._targets():
            node = self.registry.get(node_id)
            if node is None or not node.ollama_url:
                continue
            url = f"{node.ollama_url.rstrip('/')}/api/version"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        self.registry.record_heartbeat(node_id)
                        results.append(ProbeResult(node_id, True))
                    else:
                        results.append(
                            ProbeResult(node_id, False, f"HTTP {response.status}")
                        )
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                results.append(ProbeResult(node_id, False, str(exc)))
        return results

    async def run(self) -> None:
        """Probe the fleet on the heartbeat interval, forever."""
        while True:
            try:
                await self.probe_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the probe must never die
                LOGGER.warning("health probe failed: %s", exc)
            await asyncio.sleep(self.config.interval_seconds)
