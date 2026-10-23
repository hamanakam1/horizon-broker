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
import json
import logging
from dataclasses import dataclass
from typing import NoReturn

import aiohttp

from horizon_broker.config import HeartbeatConfig
from horizon_broker.contracts.nodes import ExecutionMode
from horizon_broker.nodes.registry import NodeRegistry

LOGGER = logging.getLogger(__name__)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")


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
            session = self._session
            try:
                await session.close()
            finally:
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
        return list(
            await asyncio.gather(
                *(self._probe_node(session, node_id) for node_id in self._targets())
            )
        )

    async def _probe_node(self, session: aiohttp.ClientSession, node_id: str) -> ProbeResult:
        """Probe one node independently so a slow peer cannot delay its siblings."""
        node = self.registry.get(node_id)
        if node is None or not node.ollama_url:
            return ProbeResult(node_id, False, "node is no longer registered")
        url = f"{node.ollama_url.rstrip('/')}/api/version"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            ) as response:
                if response.status == 200:
                    try:
                        body = json.loads(
                            await response.text(), parse_constant=_reject_json_constant
                        )
                    except (aiohttp.ClientError, ValueError) as exc:
                        return ProbeResult(node_id, False, f"invalid version response: {exc}")
                    version = body.get("version") if isinstance(body, dict) else None
                    if (
                        isinstance(version, str)
                        and version.strip()
                        and version == version.strip()
                    ):
                        self.registry.record_heartbeat(node_id)
                        return ProbeResult(node_id, True)
                    return ProbeResult(node_id, False, "invalid version response")
                return ProbeResult(node_id, False, f"HTTP {response.status}")
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            detail = "request timed out" if isinstance(exc, TimeoutError) else str(exc)
            return ProbeResult(node_id, False, detail or type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - one probe must not suppress its siblings
            LOGGER.warning("unexpected probe failure for node %s: %s", node_id, exc)
            return ProbeResult(node_id, False, str(exc) or type(exc).__name__)

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
