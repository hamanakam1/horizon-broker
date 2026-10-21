"""Fault-detection watchdog.

The watchdog is the component that notices that hardware died. It polls the
registry for nodes that stopped sending heartbeats and, for every node that
went silent, declares the node LOST, quarantines it and asks the supervisor to
recover the work it was running.

Detection budget (defaults): 2 s heartbeat interval, 6 s silence timeout,
3 missed beats before the node is declared lost. Recovery must complete
within 30 s of that declaration.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from horizon_broker.config import HeartbeatConfig, RecoveryConfig
from horizon_broker.contracts.nodes import NodeHealth
from horizon_broker.nodes.registry import NodeRegistry

LOGGER = logging.getLogger(__name__)

NodeLostCallback = Callable[[str, str], Awaitable[None]]
NodeDrainingCallback = Callable[[str], Awaitable[None]]


def utcnow() -> datetime:
    return datetime.now(UTC)


class FaultWatchdog:
    """Detects node loss and drives the recovery sequence."""

    def __init__(
        self,
        registry: NodeRegistry,
        *,
        config: HeartbeatConfig | None = None,
        recovery_config: RecoveryConfig | None = None,
        on_node_lost: NodeLostCallback | None = None,
        on_node_draining: NodeDrainingCallback | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.registry = registry
        self.config = config or HeartbeatConfig()
        self.recovery_config = recovery_config or RecoveryConfig()
        self._on_node_lost = on_node_lost
        self._on_node_draining = on_node_draining
        self.poll_interval = poll_interval

    async def run(self) -> None:
        """Continuously scan the registry for silent nodes."""
        while True:
            try:
                await self.scan()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the watchdog must never die
                LOGGER.exception("watchdog scan failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def scan(self, now: datetime | None = None) -> list[str]:
        """One detection pass; returns the ids of the nodes declared lost."""
        current = now or utcnow()
        declared: list[str] = []
        for node in self.registry.all():
            if node.health is NodeHealth.LOST or node.health is NodeHealth.OFFLINE:
                continue
            if node.health is NodeHealth.DRAINING:
                if self._on_node_draining is not None:
                    await self._on_node_draining(node.node_id)
                continue

            last_seen = node.last_heartbeat_at or self.registry.registered_at(node.node_id)
            if last_seen is None:
                continue
            silence = (current - last_seen).total_seconds()
            if silence < self.config.timeout_seconds:
                continue

            missed = int(silence // max(self.config.interval_seconds, 0.001))
            reason = (
                f"heartbeat missing for {silence:.1f}s "
                f"({missed} missed beats, threshold {self.config.missed_beats_before_loss})"
            )
            LOGGER.error("node %s declared LOST: %s", node.node_id, reason)
            self.registry.mark_health(node.node_id, NodeHealth.LOST, reason)
            self.registry.quarantine(node.node_id, self.recovery_config.quarantine_seconds)
            declared.append(node.node_id)
            if self._on_node_lost is not None:
                await self._on_node_lost(node.node_id, reason)
        return declared
