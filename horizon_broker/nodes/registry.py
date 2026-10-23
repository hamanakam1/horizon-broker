"""Node registry.

The broker's authoritative view of the mesh. It holds the current
:class:`NodeCapability` of every known node, the jobs in flight on each node
and the health verdict produced by the heartbeat watchdog.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from horizon_broker.contracts.nodes import NodeCapability, NodeHealth


def utcnow() -> datetime:
    return datetime.now(UTC)


class NodeRegistry:
    """In-memory registry of node capabilities."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeCapability] = {}
        self._registered_at: dict[str, datetime] = {}

    # -- registration -------------------------------------------------------

    def upsert(self, capability: NodeCapability) -> NodeCapability:
        """Update telemetry without replacing existing broker-owned assignments.

        Full heartbeat payloads may omit job IDs. Only assignment and release
        operations may change the broker's ownership of an existing node's jobs.
        """
        existing = self._nodes.get(capability.node_id)
        if existing is not None:
            capability = capability.model_copy(
                update={
                    "assigned_job_ids": existing.assigned_job_ids,
                    "last_heartbeat_at": capability.last_heartbeat_at or existing.last_heartbeat_at,
                }
            )
        # Telemetry can lag assignment; known active jobs still occupy slots.
        if capability.current_batch_size < len(capability.assigned_job_ids):
            capability = capability.model_copy(
                update={"current_batch_size": len(capability.assigned_job_ids)}
            )
        self._nodes[capability.node_id] = capability
        self._registered_at.setdefault(capability.node_id, utcnow())
        return capability

    def remove(self, node_id: str) -> None:
        """Forget a node entirely."""
        self._nodes.pop(node_id, None)
        self._registered_at.pop(node_id, None)

    def get(self, node_id: str) -> NodeCapability | None:
        return self._nodes.get(node_id)

    def all(self) -> list[NodeCapability]:
        return list(self._nodes.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._nodes.keys())

    def registered_at(self, node_id: str) -> datetime | None:
        return self._registered_at.get(node_id)

    # -- health -------------------------------------------------------------

    def mark_health(self, node_id: str, health: NodeHealth, reason: str | None = None) -> None:
        """Set the health verdict of a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        metadata = dict(node.metadata)
        if reason:
            metadata["health_reason"] = reason
        self._nodes[node_id] = node.model_copy(update={"health": health, "metadata": metadata})

    def record_heartbeat(self, node_id: str, at: datetime | None = None) -> None:
        """Record a heartbeat from a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        self._nodes[node_id] = node.model_copy(update={"last_heartbeat_at": at or utcnow()})

    def quarantine(self, node_id: str, seconds: float) -> None:
        """Exclude a node from placement for ``seconds``."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        self._nodes[node_id] = node.model_copy(
            update={"quarantined_until": utcnow() + timedelta(seconds=seconds)}
        )

    def clear_quarantine(self, node_id: str) -> None:
        """Lift the quarantine; the node may take work again."""
        node = self._nodes.get(node_id)
        if node is None or node.quarantined_until is None:
            return
        self._nodes[node_id] = node.model_copy(update={"quarantined_until": None})

    def is_quarantined(self, node_id: str, now: datetime | None = None) -> bool:
        """True while the node is excluded from placement."""
        node = self._nodes.get(node_id)
        if node is None or node.quarantined_until is None:
            return False
        return node.quarantined_until > (now or utcnow())

    # -- work accounting ----------------------------------------------------

    def assign_job(self, node_id: str, job_id: str) -> None:
        """Attach a job to a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        if job_id in node.assigned_job_ids:
            return
        assigned = node.assigned_job_ids + (job_id,)
        self._nodes[node_id] = node.model_copy(
            update={
                "assigned_job_ids": assigned,
                "current_batch_size": max(node.current_batch_size + 1, len(assigned)),
            }
        )

    def release_job(self, node_id: str, job_id: str) -> None:
        """Detach a job from a node."""
        node = self._nodes.get(node_id)
        if node is None or job_id not in node.assigned_job_ids:
            return
        assigned = tuple(item for item in node.assigned_job_ids if item != job_id)
        self._nodes[node_id] = node.model_copy(
            update={
                "assigned_job_ids": assigned,
                "current_batch_size": max(len(assigned), node.current_batch_size - 1),
            }
        )

    def jobs_on(self, node_id: str) -> tuple[str, ...]:
        node = self._nodes.get(node_id)
        return node.assigned_job_ids if node else ()

    # -- introspection ------------------------------------------------------

    def healthy_nodes(self) -> list[NodeCapability]:
        """Nodes that are not lost/offline."""
        return [
            node
            for node in self._nodes.values()
            if node.health not in (NodeHealth.LOST, NodeHealth.OFFLINE)
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        """JSON-serialisable view of the mesh."""
        return [node.model_dump(mode="json") for node in self._nodes.values()]
