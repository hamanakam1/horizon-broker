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
        self._reported_load: dict[str, int] = {}
        self._reported_assignments: dict[str, int] = {}

    # -- registration -------------------------------------------------------

    def upsert(
        self,
        capability: NodeCapability,
        *,
        preserve_reported_batch_size: bool = False,
    ) -> NodeCapability:
        """Register or update a node without confusing partial reports for occupancy.

        ``preserve_reported_batch_size`` is used only when a partial heartbeat
        omits occupancy: such a report says nothing about how busy the node
        is, so the live occupancy stands. A full capability report is the
        node's own count of the work it holds and is authoritative.
        """
        existing = self._nodes.get(capability.node_id)
        if existing is not None:
            # Announcements describe node state, but broker-owned assignments
            # are local authority. A heartbeat's default empty tuple must not
            # erase jobs already reserved by this broker.
            metadata = {**existing.metadata, **capability.metadata}
            if "health_reason" in existing.metadata:
                metadata["health_reason"] = existing.metadata["health_reason"]
            updates: dict[str, Any] = {
                "assigned_job_ids": existing.assigned_job_ids,
                # Health and quarantine are broker verdicts, not node claims.
                "health": existing.health,
                "quarantined_until": existing.quarantined_until,
                "metadata": metadata,
            }
            if preserve_reported_batch_size:
                # The report says nothing about occupancy: keep the live count.
                updates["current_batch_size"] = existing.current_batch_size
            else:
                # The node reports what it holds. Our own assignments are
                # part of that report from now on, so the pending ones - the
                # jobs placed since - are the only ones it may not know about.
                self._reported_load[capability.node_id] = capability.current_batch_size
                self._reported_assignments[capability.node_id] = len(
                    existing.assigned_job_ids
                )
                updates["current_batch_size"] = self._occupancy(
                    capability.node_id, len(existing.assigned_job_ids)
                )
            if capability.last_heartbeat_at is None:
                updates["last_heartbeat_at"] = existing.last_heartbeat_at
            capability = capability.model_copy(update=updates)
        else:
            self._reported_load[capability.node_id] = capability.current_batch_size
            self._reported_assignments[capability.node_id] = 0
        self._nodes[capability.node_id] = capability
        self._registered_at.setdefault(capability.node_id, utcnow())
        return capability
        self._nodes[capability.node_id] = capability
        self._registered_at.setdefault(capability.node_id, utcnow())
        return capability

    def remove(self, node_id: str) -> None:
        """Forget a node entirely."""
        self._nodes.pop(node_id, None)
        self._registered_at.pop(node_id, None)
        self._reported_load.pop(node_id, None)
        self._reported_assignments.pop(node_id, None)

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
        if at is not None and (at.tzinfo is None or at.utcoffset() is None):
            raise ValueError("heartbeat timestamp must be timezone-aware")
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
        if now is not None and (now.tzinfo is None or now.utcoffset() is None):
            raise ValueError("quarantine timestamp must be timezone-aware")
        node = self._nodes.get(node_id)
        if node is None or node.quarantined_until is None:
            return False
        return node.quarantined_until > (now or utcnow())

    # -- work accounting ----------------------------------------------------

    def _occupancy(self, node_id: str, assigned_count: int) -> int:
        """The live occupancy of a node.

        It is the node's last reported load plus the jobs this broker placed
        since that report (the node may not know about them yet), and never
        less than the jobs this broker owns. The broker therefore never books
        a node beyond what it can take, and a report corrects the count.
        """
        reported = self._reported_load.get(node_id, 0)
        reported_assignments = self._reported_assignments.get(node_id, 0)
        pending = max(0, assigned_count - reported_assignments)
        return max(reported + pending, assigned_count)

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
                # The node's last report may not know about this broker's work
                # yet, so the live occupancy grows with the assignment: the
                # broker must never book a node beyond what it can take.
                "current_batch_size": self._occupancy(node_id, len(assigned)),
            }
        )

    def release_job(self, node_id: str, job_id: str) -> None:
        """Detach a job from a node."""
        node = self._nodes.get(node_id)
        if node is None or job_id not in node.assigned_job_ids:
            # Late or duplicate terminal events must not erase occupancy that
            # belongs to the node or to another broker's work.
            return
        assigned = tuple(item for item in node.assigned_job_ids if item != job_id)
        self._nodes[node_id] = node.model_copy(
            update={
                "assigned_job_ids": assigned,
                # The slot this job held is freed, but never below the jobs
                # this broker still owns: a late or duplicate release must not
                # erase occupancy that belongs to the node or to another
                # broker's work.
                "current_batch_size": self._occupancy(node_id, len(assigned)),
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
