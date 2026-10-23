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
        self._pending_job_ids: dict[str, set[str]] = {}
        self._reported_load: dict[str, int] = {}
        self._reported_running: dict[str, int] = {}
        self._reported_running_jobs: dict[str, set[str]] = {}
        self._released_covered: dict[str, int] = {}

    # -- registration -------------------------------------------------------

    def upsert(
        self,
        capability: NodeCapability,
        *,
        occupancy_reported: bool = True,
    ) -> NodeCapability:
        """Register or update a node without confusing partial reports for occupancy.

        ``occupancy_reported`` says whether the report describes how busy the
        node is. A full capability report does; a partial heartbeat that
        omits the occupancy does not, and then the live occupancy stands.
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
            if not occupancy_reported:
                # The report says nothing about occupancy: keep the live count.
                updates["current_batch_size"] = existing.current_batch_size
            else:
                # The node reports the work it is running. The slots this
                # broker reserved for a phase that has not started yet are
                # invisible to it, so they are added to the report.
                pending = len(self._pending_job_ids.get(capability.node_id, ()))
                updates["current_batch_size"] = self._occupancy(
                    capability.node_id,
                    len(existing.assigned_job_ids),
                    reported=capability.current_batch_size,
                )
                self._reported_load[capability.node_id] = capability.current_batch_size
                self._reported_running[capability.node_id] = (
                    len(existing.assigned_job_ids) - pending
                )
                self._reported_running_jobs[capability.node_id] = {
                    job_id
                    for job_id in existing.assigned_job_ids
                    if job_id not in self._pending_job_ids.get(capability.node_id, set())
                }
                self._released_covered[capability.node_id] = 0
            if capability.last_heartbeat_at is None:
                updates["last_heartbeat_at"] = existing.last_heartbeat_at
            capability = capability.model_copy(update=updates)
        else:
            # A new node: its first report is the baseline.
            self._reported_load[capability.node_id] = capability.current_batch_size
            self._reported_running[capability.node_id] = 0
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
        self._pending_job_ids.pop(node_id, None)
        self._reported_load.pop(node_id, None)
        self._reported_running.pop(node_id, None)
        self._reported_running_jobs.pop(node_id, None)
        self._released_covered.pop(node_id, None)

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

    def _occupancy(
        self,
        node_id: str,
        assigned_count: int,
        *,
        reported: int | None = None,
    ) -> int:
        """The live occupancy of a node.

        A node reports the work it is *running*, and that report already
        counts this broker's running jobs. Two things sit on top of it: the
        jobs this broker started after the report (the node does not know
        about them yet) and the slots it reserved for a phase that has not
        started (the node cannot count those at all). The broker's own jobs
        are the floor: it never books a node beyond what it can take.
        """
        pending = len(self._pending_job_ids.get(node_id, ()))
        if reported is not None:
            # A fresh report covers every running job of this broker.
            return max(assigned_count, reported + pending)
        running = assigned_count - pending
        unreported = max(0, running - self._reported_running.get(node_id, 0))
        # The jobs this broker finished are gone from the node too: it has
        # simply not said so yet. Its report is corrected here rather than
        # waiting for the next heartbeat.
        finished = self._released_covered.get(node_id, 0)
        return max(
            assigned_count,
            self._reported_load.get(node_id, 0) - finished + unreported + pending,
        )

    def reserve_job(self, node_id: str, job_id: str) -> None:
        """Hold a slot for a phase of this execution that has not started yet.

        A reserved slot is an assignment the node may not know about yet, so
        it is kept apart from the node's reported running work. Activating
        the phase does not charge the slot a second time.
        """
        node = self._nodes.get(node_id)
        if node is None or job_id in node.assigned_job_ids:
            return
        self.assign_job(node_id, job_id)
        self._pending_job_ids.setdefault(node_id, set()).add(job_id)

    def _clear_pending(self, node_id: str, job_id: str) -> None:
        """Forget a reservation: the phase started, or the slot was released."""
        pending = self._pending_job_ids.get(node_id)
        if pending is not None:
            pending.discard(job_id)
            if not pending:
                self._pending_job_ids.pop(node_id, None)

    def assign_job(self, node_id: str, job_id: str) -> None:
        """Activate work without charging a previously reserved slot twice."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        self._clear_pending(node_id, job_id)
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
        self._clear_pending(node_id, job_id)
        node = self._nodes.get(node_id)
        if node is None or job_id not in node.assigned_job_ids:
            # Late or duplicate terminal events must not erase occupancy that
            # belongs to the node or to another broker's work.
            return
        covered = self._reported_running_jobs.get(node_id, set())
        if job_id in covered:
            # The node's last report counted this job; it is finished, so the
            # reported load is one too high until the next report says so.
            self._released_covered[node_id] = self._released_covered.get(node_id, 0) + 1
            covered.discard(job_id)
        assigned = tuple(item for item in node.assigned_job_ids if item != job_id)
        self._nodes[node_id] = node.model_copy(
            update={
                "assigned_job_ids": assigned,
                # The slot this job held is freed, but never below the jobs
                # this broker still owns: a late or duplicate release must not
                # erase occupancy that belongs to the node or to another
                # broker's work. A job the node has already reported is the
                # node's to give back - it frees the slot when its next
                # report arrives.
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
