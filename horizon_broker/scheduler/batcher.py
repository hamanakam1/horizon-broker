"""Batch planning.

A Flex batch is a *scheduling* construct: the broker groups prompts that can
share a weight load, then routes each prompt independently (a prefill-heavy
prompt and a decode-heavy prompt inside the same batch may land on different
nodes). Jobs that route to the same node are dispatched together so the node
can amortise the weight load across them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from horizon_broker.contracts.jobs import InferenceJob
from horizon_broker.contracts.nodes import NodeCapability
from horizon_broker.routing.pd_dag import PDDagRouter, RoutingDecision


@dataclass(frozen=True)
class DispatchBatch:
    """Jobs that are dispatched to a single node together."""

    node_id: str
    jobs: tuple[InferenceJob, ...]
    decisions: tuple[RoutingDecision, ...]

    @property
    def size(self) -> int:
        return len(self.jobs)

    def job_ids(self) -> tuple[str, ...]:
        return tuple(job.job_id for job in self.jobs)


class BatchPlanner:
    """Routes jobs and groups the results by target node."""

    def __init__(self, router: PDDagRouter) -> None:
        self.router = router

    def plan(
        self,
        jobs: list[InferenceJob],
        nodes: list[NodeCapability],
        *,
        excluded: frozenset[str] = frozenset(),
        excluded_by_job: Mapping[str, frozenset[str]] | None = None,
    ) -> list[DispatchBatch]:
        """Route every job and group the ones that share a destination node."""
        grouped: dict[str, tuple[list[InferenceJob], list[RoutingDecision]]] = {}
        projected = {node.node_id: node for node in nodes}
        for job in jobs:
            job_excluded = excluded | (excluded_by_job or {}).get(job.job_id, frozenset())
            try:
                decision = self.router.route(job, list(projected.values()), excluded=job_excluded)
            except Exception:
                # Unroutable jobs stay in the queue; the engine reports them.
                continue
            # Each planned start consumes a primary slot. Keep the real fleet
            # untouched; dispatch will check its live state before admission.
            node = projected[decision.primary_node_id]
            projected[node.node_id] = node.model_copy(
                update={
                    "current_batch_size": max(node.current_batch_size, len(node.assigned_job_ids)) + 1
                }
            )
            bucket = grouped.setdefault(decision.primary_node_id, ([], []))
            bucket[0].append(job)
            bucket[1].append(decision)

        batches = [
            DispatchBatch(node_id=node_id, jobs=tuple(entry[0]), decisions=tuple(entry[1]))
            for node_id, entry in grouped.items()
        ]
        batches.sort(key=lambda batch: (-batch.size, batch.node_id))
        return batches
