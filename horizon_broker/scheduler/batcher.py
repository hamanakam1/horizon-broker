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

from horizon_broker.contracts.errors import NoCapacityError
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
        grouped: dict[tuple[str, str], tuple[list[InferenceJob], list[RoutingDecision]]] = {}
        # Planning projects the occupancy: every planned start consumes a slot,
        # so the router sees the node it would actually face. The real fleet
        # stays untouched, and dispatch re-checks the live state before it
        # admits the job.
        projected = {node.node_id: node for node in nodes}
        for job in jobs:
            job_excluded = excluded | (excluded_by_job or {}).get(job.job_id, frozenset())
            try:
                decision = self.router.route(job, list(projected.values()), excluded=job_excluded)
            except NoCapacityError:
                # Unroutable jobs stay in the queue; the engine reports them.
                continue
            planned_node = projected[decision.primary_node_id]
            projected[planned_node.node_id] = planned_node.model_copy(
                update={
                    "current_batch_size": max(
                        planned_node.current_batch_size, len(planned_node.assigned_job_ids)
                    )
                    + 1
                }
            )
            # A batch shares the model-weight admission on the target node.
            # Keep this invariant in the planner as well as the supervisor so
            # stale or direct planner callers cannot manufacture a mixed-model
            # assignment that is rejected only after routing.
            key = (decision.primary_node_id, job.payload.model)
            bucket = grouped.setdefault(key, ([], []))
            bucket[0].append(job)
            bucket[1].append(decision)

        batches: list[DispatchBatch] = []
        for (node_id, _model), entry in grouped.items():
            node = next((candidate for candidate in nodes if candidate.node_id == node_id), None)
            if node is None:
                continue
            # A node's advertised limit includes work already assigned to it.
            # Planning one oversized group would make the registry truthful but
            # still over-admit work when the engine starts each job in turn.
            capacity = node.batch_headroom
            if capacity <= 0:
                continue
            jobs_for_node, decisions_for_node = entry
            for offset in range(0, len(jobs_for_node), capacity):
                batches.append(
                    DispatchBatch(
                        node_id=node_id,
                        jobs=tuple(jobs_for_node[offset : offset + capacity]),
                        decisions=tuple(decisions_for_node[offset : offset + capacity]),
                    )
                )
        batches.sort(key=lambda batch: (-batch.size, batch.node_id))
        return batches
