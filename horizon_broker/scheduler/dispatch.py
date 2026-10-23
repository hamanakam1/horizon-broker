"""Dispatch layer.

Sits between the scheduler and the supervisor: it asks the PD-DAG router where
a job should run and hands the job to the supervisor for execution.
"""

from __future__ import annotations

import logging

from horizon_broker.contracts.jobs import InferenceJob
from horizon_broker.contracts.nodes import NodeCapability
from horizon_broker.nodes.executor import JobSupervisor
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.routing.pd_dag import PDDagRouter, RoutingDecision

LOGGER = logging.getLogger(__name__)


class Dispatcher:
    """Routes a job and starts it on the selected node."""

    def __init__(
        self,
        registry: NodeRegistry,
        router: PDDagRouter,
        supervisor: JobSupervisor,
    ) -> None:
        self.registry = registry
        self.router = router
        self.supervisor = supervisor

    def route(
        self,
        job: InferenceJob,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> RoutingDecision:
        """Pick a node for ``job``.

        Raises :class:`NoCapacityError` when the mesh cannot serve the job.
        """
        nodes: list[NodeCapability] = self.registry.all()
        return self.router.route(job, nodes, excluded=excluded)

    async def dispatch(
        self,
        job: InferenceJob,
        decision: RoutingDecision | None = None,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> RoutingDecision:
        """Refresh placement so earlier plans cannot overbook the live fleet."""
        effective_excluded = excluded | (decision.excluded if decision else frozenset())
        chosen = self.route(job, excluded=effective_excluded)
        # There is no await between routing and the supervisor's reservation.
        # Another dispatch therefore sees this job's occupied primary slot.
        LOGGER.info(
            "dispatching job %s (%s) to node %s: %s",
            job.job_id,
            job.payload.model,
            chosen.primary_node_id,
            chosen.rationale,
        )
        await self.supervisor.start(job, chosen)
        return chosen
