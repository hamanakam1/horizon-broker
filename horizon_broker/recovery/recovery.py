"""Recovery coordinator.

When a node dies mid-inference the coordinator must, inside a bounded budget
(30 seconds by default):

1. load the newest :class:`FaultRecoverySnapshot` for the job,
2. validate it against the job (prompt hash + model must match),
3. re-queue the job with the recovered resume state,
4. dispatch it to a *different* node.

If no usable snapshot exists the job is restarted from scratch on the new
node -- degraded, but never lost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from horizon_broker.config import RecoveryConfig
from horizon_broker.contracts.errors import (
    BrokerError,
    MaxAttemptsExceededError,
    RecoveryBudgetExceededError,
)
from horizon_broker.contracts.jobs import InferenceJob, JobState, utcnow
from horizon_broker.contracts.nodes import NodeCapability
from horizon_broker.recovery.snapshot_store import SnapshotStore

LOGGER = logging.getLogger(__name__)

RequeueCallback = Callable[[InferenceJob], InferenceJob]
DispatchCallback = Callable[[InferenceJob, frozenset[str]], Awaitable[None]]


@dataclass(frozen=True)
class RecoveryOutcome:
    """Result of a recovery attempt."""

    job_id: str
    resumed: bool
    snapshot_id: str | None
    elapsed_seconds: float
    within_budget: bool
    dispatched: bool
    excluded_node_ids: frozenset[str]
    failure: BrokerError | None = None
    """The typed failure that ended the attempt, if it failed."""

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "resumed": self.resumed,
            "snapshot_id": self.snapshot_id,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "within_budget": self.within_budget,
            "dispatched": self.dispatched,
            "excluded_node_ids": sorted(self.excluded_node_ids),
            "failure": type(self.failure).__name__ if self.failure else None,
        }


class RecoveryCoordinator:
    """Rebuilds and re-dispatches jobs whose node died."""

    def __init__(
        self,
        *,
        store: SnapshotStore,
        config: RecoveryConfig,
        requeue: RequeueCallback,
        dispatch: DispatchCallback,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.config = config
        self._requeue = requeue
        self._dispatch = dispatch
        self._clock = clock
        self._avoid: dict[str, set[str]] = {}

    def avoid_node(self, job_id: str, node_id: str) -> None:
        """Exclude a node from future placement of ``job_id``."""
        self._avoid.setdefault(job_id, set()).add(node_id)

    def avoided_nodes(self, job_id: str) -> frozenset[str]:
        """Nodes excluded for ``job_id`` so far."""
        return frozenset(self._avoid.get(job_id, set()))

    async def recover(
        self,
        job: InferenceJob,
        failed_node_id: str,
        *,
        reason: str = "node lost",
        available_nodes: list[NodeCapability] | None = None,
    ) -> RecoveryOutcome:
        """Recover ``job`` after the loss of ``failed_node_id``."""
        started = self._clock()
        failure: BrokerError | None = None
        self.avoid_node(job.job_id, failed_node_id)
        excluded = self.avoided_nodes(job.job_id)

        if job.attempt + 1 >= self.config.max_attempts:
            failure = MaxAttemptsExceededError(
                f"max attempts exceeded after {reason} "
                f"({job.attempt + 1} of {self.config.max_attempts})"
            )
            job.state = JobState.FAILED
            job.error = str(failure)
            LOGGER.error("job %s failed permanently: %s", job.job_id, job.error)
            return RecoveryOutcome(
                job_id=job.job_id,
                resumed=False,
                snapshot_id=None,
                elapsed_seconds=self._clock() - started,
                within_budget=True,
                dispatched=False,
                excluded_node_ids=excluded,
                failure=failure,
            )

        snapshot = await self.store.latest_for_job(job.job_id)
        resumed = False
        if snapshot is not None:
            if snapshot.resume_point.prompt_hash != job.payload.prompt_hash:
                LOGGER.warning(
                    "discarding snapshot %s: prompt hash mismatch for job %s",
                    snapshot.snapshot_id,
                    job.job_id,
                )
            elif snapshot.model and snapshot.model != job.payload.model:
                LOGGER.warning(
                    "discarding snapshot %s: model mismatch for job %s",
                    snapshot.snapshot_id,
                    job.job_id,
                )
            else:
                resumed = True

        # The job is mutated in place: every holder of the reference (the job
        # store, the supervisor) must observe the recovered state.
        job.attempt += 1
        job.state = JobState.QUEUED
        job.assigned_node_id = None
        if resumed and snapshot is not None:
            job.resume_point = snapshot.resume_point
            job.kv_cache_state_id = (
                snapshot.resume_point.kv_cache_ref or snapshot.resume_point.kv_cache_state_id
            )
            job.tags = tuple(dict.fromkeys(job.tags + ("resumed",)))

        # The recovered job is dispatched directly. It is deliberately *not*
        # pushed back into the shared queue: the scheduler lane would be free
        # to pick it up a second time and dispatch it to the node that just
        # failed. The queue is only used as a fallback when no node can take
        # the work right now.
        dispatched = False
        try:
            await self._dispatch(job, excluded)
            dispatched = True
        except Exception as exc:  # noqa: BLE001 - the job is re-queued and retried later
            LOGGER.error("recovery dispatch failed for job %s: %s", job.job_id, exc)
            job.not_before = utcnow() + timedelta(seconds=5)
            self._requeue(job)

        elapsed = self._clock() - started
        within_budget = elapsed <= self.config.recovery_budget_seconds
        if not within_budget:
            failure = RecoveryBudgetExceededError(
                f"recovery for job {job.job_id} took {elapsed:.2f}s, "
                f"exceeding the {self.config.recovery_budget_seconds:.2f}s budget"
            )
            LOGGER.error("%s", failure)
        LOGGER.info(
            "recovery for job %s: resumed=%s snapshot=%s elapsed=%.2fs",
            job.job_id,
            resumed,
            snapshot.snapshot_id if snapshot else None,
            elapsed,
        )
        return RecoveryOutcome(
            job_id=job.job_id,
            resumed=resumed,
            snapshot_id=snapshot.snapshot_id if snapshot else None,
            elapsed_seconds=elapsed,
            within_budget=within_budget,
            dispatched=dispatched,
            excluded_node_ids=excluded,
            failure=failure,
        )
