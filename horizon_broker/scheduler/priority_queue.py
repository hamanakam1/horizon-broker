"""Priority queue for inference jobs.

Two lanes live side by side:

``INTERACTIVE``
    Ordered by deadline. The head of the queue is always the job with the
    tightest latency budget.

``FLEX``
    Ordered by arrival. The queue itself does not decide *when* to dispatch;
    that is the windowing policy's job (see :mod:`horizon_broker.scheduler.window`).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from horizon_broker.contracts.jobs import InferenceJob, JobState, PriorityTier
from horizon_broker.scheduler.window import (
    FlexBatch,
    WindowPolicy,
    WindowState,
    evaluate_window,
)


@dataclass(order=True)
class _Entry:
    """Heap entry; ``job`` is excluded from comparison."""

    sort_key: tuple[float, ...]
    seq: int
    job: InferenceJob = field(compare=False)


@dataclass(frozen=True)
class QueueStats:
    """Snapshot of the queue occupancy."""

    interactive: int
    flex: int
    expired: int
    open_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "interactive": self.interactive,
            "flex": self.flex,
            "expired": self.expired,
            "open_windows": self.open_windows,
        }


class JobPriorityQueue:
    """Async-friendly priority queue with per-model Flex windows."""

    def __init__(self) -> None:
        self._interactive: list[_Entry] = []
        self._flex: list[_Entry] = []
        self._flex_by_model: dict[str, list[InferenceJob]] = defaultdict(list)
        self._by_id: dict[str, InferenceJob] = {}
        self._windows: dict[str, WindowState] = {}
        self._seq = itertools.count()
        self._expired: list[InferenceJob] = []
        self.wakeup = asyncio.Event()

    # -- mutation -----------------------------------------------------------

    def submit(self, job: InferenceJob) -> InferenceJob:
        """Enqueue a job on the lane matching its priority tier."""
        if job.job_id in self._by_id:
            return self._by_id[job.job_id]
        self._by_id[job.job_id] = job
        if job.priority_tier is PriorityTier.INTERACTIVE:
            deadline = job.deadline_at.timestamp() if job.deadline_at else float("inf")
            heapq.heappush(
                self._interactive,
                _Entry((deadline, job.submitted_at.timestamp()), next(self._seq), job),
            )
        else:
            job.state = JobState.WINDOWING
            heapq.heappush(
                self._flex,
                _Entry((job.submitted_at.timestamp(),), next(self._seq), job),
            )
            self._flex_by_model[job.payload.model].append(job)
        self.wakeup.set()
        return job

    def requeue_front(self, job: InferenceJob, *, reason: str = "recovered") -> InferenceJob:
        """Put a job back at the head of its lane (used after a fault).

        Recovered work keeps its original latency budget but is promoted so it
        is picked up before any other waiting job of the same tier. The job
        instance is mutated in place so that every holder of the reference
        (the engine's job store, the recovery coordinator) sees the same
        object.
        """
        self._remove_locked(job.job_id)
        job.state = JobState.QUEUED
        job.assigned_node_id = None
        job.tags = tuple(dict.fromkeys(job.tags + (f"recovered:{reason}",)))
        self._by_id[job.job_id] = job
        if job.priority_tier is PriorityTier.INTERACTIVE:
            deadline = job.deadline_at.timestamp() if job.deadline_at else float("inf")
            heapq.heappush(
                self._interactive,
                _Entry((deadline - 1e-6, job.submitted_at.timestamp()), next(self._seq), job),
            )
        else:
            job.state = JobState.WINDOWING
            heapq.heappush(
                self._flex,
                _Entry((job.submitted_at.timestamp() - 1e-6,), next(self._seq), job),
            )
            self._flex_by_model[job.payload.model].append(job)
        self.wakeup.set()
        return job

    def remove(self, job_id: str) -> InferenceJob | None:
        """Remove a job from the queue (cancellation)."""
        return self._remove_locked(job_id)

    def _remove_locked(self, job_id: str) -> InferenceJob | None:
        job = self._by_id.pop(job_id, None)
        if job is None:
            return None
        self._interactive = [entry for entry in self._interactive if entry.job.job_id != job_id]
        heapq.heapify(self._interactive)
        self._flex = [entry for entry in self._flex if entry.job.job_id != job_id]
        heapq.heapify(self._flex)
        pending = self._flex_by_model.get(job.payload.model)
        if pending is not None:
            self._flex_by_model[job.payload.model] = [
                item for item in pending if item.job_id != job_id
            ]
            if not self._flex_by_model[job.payload.model]:
                self._flex_by_model.pop(job.payload.model, None)
        return job

    # -- consumption --------------------------------------------------------

    def pop_interactive(self, now: float) -> InferenceJob | None:
        """Pop the most urgent interactive job.

        Jobs whose latency budget has already elapsed are retired into the
        expired list instead of being dispatched. Jobs whose ``not_before`` has
        not been reached yet stay queued.
        """
        skipped: list[_Entry] = []
        try:
            while self._interactive:
                entry = heapq.heappop(self._interactive)
                job = entry.job
                if job.job_id not in self._by_id:
                    continue
                if job.not_before is not None and job.not_before.timestamp() > now:
                    skipped.append(entry)
                    continue
                if job.deadline_at is not None and job.deadline_at.timestamp() <= now:
                    self._by_id.pop(job.job_id, None)
                    job.state = JobState.FAILED
                    job.error = "latency budget expired before dispatch"
                    self._expired.append(job)
                    continue
                self._by_id.pop(job.job_id, None)
                return job
            return None
        finally:
            for entry in skipped:
                heapq.heappush(self._interactive, entry)

    def plan_flex(
        self,
        now: float,
        policy: WindowPolicy,
        *,
        power_ok: bool = True,
        fleet_idle: bool = False,
    ) -> list[FlexBatch]:
        """Evaluate every open window and return the batches that should run.

        Jobs that are not yet eligible (``not_before`` in the future) stay in
        the queue.
        """
        if not power_ok:
            return []
        batches: list[FlexBatch] = []
        for model in list(self._flex_by_model.keys()):
            pending = self._flex_by_model.get(model, [])
            eligible = [
                job
                for job in pending
                if job.not_before is None or job.not_before.timestamp() <= now
            ]
            if not eligible:
                continue
            window = self._windows.get(model)
            if window is None:
                window = WindowState(
                    model=model,
                    opened_at=now,
                    deadline=now + policy.window_seconds,
                )
                self._windows[model] = window
            decision = evaluate_window(
                window, len(eligible), policy, now, fleet_idle=fleet_idle
            )
            if not decision.should_dispatch:
                continue
            selected = eligible[: policy.max_batch_size]
            for job in selected:
                self._remove_locked(job.job_id)
            self._windows.pop(model, None)
            batches.append(
                FlexBatch(
                    model=model,
                    jobs=tuple(selected),
                    reason=decision.reason,
                    opened_at=window.opened_at,
                    closed_at=now,
                )
            )
        return batches

    # -- introspection ------------------------------------------------------

    def get(self, job_id: str) -> InferenceJob | None:
        return self._by_id.get(job_id)

    def contains(self, job_id: str) -> bool:
        return job_id in self._by_id

    def expired_jobs(self) -> list[InferenceJob]:
        """Jobs retired because their latency budget elapsed."""
        return list(self._expired)

    def drain_expired(self) -> list[InferenceJob]:
        """Return the retired jobs and forget them.

        The engine consumes them: a job that ran out of latency budget is a
        terminal answer that has to reach the caller waiting for it.
        """
        expired = self._expired
        self._expired = []
        return expired

    def pending_flex(self, model: str | None = None) -> list[InferenceJob]:
        """Flex jobs still waiting, optionally filtered by model."""
        if model is None:
            return [job for jobs in self._flex_by_model.values() for job in jobs]
        return list(self._flex_by_model.get(model, []))

    def stats(self) -> QueueStats:
        return QueueStats(
            interactive=len(self._interactive),
            flex=sum(len(jobs) for jobs in self._flex_by_model.values()),
            expired=len(self._expired),
            open_windows=len(self._windows),
        )
