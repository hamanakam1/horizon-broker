"""Priority queue behaviour: ordering, expiry, promotion and per-model grouping."""

from __future__ import annotations

import time

from horizon_broker.contracts.jobs import JobState, PriorityTier
from horizon_broker.scheduler.priority_queue import JobPriorityQueue
from tests.conftest import make_job


def test_interactive_jobs_are_ordered_by_deadline() -> None:
    queue = JobPriorityQueue()
    relaxed = make_job(max_latency_ms=60_000)
    urgent = make_job(max_latency_ms=1_000)
    queue.submit(relaxed)
    queue.submit(urgent)
    now = time.time()
    assert queue.pop_interactive(now).job_id == urgent.job_id
    assert queue.pop_interactive(now).job_id == relaxed.job_id


def test_expired_jobs_are_retired_instead_of_dispatched() -> None:
    queue = JobPriorityQueue()
    job = make_job(max_latency_ms=1)
    queue.submit(job)
    time.sleep(0.01)
    assert queue.pop_interactive(time.time() + 10) is None
    assert job.state is JobState.FAILED
    assert queue.expired_jobs()[0].job_id == job.job_id


def test_requeue_front_preserves_identity_and_promotes() -> None:
    queue = JobPriorityQueue()
    job = make_job()
    queue.submit(job)
    promoted = queue.requeue_front(job, reason="recovery")
    assert promoted is job
    assert job.state is JobState.QUEUED
    assert "recovered:recovery" in job.tags
    assert queue.pop_interactive(time.time()).job_id == job.job_id


def test_flex_jobs_are_grouped_per_model() -> None:
    queue = JobPriorityQueue()
    first = make_job(tier=PriorityTier.FLEX, model="qwen2.5:7b")
    second = make_job(tier=PriorityTier.FLEX, model="qwen2.5:7b")
    third = make_job(tier=PriorityTier.FLEX, model="qwen3:8b")
    for job in (first, second, third):
        queue.submit(job)
    assert len(queue.pending_flex("qwen2.5:7b")) == 2
    assert len(queue.pending_flex("qwen3:8b")) == 1
    assert queue.stats().flex == 3


def test_remove_takes_the_job_out_of_the_queue() -> None:
    queue = JobPriorityQueue()
    job = make_job(tier=PriorityTier.FLEX)
    queue.submit(job)
    assert queue.remove(job.job_id) is job
    assert queue.pending_flex() == []
    assert queue.remove(job.job_id) is None


def test_not_before_defers_dispatch_without_blocking_the_lane() -> None:
    queue = JobPriorityQueue()
    deferred = make_job()
    deferred.not_before = None
    ready = make_job(max_latency_ms=2_000)
    queue.submit(deferred)
    queue.submit(ready)
    deferred.not_before = deferred.submitted_at.replace(year=2099)
    now = time.time()
    popped = queue.pop_interactive(now)
    assert popped is not None
    assert popped.job_id == ready.job_id
    assert queue.get(deferred.job_id) is not None
