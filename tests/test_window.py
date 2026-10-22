"""Windowing: the sleep-time batching policy."""

from __future__ import annotations

import time

from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.scheduler.priority_queue import JobPriorityQueue
from horizon_broker.scheduler.window import WindowPolicy
from tests.conftest import make_job

POLICY = WindowPolicy(window_seconds=10.0, min_batch_size=4, max_batch_size=8, min_dwell_seconds=1.0)


def test_window_holds_until_the_minimum_batch_is_reached() -> None:
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(3):
        queue.submit(make_job(tier=PriorityTier.FLEX))
    assert queue.plan_flex(now, POLICY) == []


def test_window_dispatches_when_min_batch_and_dwell_are_satisfied() -> None:
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(4):
        queue.submit(make_job(tier=PriorityTier.FLEX))
    assert queue.plan_flex(now, POLICY) == []
    batches = queue.plan_flex(now + 1.5, POLICY)
    assert len(batches) == 1
    assert batches[0].size == 4
    assert batches[0].reason == "min_batch_reached"
    assert queue.pending_flex() == []


def test_window_expires_and_flushes_what_it_has() -> None:
    queue = JobPriorityQueue()
    now = time.time()
    queue.submit(make_job(tier=PriorityTier.FLEX))
    assert queue.plan_flex(now, POLICY) == []
    batches = queue.plan_flex(now + 11.0, POLICY)
    assert len(batches) == 1
    assert batches[0].reason == "window_expired"
    assert batches[0].size == 1


def test_a_full_batch_leaves_the_window_immediately() -> None:
    policy = WindowPolicy(window_seconds=60.0, min_batch_size=4, max_batch_size=4, min_dwell_seconds=30.0)
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(4):
        queue.submit(make_job(tier=PriorityTier.FLEX))
    batches = queue.plan_flex(now, policy)
    assert len(batches) == 1
    assert batches[0].reason == "batch_full"


def test_windows_are_independent_per_model() -> None:
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(4):
        queue.submit(make_job(tier=PriorityTier.FLEX, model="qwen2.5:7b"))
    for _ in range(4):
        queue.submit(make_job(tier=PriorityTier.FLEX, model="qwen3:8b"))
    # The first pass opens a window per model; nothing is dispatched yet.
    assert queue.plan_flex(now, POLICY) == []
    batches = queue.plan_flex(now + 1.5, POLICY)
    models = {batch.model for batch in batches}
    assert models == {"qwen2.5:7b", "qwen3:8b"}
    for batch in batches:
        assert batch.size == 4


def test_power_gate_holds_the_batch_back() -> None:
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(4):
        queue.submit(make_job(tier=PriorityTier.FLEX))
    assert queue.plan_flex(now + 1.5, POLICY, power_ok=False) == []
    assert len(queue.pending_flex()) == 4


def test_batch_is_capped_at_max_batch_size() -> None:
    policy = WindowPolicy(window_seconds=60.0, min_batch_size=2, max_batch_size=3, min_dwell_seconds=0.0)
    queue = JobPriorityQueue()
    now = time.time()
    for _ in range(7):
        queue.submit(make_job(tier=PriorityTier.FLEX))
    batches = queue.plan_flex(now, policy)
    assert batches[0].size == 3
    assert len(queue.pending_flex()) == 4


def test_an_idle_fleet_flushes_the_window_early() -> None:
    """One job must not wait for a batch that will never arrive."""
    policy = WindowPolicy(
        window_seconds=60.0,
        min_batch_size=8,
        max_batch_size=16,
        min_dwell_seconds=30.0,
        idle_flush=True,
        idle_flush_seconds=2.0,
    )
    queue = JobPriorityQueue()
    now = time.time()
    queue.submit(make_job(tier=PriorityTier.FLEX))

    # The first pass opens the window; the fleet is busy, so nothing goes out.
    assert queue.plan_flex(now, policy, fleet_idle=False) == []
    batches = queue.plan_flex(now + 3.0, policy, fleet_idle=True)
    assert len(batches) == 1
    assert batches[0].reason == "fleet_idle"
    assert batches[0].size == 1


def test_the_idle_flush_can_be_switched_off() -> None:
    policy = WindowPolicy(
        window_seconds=60.0,
        min_batch_size=8,
        max_batch_size=16,
        min_dwell_seconds=30.0,
        idle_flush=False,
        idle_flush_seconds=2.0,
    )
    queue = JobPriorityQueue()
    now = time.time()
    queue.submit(make_job(tier=PriorityTier.FLEX))
    assert queue.plan_flex(now, policy) == []
    assert queue.plan_flex(now + 3.0, policy, fleet_idle=True) == []
