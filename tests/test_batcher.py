"""Batch planning respects the capacity advertised by each node."""

from __future__ import annotations

import pytest

from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.routing.pd_dag import PDDagRouter
from horizon_broker.scheduler.batcher import BatchPlanner
from tests.conftest import make_job, make_node


def test_planner_splits_a_group_at_remaining_node_capacity() -> None:
    node = make_node("small-node", max_batch=3, current_batch=1)
    jobs = [make_job(tier=PriorityTier.FLEX, prompt=f"job-{index}") for index in range(4)]

    batches = BatchPlanner(PDDagRouter()).plan(jobs, [node])

    assert [batch.size for batch in batches] == [2, 2]
    assert [batch.job_ids() for batch in batches] == [
        (jobs[0].job_id, jobs[1].job_id),
        (jobs[2].job_id, jobs[3].job_id),
    ]


def test_planner_drops_a_node_with_no_remaining_capacity() -> None:
    node = make_node("full-node", max_batch=2, current_batch=2)
    jobs = [make_job(tier=PriorityTier.FLEX)]

    assert BatchPlanner(PDDagRouter()).plan(jobs, [node]) == []


def test_planner_keeps_different_models_in_separate_batches() -> None:
    node = make_node(
        "shared-node",
        max_batch=4,
        models=("synthetic-model-a", "synthetic-model-b"),
    )
    jobs = [
        make_job(tier=PriorityTier.FLEX, model="synthetic-model-a", prompt="a"),
        make_job(tier=PriorityTier.FLEX, model="synthetic-model-b", prompt="b"),
    ]

    batches = BatchPlanner(PDDagRouter()).plan(jobs, [node])

    assert [batch.job_ids() for batch in batches] == [
        (jobs[0].job_id,),
        (jobs[1].job_id,),
    ]


def test_planner_does_not_hide_an_unexpected_router_failure() -> None:
    class BrokenRouter(PDDagRouter):
        def route(self, job, nodes, *, excluded=frozenset()):
            raise RuntimeError("router invariant failed")

    node = make_node("healthy-node")
    job = make_job(tier=PriorityTier.FLEX)

    with pytest.raises(RuntimeError, match="router invariant failed"):
        BatchPlanner(BrokenRouter()).plan([job], [node])


def test_planner_keeps_expected_no_capacity_failures_retryable() -> None:
    class FullRouter(PDDagRouter):
        def route(self, job, nodes, *, excluded=frozenset()):
            raise NoCapacityError("no capacity")

    node = make_node("healthy-node")
    job = make_job(tier=PriorityTier.FLEX)

    assert BatchPlanner(FullRouter()).plan([job], [node]) == []
