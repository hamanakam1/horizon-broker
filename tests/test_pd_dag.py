"""PD-DAG routing: prefill/decode disaggregation."""

from __future__ import annotations

import pytest

from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.contracts.nodes import NodeRole
from horizon_broker.routing.arithmetic import ComputePhase, analyse
from horizon_broker.routing.pd_dag import PDDagRouter, RoutingMode
from tests.conftest import make_job, make_node, make_payload


def fleet() -> list:
    """A three-node fleet: prefill, decode and general."""
    return [
        make_node(
            "m3-max",
            role=NodeRole.PREFILL,
            prefill=True,
            total_vram=128_000_000_000,
            available_vram=120_000_000_000,
            models=(),
            tokens_per_second=45.0,
        ),
        make_node(
            "pi-decode",
            role=NodeRole.DECODE,
            decode=True,
            models=(),
            tokens_per_second=22.0,
        ),
        make_node("pi-general", role=NodeRole.GENERAL, models=()),
    ]


def test_massive_prompt_with_short_output_is_prefill_bound() -> None:
    profile = analyse(make_payload(prompt="x", output_tokens=16, prompt_tokens=32_768))
    assert profile.phase is ComputePhase.PREFILL_BOUND


def test_short_prompt_with_massive_output_is_decode_bound() -> None:
    profile = analyse(make_payload(prompt="x", output_tokens=8_192, prompt_tokens=32))
    assert profile.phase is ComputePhase.DECODE_BOUND


def test_balanced_work_stays_balanced() -> None:
    profile = analyse(make_payload(prompt="x", output_tokens=512, prompt_tokens=512))
    assert profile.phase is ComputePhase.BALANCED


def test_prefill_heavy_job_routes_to_the_prefill_node() -> None:
    router = PDDagRouter()
    job = make_job(
        tier=PriorityTier.INTERACTIVE,
        prompt_tokens=32_768,
        output_tokens=16,
        max_latency_ms=10_000,
    )
    decision = router.route(job, fleet())
    assert decision.mode is RoutingMode.DIRECT
    assert decision.primary_node_id == "m3-max"
    assert decision.profile.phase is ComputePhase.PREFILL_BOUND


def test_decode_heavy_job_routes_to_the_decode_node() -> None:
    router = PDDagRouter()
    job = make_job(
        tier=PriorityTier.FLEX,
        prompt_tokens=32,
        output_tokens=8_192,
        max_latency_ms=600_000,
    )
    decision = router.route(job, fleet())
    assert decision.mode is RoutingMode.DIRECT
    assert decision.primary_node_id == "pi-decode"
    assert decision.profile.phase is ComputePhase.DECODE_BOUND


def test_balanced_job_routes_to_a_general_node() -> None:
    router = PDDagRouter()
    job = make_job(prompt_tokens=512, output_tokens=512)
    decision = router.route(job, fleet())
    assert decision.primary_node_id in {"pi-general", "pi-decode", "m3-max"}
    assert decision.profile.phase is ComputePhase.BALANCED


def test_routing_fails_loudly_when_the_mesh_cannot_serve_the_model() -> None:
    router = PDDagRouter()
    job = make_job(model="qwen2.5:32b")
    with pytest.raises(NoCapacityError):
        router.route(job, [make_node("pi-only", models=("qwen2.5:7b",))])


def test_excluded_nodes_are_never_selected() -> None:
    router = PDDagRouter()
    job = make_job(prompt_tokens=32_768, output_tokens=16)
    decision = router.route(job, fleet(), excluded=frozenset({"m3-max"}))
    assert decision.primary_node_id != "m3-max"


def test_loaded_nodes_are_preferred_over_idle_ones() -> None:
    router = PDDagRouter()
    nodes = fleet()
    nodes[0] = nodes[0].model_copy(update={"current_batch_size": 4})
    job = make_job(prompt_tokens=512, output_tokens=512)
    decision = router.route(job, nodes)
    assert decision.primary_node_id != "m3-max"
