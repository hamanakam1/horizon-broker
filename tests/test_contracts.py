"""Contract level tests: the wire format is the API, so it is pinned here."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from horizon_broker.contracts.jobs import (
    InferenceJob,
    InferencePayload,
    JobState,
    PriorityTier,
    SamplingParams,
)
from horizon_broker.contracts.nodes import NodeCapability, NodeHealth, NodeRole, PowerClass
from horizon_broker.contracts.recovery import (
    FaultRecoverySnapshot,
    PartialToken,
    ResumePoint,
)


def test_inference_job_exposes_the_four_contract_fields() -> None:
    job = InferenceJob(
        priority_tier=PriorityTier.FLEX,
        payload=InferencePayload(model="qwen2.5:7b", prompt="scan the repo"),
        max_acceptable_latency_ms=3_600_000,
        kv_cache_state_id="edge-123",
    )
    assert job.priority_tier is PriorityTier.FLEX
    assert job.payload.model == "qwen2.5:7b"
    assert job.max_acceptable_latency_ms == 3_600_000
    assert job.kv_cache_state_id == "edge-123"
    assert job.state is JobState.QUEUED


def test_deadline_is_derived_from_the_latency_budget() -> None:
    job = InferenceJob(
        priority_tier=PriorityTier.INTERACTIVE,
        payload=InferencePayload(model="qwen2.5:7b", prompt="hi"),
        max_acceptable_latency_ms=5_000,
    )
    assert job.deadline_at is not None
    delta_ms = (job.deadline_at - job.submitted_at).total_seconds() * 1000
    assert delta_ms == pytest.approx(5_000, abs=1)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="qwen2.5:7b", prompt="hi"),
            max_acceptable_latency_ms=1_000,
            surprise="not allowed",
        )


def test_payload_token_estimates() -> None:
    payload = InferencePayload(
        model="qwen2.5:7b",
        prompt="x" * 400,
        expected_output_tokens=256,
    )
    assert payload.estimated_prompt_tokens == 100
    assert payload.estimated_output_tokens == 256


def test_payload_falls_back_to_num_predict() -> None:
    payload = InferencePayload(
        model="qwen2.5:7b",
        prompt="hi",
        parameters=SamplingParams(num_predict=99),
    )
    assert payload.estimated_output_tokens == 99


def test_prompt_hash_is_stable_and_sensitive() -> None:
    first = InferencePayload(model="qwen2.5:7b", prompt="alpha")
    second = InferencePayload(model="qwen2.5:7b", prompt="alpha")
    other = InferencePayload(model="qwen2.5:7b", prompt="beta")
    assert first.prompt_hash == second.prompt_hash
    assert first.prompt_hash != other.prompt_hash


def test_node_capability_clamps_impossible_memory_envelope() -> None:
    node = NodeCapability(
        node_id="n1",
        available_vram_bytes=900,
        total_vram_bytes=100,
        current_batch_size=9,
        max_batch_size=4,
    )
    assert node.available_vram_bytes == 100
    assert node.current_batch_size == 4
    assert node.batch_headroom == 0


def test_node_capability_acceptance_rules() -> None:
    node = NodeCapability(
        node_id="n1",
        role=NodeRole.DECODE,
        is_decode_optimized=True,
        available_vram_bytes=10,
        total_vram_bytes=10,
        current_batch_size=1,
        max_batch_size=2,
        model_whitelist=("qwen2.5:7b",),
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.4,
    )
    assert node.supports_model("qwen2.5:7b")
    assert not node.supports_model("qwen2.5:32b")
    assert node.accepts_work("qwen2.5:7b")
    assert node.vram_headroom_ratio == pytest.approx(1.0)


def test_fault_recovery_snapshot_hash_detects_tampering() -> None:
    snapshot = FaultRecoverySnapshot(
        job_id="job-1",
        node_id="node-a",
        attempt=0,
        model="qwen2.5:7b",
        step=42,
        partial_tokens=[PartialToken(index=0, text="Hel")],
        resume_point=ResumePoint(
            decode_step=42,
            generated_token_count=1,
            prompt_hash="abc",
            model="qwen2.5:7b",
        ),
    )
    original = snapshot.compute_content_hash()
    assert original == snapshot.compute_content_hash()
    snapshot.partial_tokens.append(PartialToken(index=1, text="lo"))
    assert snapshot.compute_content_hash() != original


def test_snapshot_body_carries_the_hash_and_version() -> None:
    snapshot = FaultRecoverySnapshot(
        job_id="job-1",
        node_id="node-a",
        model="qwen2.5:7b",
        step=1,
        resume_point=ResumePoint(decode_step=1, generated_token_count=1, prompt_hash="abc"),
    )
    body = snapshot.to_ledger_body()
    assert body["content_hash"] == snapshot.compute_content_hash()
    assert body["schema_version"] == "1.0"
    assert body["job_id"] == "job-1"


def test_health_transitions_are_explicit() -> None:
    node = NodeCapability(
        node_id="n1",
        health=NodeHealth.DRAINING,
        available_vram_bytes=1000,
        total_vram_bytes=1000,
    )
    assert node.health is NodeHealth.DRAINING
    assert not node.accepts_work("anything")
