"""Contract level tests: the wire format is the API, so it is pinned here."""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from pydantic import ValidationError

from horizon_broker.contracts.jobs import (
    InferenceJob,
    InferencePayload,
    JobState,
    PriorityTier,
    SamplingParams,
)
from horizon_broker.contracts.ledger import LedgerCommitReceipt, LedgerRecord
from horizon_broker.contracts.nodes import (
    ExecutionMode,
    NodeCapability,
    NodeHealth,
    NodeRole,
    PowerClass,
)
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


def test_job_clock_helpers_reject_naive_timestamps() -> None:
    job = InferenceJob(
        priority_tier=PriorityTier.INTERACTIVE,
        payload=InferencePayload(model="qwen2.5:7b", prompt="hi"),
        max_acceptable_latency_ms=5_000,
    )
    naive = datetime(2026, 10, 23, 12, 0, 0)

    with pytest.raises(ValueError, match="job timestamp must be timezone-aware"):
        job.remaining_latency_ms(naive)
    with pytest.raises(ValueError, match="job timestamp must be timezone-aware"):
        job.is_expired(naive)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="qwen2.5:7b", prompt="hi"),
            max_acceptable_latency_ms=1_000,
            surprise="not allowed",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collection", " horizon_broker.snapshots"),
        ("document_id", "snapshot-1 "),
        ("schema_version", "2.0"),
        ("content_hash", "not-a-hash"),
        ("tags", ("horizon_broker", " ")),
        ("tags", ("horizon_broker", "horizon_broker")),
    ],
)
def test_ledger_records_reject_non_canonical_envelopes(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LedgerRecord(
            collection=value if field == "collection" else "horizon_broker.snapshots",
            document_id=value if field == "document_id" else "snapshot-1",
            schema_version=value if field == "schema_version" else "1.0",
            content_hash=value if field == "content_hash" else "",
            tags=value if field == "tags" else (),
        )


@pytest.mark.parametrize(
    "commit_url",
    [
        "relative/doc-1",
        "javascript:alert(1)",
        " http://ledger/doc-1",
        "http://user:secret@ledger/doc-1",
        "http://ledger:not-a-port/doc-1",
        "http://ledger:70000/doc-1",
        "http://[malformed/doc-1",
        "http://:80/doc-1",
    ],
)
def test_ledger_commit_receipts_reject_non_http_urls(commit_url: str) -> None:
    with pytest.raises(ValidationError, match="commit_url"):
        LedgerCommitReceipt(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            content_hash="0" * 64,
            commit_url=commit_url,
        )


def test_ledger_commit_receipts_require_a_sha256_hash() -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        LedgerCommitReceipt(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            content_hash="not-a-hash",
        )


def test_ledger_commit_receipts_reject_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="committed_at must be timezone-aware"):
        LedgerCommitReceipt(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            content_hash="0" * 64,
            committed_at=datetime(2026, 10, 23, 12, 0, 0),
        )


def test_ledger_records_reject_a_payload_hash_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        LedgerRecord(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            payload={"job_id": "job-1"},
            content_hash="0" * 64,
        )


def test_ledger_records_reject_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="created_at must be timezone-aware"):
        LedgerRecord(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            created_at=datetime(2026, 10, 23, 12, 0, 0),
        )


def test_ledger_records_reject_non_standard_json_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="snapshot-1",
            payload={"score": math.nan},
        )


def test_payload_token_estimates() -> None:
    payload = InferencePayload(
        model="qwen2.5:7b",
        prompt="x" * 400,
        expected_output_tokens=256,
    )
    assert payload.estimated_prompt_tokens == 100
    assert payload.estimated_output_tokens == 256


@pytest.mark.parametrize("model", ["", "   ", " synthetic-model"])
def test_payload_rejects_an_empty_model_identifier(model: str) -> None:
    with pytest.raises(ValidationError, match="model must not"):
        InferencePayload(model=model)


def test_payload_falls_back_to_num_predict() -> None:
    payload = InferencePayload(
        model="qwen2.5:7b",
        prompt="hi",
        parameters=SamplingParams(num_predict=99),
    )
    assert payload.estimated_output_tokens == 99


def test_payload_properties_revalidate_mutated_messages() -> None:
    payload = InferencePayload(model="synthetic-model", prompt="hello")
    payload.messages.append({"role": "invalid", "content": "corrupt"})

    with pytest.raises(ValidationError, match="role"):
        _ = payload.estimated_prompt_tokens
    with pytest.raises(ValidationError, match="role"):
        _ = payload.prompt_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", True),
        ("top_p", False),
        ("top_k", True),
        ("num_predict", True),
        ("seed", False),
        ("repeat_penalty", True),
    ],
)
def test_sampling_parameters_reject_boolean_numerics(field: str, value: bool) -> None:
    with pytest.raises(ValidationError):
        SamplingParams(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.5"),
        ("top_p", "0.5"),
        ("top_k", "10"),
        ("num_predict", "10"),
        ("seed", "10"),
        ("repeat_penalty", "1.1"),
    ],
)
def test_sampling_parameters_reject_string_numerics(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        SamplingParams(**{field: value})


@pytest.mark.parametrize("field", ["expected_output_tokens", "prompt_tokens_hint"])
def test_payload_token_hints_reject_boolean_numerics(field: str) -> None:
    with pytest.raises(ValidationError):
        InferencePayload(model="synthetic-model", **{field: True})


@pytest.mark.parametrize("field", ["expected_output_tokens", "prompt_tokens_hint"])
def test_payload_token_hints_reject_string_numerics(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be an integer"):
        InferencePayload(model="synthetic-model", **{field: "10"})


@pytest.mark.parametrize("field", ["max_acceptable_latency_ms", "attempt"])
def test_job_counters_reject_boolean_numerics(field: str) -> None:
    with pytest.raises(ValidationError):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="synthetic-model"),
            max_acceptable_latency_ms=1_000 if field == "attempt" else True,
            **({field: True} if field == "attempt" else {}),
        )


@pytest.mark.parametrize("field", ["max_acceptable_latency_ms", "attempt"])
def test_job_counters_reject_string_numerics(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be an integer"):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="synthetic-model"),
            max_acceptable_latency_ms="1000" if field == "max_acceptable_latency_ms" else 1000,
            **({field: "0"} if field == "attempt" else {}),
        )


def test_an_overflowing_latency_budget_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="max_acceptable_latency_ms"):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="synthetic-model"),
            max_acceptable_latency_ms=10**1000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", ""),
        ("job_id", " job-1"),
        ("kv_cache_state_id", " cache-1 "),
        ("assigned_node_id", " node-1"),
        ("caller", " caller "),
    ],
)
def test_jobs_reject_non_canonical_identities(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match="identity must not"):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="synthetic-model"),
            max_acceptable_latency_ms=1000,
            **{field: value},
        )


@pytest.mark.parametrize("tags", [("",), (" tag",), ("tag", "tag")])
def test_jobs_reject_ambiguous_tags(tags: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="tags"):
        InferenceJob(
            priority_tier=PriorityTier.INTERACTIVE,
            payload=InferencePayload(model="synthetic-model"),
            max_acceptable_latency_ms=1000,
            tags=tags,
        )


@pytest.mark.parametrize("field", ["index", "token_id", "logprob"])
def test_partial_tokens_reject_boolean_numerics(field: str) -> None:
    with pytest.raises(ValidationError):
        PartialToken(**{field: True})


def test_partial_tokens_reject_an_overflowing_logprob() -> None:
    with pytest.raises(ValidationError, match="logprob"):
        PartialToken(index=0, logprob=10**1000)


@pytest.mark.parametrize(
    ("field", "value"),
    [("index", "1"), ("token_id", "2"), ("logprob", "0.5"), ("text", 1)],
)
def test_partial_tokens_reject_coercible_non_wire_types(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        fields: dict[str, object] = {"index": 0, field: value}
        PartialToken(**fields)


@pytest.mark.parametrize("field", ["decode_step", "generated_token_count"])
def test_resume_points_reject_boolean_counters(field: str) -> None:
    with pytest.raises(ValidationError):
        ResumePoint(
            **{
                "decode_step": 0 if field == "generated_token_count" else True,
                "generated_token_count": 0 if field == "decode_step" else True,
                "prompt_hash": "synthetic-hash",
            }
        )


@pytest.mark.parametrize("field", ["decode_step", "generated_token_count"])
def test_resume_points_reject_string_counters(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be an integer"):
        ResumePoint(
            **{
                "decode_step": "0" if field == "decode_step" else 0,
                "generated_token_count": "0" if field == "generated_token_count" else 0,
                "prompt_hash": "0" * 64,
            }
        )


@pytest.mark.parametrize("field", ["attempt", "step"])
def test_recovery_snapshots_reject_boolean_counters(field: str) -> None:
    with pytest.raises(ValidationError):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            attempt=True if field == "attempt" else 0,
            step=True if field == "step" else 0,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
        )


@pytest.mark.parametrize("field", ["attempt", "step"])
def test_recovery_snapshots_reject_string_counters(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be an integer"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            attempt="0" if field == "attempt" else 0,
            step="0" if field == "step" else 0,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
        )


@pytest.mark.parametrize("field", ["snapshot_id", "job_id", "node_id", "model"])
def test_recovery_snapshots_reject_empty_identity(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must not be empty"):
        FaultRecoverySnapshot(
            snapshot_id="" if field == "snapshot_id" else "snapshot-1",
            job_id="" if field == "job_id" else "job-1",
            node_id="" if field == "node_id" else "node-1",
            model="" if field == "model" else "synthetic-model",
            step=0,
            resume_point=ResumePoint(
                decode_step=1,
                generated_token_count=1,
                prompt_hash="0" * 64,
            ),
        )


def test_resume_points_reject_empty_prompt_hash() -> None:
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        ResumePoint(decode_step=0, generated_token_count=0, prompt_hash=" ")


def test_resume_points_reject_non_hash_prompt_identity() -> None:
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        ResumePoint(decode_step=0, generated_token_count=0, prompt_hash="synthetic-hash")


def test_resume_points_reject_conflicting_cache_aliases() -> None:
    with pytest.raises(ValidationError, match="kv_cache_ref and kv_cache_state_id"):
        ResumePoint(
            decode_step=0,
            generated_token_count=0,
            prompt_hash="0" * 64,
            kv_cache_ref="cache-a",
            kv_cache_state_id="cache-b",
        )


@pytest.mark.parametrize("field", ["kv_cache_ref", "kv_cache_state_id"])
def test_resume_points_reject_empty_cache_references(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must not be empty"):
        ResumePoint(
            decode_step=0,
            generated_token_count=0,
            prompt_hash="0" * 64,
            **{field: "  "},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("kv_cache_ref", " cache-a"), ("kv_cache_state_id", "cache-a "), ("model", " model-a")],
)
def test_resume_points_reject_non_canonical_identity_fields(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        ResumePoint(
            decode_step=0,
            generated_token_count=0,
            prompt_hash="0" * 64,
            **{field: value},
        )


def test_resume_points_reject_unsupported_checksum_algorithm() -> None:
    with pytest.raises(ValidationError, match="checksum_algorithm must be sha256"):
        ResumePoint(
            decode_step=0,
            generated_token_count=0,
            prompt_hash="0" * 64,
            checksum_algorithm="sha512",
        )


def test_recovery_snapshots_reject_a_model_mismatch() -> None:
    with pytest.raises(ValidationError, match="resume_point model"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="model-a",
            step=0,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
            prompt_hash="0" * 64,
                model="model-b",
            ),
        )


@pytest.mark.parametrize("field", ["ledger_document_id", "content_hash"])
def test_recovery_snapshots_reject_malformed_optional_integrity_fields(field: str) -> None:
    value = " " if field == "ledger_document_id" else "not-a-hash"
    message = "must not be empty" if field == "ledger_document_id" else "lowercase SHA-256"
    with pytest.raises(ValidationError, match=message):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=0,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
            **{field: value},
        )


def test_recovery_snapshots_reject_padded_identity() -> None:
    with pytest.raises(ValidationError, match="snapshot_id must not have surrounding"):
        FaultRecoverySnapshot(
            snapshot_id=" snapshot-1",
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=0,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
        )


def test_recovery_snapshots_reject_a_mismatched_content_hash() -> None:
    with pytest.raises(ValidationError, match="content_hash does not match"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=0,
            content_hash="0" * 64,
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
        )


@pytest.mark.parametrize("indices", [(1,), (0, 2)])
def test_recovery_snapshots_reject_non_sequential_partial_tokens(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="partial_tokens indices"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=0,
            partial_tokens=[PartialToken(index=index, text="token") for index in indices],
            resume_point=ResumePoint(
                decode_step=0,
                generated_token_count=0,
                prompt_hash="0" * 64,
            ),
        )


@pytest.mark.parametrize(
    ("step", "decode_step", "generated_token_count", "message"),
    [
        (2, 1, 1, "step must match resume_point decode_step"),
        (2, 2, 1, "resume_point counters must match"),
    ],
)
def test_recovery_snapshots_reject_ambiguous_progress_coordinates(
    step: int, decode_step: int, generated_token_count: int, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=step,
            resume_point=ResumePoint(
                decode_step=decode_step,
                generated_token_count=generated_token_count,
                prompt_hash="0" * 64,
                model="synthetic-model",
            ),
        )


def test_recovery_snapshots_reject_more_partial_tokens_than_progress() -> None:
    with pytest.raises(ValidationError, match="partial_tokens cannot exceed"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=1,
            partial_tokens=[
                PartialToken(index=0, text="a"),
                PartialToken(index=1, text="b"),
            ],
            resume_point=ResumePoint(
                decode_step=1,
                generated_token_count=1,
                prompt_hash="0" * 64,
                model="synthetic-model",
            ),
        )


def test_recovery_snapshots_reject_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="created_at must be timezone-aware"):
        FaultRecoverySnapshot(
            job_id="job-1",
            node_id="node-1",
            model="synthetic-model",
            step=1,
            created_at=datetime(2026, 10, 23, 12, 0, 0),
            partial_tokens=[PartialToken(index=0, text="a")],
            resume_point=ResumePoint(
                decode_step=1,
                generated_token_count=1,
                prompt_hash="0" * 64,
                model="synthetic-model",
            ),
        )


def test_snapshot_serialization_revalidates_mutated_partial_tokens() -> None:
    snapshot = FaultRecoverySnapshot(
        job_id="job-1",
        node_id="node-1",
        model="synthetic-model",
        step=1,
        partial_tokens=[PartialToken(index=0, text="token")],
        resume_point=ResumePoint(
            decode_step=1,
            generated_token_count=1,
            prompt_hash="0" * 64,
        ),
    )
    snapshot.partial_tokens.append(PartialToken(index=2, text="corrupt"))

    with pytest.raises(ValidationError, match="partial_tokens indices"):
        snapshot.to_ledger_body()


@pytest.mark.parametrize("contract", ["job", "snapshot"])
def test_replayable_contracts_reject_unsupported_schema_versions(contract: str) -> None:
    if contract == "job":
        with pytest.raises(ValidationError, match="unsupported schema_version"):
            InferenceJob(
                priority_tier=PriorityTier.INTERACTIVE,
                payload=InferencePayload(model="synthetic-model"),
                max_acceptable_latency_ms=1_000,
                schema_version="2.0",
            )
    else:
        with pytest.raises(ValidationError, match="unsupported schema_version"):
            FaultRecoverySnapshot(
                job_id="job-1",
                node_id="node-1",
                model="synthetic-model",
                step=0,
                schema_version="2.0",
                resume_point=ResumePoint(
                    decode_step=0,
                    generated_token_count=0,
                prompt_hash="0" * 64,
                ),
            )


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


@pytest.mark.parametrize("node_id", ["", "   ", " node-1"])
def test_node_capability_rejects_an_empty_identity(node_id: str) -> None:
    with pytest.raises(ValidationError, match="node_id must not"):
        NodeCapability(node_id=node_id, available_vram_bytes=1, total_vram_bytes=1)


def test_node_capability_rejects_empty_model_whitelist_entries() -> None:
    with pytest.raises(ValidationError, match="model_whitelist entries"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            model_whitelist=("",),
        )


@pytest.mark.parametrize(
    "model_whitelist", [(" synthetic-model",), ("synthetic-model", "synthetic-model")]
)
def test_node_capability_rejects_ambiguous_model_whitelist_entries(
    model_whitelist: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="model_whitelist entries"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            model_whitelist=model_whitelist,
        )


@pytest.mark.parametrize(
    "assigned_job_ids", [("",), (" job-1",), ("job-1", "job-1")]
)
def test_node_capability_rejects_ambiguous_assigned_job_ids(
    assigned_job_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="assigned_job_ids"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            assigned_job_ids=assigned_job_ids,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("available_vram_bytes", "8"),
        ("total_vram_bytes", "16"),
        ("current_batch_size", "1"),
        ("max_batch_size", "2"),
        ("tokens_per_second", "1.5"),
        ("power_headroom", "0.5"),
    ],
)
def test_node_capability_rejects_string_telemetry(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        fields: dict[str, object] = {
            "node_id": "node-1",
            "available_vram_bytes": 8,
            "total_vram_bytes": 16,
            field: value,
        }
        NodeCapability(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint", "file:///tmp/node"),
        ("ollama_url", "not-a-url"),
        ("agent_ws_url", "http://node"),
        ("endpoint", "http://node "),
        ("agent_ws_url", " ws://node"),
        ("endpoint", "http://user:secret@node"),
        ("ollama_url", "http://node:not-a-port"),
        ("agent_ws_url", "ws://node:70000"),
        ("endpoint", "http://node:0"),
        ("endpoint", "http://[malformed"),
        ("agent_ws_url", "ws://[malformed"),
        ("endpoint", "http://:80"),
    ],
)
def test_node_capability_rejects_invalid_endpoints(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            **{field: value},
        )


@pytest.mark.parametrize("field", ["last_heartbeat_at", "quarantined_until"])
def test_node_capability_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValidationError, match="node timestamps must be timezone-aware"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            **{field: datetime(2026, 10, 23, 12, 0, 0)},
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, 10**1000])
def test_node_capability_rejects_non_finite_rate_telemetry(value: float) -> None:
    with pytest.raises(ValidationError, match="tokens_per_second must be finite"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            tokens_per_second=value,
        )


@pytest.mark.parametrize("metadata", [{"score": math.nan}, {"object": object()}])
def test_node_capability_rejects_non_json_metadata(metadata: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="metadata must contain only JSON-safe values"):
        NodeCapability(
            node_id="node-1",
            available_vram_bytes=1,
            total_vram_bytes=1,
            metadata=metadata,
        )


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


def test_batch_support_requires_an_explicit_agent_container_opt_in() -> None:
    direct = NodeCapability(
        node_id="direct",
        available_vram_bytes=10,
        total_vram_bytes=10,
        metadata={"supports_batch": True},
    )
    agent = direct.model_copy(
        update={"execution_mode": ExecutionMode.AGENT_CONTAINER}
    )
    string_opt_in = agent.model_copy(update={"metadata": {"supports_batch": "true"}})

    assert not direct.supports_batch
    assert agent.supports_batch
    assert not string_opt_in.supports_batch


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
                generated_token_count=42,
            prompt_hash="0" * 64,
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
        resume_point=ResumePoint(
            decode_step=1, generated_token_count=1, prompt_hash="0" * 64
        ),
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
