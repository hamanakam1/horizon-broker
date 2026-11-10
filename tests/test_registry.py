"""Node registry ownership and capacity accounting tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from horizon_broker.contracts.nodes import NodeCapability, NodeHealth
from horizon_broker.nodes.registry import NodeRegistry


def test_releasing_an_unknown_job_preserves_reported_node_occupancy() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
        current_batch_size=2,
    )
    registry.upsert(node)

    registry.release_job(node.node_id, "late-job")

    current = registry.get(node.node_id)
    assert current is not None
    assert current.current_batch_size == 2
    assert current.assigned_job_ids == ()


def test_record_heartbeat_rejects_a_naive_timestamp() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
    )
    registry.upsert(node)

    with pytest.raises(ValueError, match="heartbeat timestamp must be timezone-aware"):
        registry.record_heartbeat(node.node_id, datetime(2026, 10, 23, 12, 0, 0))


def test_quarantine_lookup_rejects_a_naive_timestamp() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        quarantined_until=datetime.now(UTC) + timedelta(seconds=60),
    )
    registry.upsert(node)

    with pytest.raises(ValueError, match="quarantine timestamp must be timezone-aware"):
        registry.is_quarantined(node.node_id, datetime(2026, 10, 23, 12, 0, 0))


def test_quarantine_lookup_rejects_a_naive_timestamp_for_an_unknown_node() -> None:
    registry = NodeRegistry()

    with pytest.raises(ValueError, match="quarantine timestamp must be timezone-aware"):
        registry.is_quarantined("missing-node", datetime(2026, 10, 23, 12, 0, 0))


def test_heartbeat_rejects_a_naive_timestamp_for_an_unknown_node() -> None:
    registry = NodeRegistry()

    with pytest.raises(ValueError, match="heartbeat timestamp must be timezone-aware"):
        registry.record_heartbeat("missing-node", datetime(2026, 10, 23, 12, 0, 0))


def test_releasing_a_local_job_preserves_remote_occupancy() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
        current_batch_size=2,
    )
    registry.upsert(node)
    registry.assign_job(node.node_id, "local-job")

    registry.release_job(node.node_id, "local-job")

    current = registry.get(node.node_id)
    assert current is not None
    assert current.assigned_job_ids == ()
    assert current.current_batch_size == 2


def test_new_registration_always_seeds_the_reported_occupancy_baseline() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
        current_batch_size=2,
    )

    registry.upsert(node, preserve_reported_batch_size=True)
    registry.assign_job(node.node_id, "local-job")
    registry.release_job(node.node_id, "local-job")

    current = registry.get(node.node_id)
    assert current is not None
    assert current.current_batch_size == 2


def test_removing_a_node_discards_its_stale_occupancy_baseline() -> None:
    registry = NodeRegistry()
    first = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
        current_batch_size=3,
    )
    registry.upsert(first)
    registry.remove(first.node_id)

    replacement = first.model_copy(update={"current_batch_size": 0})
    registry.upsert(replacement)
    registry.assign_job(replacement.node_id, "local-job")
    registry.release_job(replacement.node_id, "local-job")

    current = registry.get(replacement.node_id)
    assert current is not None
    assert current.current_batch_size == 0


def test_a_capability_refresh_preserves_broker_owned_assignments() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
    )
    registry.upsert(node)
    registry.assign_job(node.node_id, "job-1")

    refreshed = NodeCapability(
        node_id=node.node_id,
        available_vram_bytes=7,
        total_vram_bytes=8,
        max_batch_size=4,
        current_batch_size=0,
    )
    registry.upsert(refreshed)

    current = registry.get(node.node_id)
    assert current is not None
    assert registry.jobs_on(node.node_id) == ("job-1",)
    assert current.current_batch_size == 1


def test_a_capability_refresh_cannot_clear_watchdog_health_or_quarantine() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
    )
    registry.upsert(node)
    registry.mark_health(node.node_id, NodeHealth.LOST, "synthetic loss")
    registry.quarantine(node.node_id, 60.0)
    before = registry.get(node.node_id)
    assert before is not None
    quarantine_until = before.quarantined_until
    assert quarantine_until is not None

    registry.upsert(
        NodeCapability(
            node_id=node.node_id,
            available_vram_bytes=8,
            total_vram_bytes=8,
            max_batch_size=4,
        )
    )

    current = registry.get(node.node_id)
    assert current is not None
    assert current.health is NodeHealth.LOST
    assert current.quarantined_until == quarantine_until
    assert current.quarantined_until > datetime.now(UTC) + timedelta(seconds=59)


def test_a_capability_refresh_merges_metadata_without_erasing_batch_opt_in() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
        execution_mode="agent_container",
        metadata={"supports_batch": True, "health_reason": "synthetic"},
    )
    registry.upsert(node)

    registry.upsert(
        NodeCapability(
            node_id=node.node_id,
            available_vram_bytes=7,
            total_vram_bytes=8,
            max_batch_size=4,
        )
    )

    current = registry.get(node.node_id)
    assert current is not None
    assert current.metadata["supports_batch"] is True
    assert current.metadata["health_reason"] == "synthetic"


def test_a_capability_refresh_cannot_overwrite_the_broker_health_reason() -> None:
    registry = NodeRegistry()
    node = NodeCapability(
        node_id="agent-node",
        available_vram_bytes=8,
        total_vram_bytes=8,
        max_batch_size=4,
    )
    registry.upsert(node)
    registry.mark_health(node.node_id, NodeHealth.LOST, "broker verdict")

    registry.upsert(
        node.model_copy(update={"metadata": {"health_reason": "node claim"}})
    )

    current = registry.get(node.node_id)
    assert current is not None
    assert current.metadata["health_reason"] == "broker verdict"
