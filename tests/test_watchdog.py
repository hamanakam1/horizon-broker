"""Watchdog: detecting that a node died."""

from __future__ import annotations

from datetime import timedelta

from horizon_broker.config import HeartbeatConfig, RecoveryConfig
from horizon_broker.contracts.nodes import NodeHealth
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.nodes.watchdog import FaultWatchdog
from tests.conftest import make_node


def utcnow():
    from horizon_broker.contracts.recovery import utcnow as _utcnow

    return _utcnow()


async def test_a_silent_node_is_declared_lost() -> None:
    registry = NodeRegistry()
    node = make_node("pi-01")
    registry.upsert(node)
    registry.assign_job("pi-01", "job-1")
    registry.record_heartbeat("pi-01", utcnow() - timedelta(seconds=30))

    lost: list[tuple[str, str]] = []
    watchdog = FaultWatchdog(
        registry,
        config=HeartbeatConfig(timeout_seconds=6.0, interval_seconds=2.0),
        recovery_config=RecoveryConfig(quarantine_seconds=60.0),
        on_node_lost=lambda node_id, reason: _record(lost, node_id, reason),
    )
    declared = await watchdog.scan()
    assert declared == ["pi-01"]
    assert registry.get("pi-01").health is NodeHealth.LOST
    assert registry.is_quarantined("pi-01")
    assert lost and lost[0][0] == "pi-01"


async def test_a_responsive_node_is_left_alone() -> None:
    registry = NodeRegistry()
    registry.upsert(make_node("pi-01"))
    registry.record_heartbeat("pi-01", utcnow())
    lost: list[tuple[str, str]] = []
    watchdog = FaultWatchdog(registry, on_node_lost=lambda node_id, reason: _record(lost, node_id, reason))
    assert await watchdog.scan() == []
    assert registry.get("pi-01").health is NodeHealth.HEALTHY
    assert lost == []


async def test_a_draining_node_is_migrated_without_being_lost() -> None:
    registry = NodeRegistry()
    node = make_node("trailer-01")
    registry.upsert(node.model_copy(update={"health": NodeHealth.DRAINING}))
    draining: list[str] = []
    watchdog = FaultWatchdog(
        registry,
        on_node_draining=lambda node_id: _record_draining(draining, node_id),
    )
    assert await watchdog.scan() == []
    assert draining == ["trailer-01"]
    assert registry.get("trailer-01").health is NodeHealth.DRAINING


async def test_a_node_returns_to_the_fleet_after_its_quarantine_expires() -> None:
    """A node that served out its cooling-off period is usable again.

    The quarantine is a setting with a duration; a node that failed once
    must not be lost for the life of the process.
    """
    registry = NodeRegistry()
    node = make_node("pi-01").model_copy(
        update={
            "health": NodeHealth.LOST,
            "quarantined_until": utcnow() - timedelta(seconds=1),
        }
    )
    registry.upsert(node)
    registry.record_heartbeat("pi-01", utcnow())

    watchdog = FaultWatchdog(registry)
    assert await watchdog.scan() == []
    assert registry.get("pi-01").health is NodeHealth.HEALTHY
    assert not registry.is_quarantined("pi-01")


async def _record(sink: list[tuple[str, str]], node_id: str, reason: str) -> None:
    sink.append((node_id, reason))


async def _record_draining(sink: list[str], node_id: str) -> None:
    sink.append(node_id)
