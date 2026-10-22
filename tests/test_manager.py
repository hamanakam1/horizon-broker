"""The node connection manager."""

from __future__ import annotations

from pathlib import Path

from horizon_broker.contracts.nodes import ExecutionMode, NodeHealth
from horizon_broker.nodes.executor import JobSupervisor
from horizon_broker.nodes.manager import NodeConnectionManager
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.recovery.snapshot_store import SnapshotStore
from tests.conftest import make_node


def build(tmp_path: Path) -> tuple[NodeConnectionManager, NodeRegistry]:
    registry = NodeRegistry()
    supervisor = JobSupervisor(
        registry=registry,
        store=SnapshotStore(ledger=None, journal=LocalJournal(tmp_path / "journal.jsonl")),
    )
    return NodeConnectionManager(registry, supervisor), registry


async def test_the_connections_share_one_session(tmp_path: Path) -> None:
    """A session per node would leak a socket pool per reconnect."""
    manager, registry = build(tmp_path)
    registry.upsert(
        make_node(
            "agent-01",
            execution_mode=ExecutionMode.AGENT_CONTAINER,
            agent_ws_url="ws://127.0.0.1:1/broker",
        )
    )
    try:
        await manager._sync()
        connection = manager._connections["agent-01"]
        assert connection._session is not None
        assert connection._session is manager._session
    finally:
        await manager.close()


async def test_a_removed_node_loses_its_channel(tmp_path: Path) -> None:
    manager, registry = build(tmp_path)
    registry.upsert(
        make_node(
            "agent-01",
            execution_mode=ExecutionMode.AGENT_CONTAINER,
            agent_ws_url="ws://127.0.0.1:1/broker",
        )
    )
    try:
        await manager._sync()
        assert "agent-01" in manager._connections
        registry.remove("agent-01")
        await manager._sync()
        assert "agent-01" not in manager._connections
    finally:
        await manager.close()


async def test_a_lost_node_gets_no_channel(tmp_path: Path) -> None:
    manager, registry = build(tmp_path)
    node = make_node(
        "agent-01",
        execution_mode=ExecutionMode.AGENT_CONTAINER,
        agent_ws_url="ws://127.0.0.1:1/broker",
    )
    registry.upsert(node.model_copy(update={"health": NodeHealth.LOST}))
    try:
        await manager._sync()
        assert manager._connections == {}
    finally:
        await manager.close()


async def test_an_ollama_node_gets_no_channel(tmp_path: Path) -> None:
    manager, registry = build(tmp_path)
    registry.upsert(make_node("node-a", execution_mode=ExecutionMode.OLLAMA_DIRECT))
    try:
        await manager._sync()
        assert manager._connections == {}
    finally:
        await manager.close()
