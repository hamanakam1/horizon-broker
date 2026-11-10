"""The node connection manager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

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


async def test_task_creation_failure_closes_run_coroutine_and_rolls_back_channel(
    tmp_path: Path,
) -> None:
    manager, registry = build(tmp_path)
    registry.upsert(
        make_node(
            "agent-01",
            execution_mode=ExecutionMode.AGENT_CONTAINER,
            agent_ws_url="ws://127.0.0.1:1/broker",
        )
    )
    captured = []

    def fail_create_task(coroutine):
        captured.append(coroutine)
        raise RuntimeError("synthetic task creation failure")

    try:
        with patch.object(asyncio, "create_task", side_effect=fail_create_task):
            with pytest.raises(RuntimeError, match="synthetic task creation failure"):
                await manager._sync()
        assert manager._connections == {}
        assert manager._tasks == {}
        assert captured and captured[0].cr_frame is None
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


async def test_a_finished_connection_task_is_reaped(tmp_path: Path) -> None:
    manager, _ = build(tmp_path)

    async def failed_connection() -> None:
        raise RuntimeError("synthetic connection exit")

    task = asyncio.create_task(failed_connection())
    await asyncio.sleep(0)
    manager._connections["agent-01"] = object()  # type: ignore[assignment]
    manager._tasks["agent-01"] = task

    manager._reap_finished_tasks()

    assert "agent-01" not in manager._connections
    assert "agent-01" not in manager._tasks
    await manager.close()


async def test_manager_close_awaits_connection_tasks(tmp_path: Path) -> None:
    manager, _ = build(tmp_path)

    async def running_connection() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(running_connection())
    manager._connections["agent-01"] = object()  # type: ignore[assignment]
    manager._tasks["agent-01"] = task

    class Connection:
        async def close(self) -> None:
            return None

    manager._connections["agent-01"] = Connection()  # type: ignore[assignment]

    await manager.close()

    assert task.done()
    assert manager._connections == {}


async def test_removing_a_node_closes_and_awaits_its_channel(tmp_path: Path) -> None:
    manager, registry = build(tmp_path)
    node = make_node(
        "agent-01",
        execution_mode=ExecutionMode.AGENT_CONTAINER,
        agent_ws_url="ws://127.0.0.1:1/broker",
    )
    registry.upsert(node)
    await manager._sync()
    connection = manager._connections[node.node_id]
    task = manager._tasks[node.node_id]
    closed = False

    async def close() -> None:
        nonlocal closed
        closed = True

    connection.close = close  # type: ignore[method-assign]
    registry.remove(node.node_id)
    await manager._sync()

    assert closed is True
    assert task.done()
    assert node.node_id not in manager._connections
    assert node.node_id not in manager._tasks
    await manager.close()


async def test_channel_close_failure_still_awaits_its_task(tmp_path: Path) -> None:
    manager, _ = build(tmp_path)

    async def running_connection() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(running_connection())

    class FailingConnection:
        async def close(self) -> None:
            raise OSError("synthetic close failure")

    manager._connections["agent-01"] = FailingConnection()  # type: ignore[assignment]
    manager._tasks["agent-01"] = task

    await manager.close()

    assert task.done()
    assert manager._connections == {}
    assert manager._tasks == {}


async def test_manager_close_cleans_up_an_orphaned_task(tmp_path: Path) -> None:
    manager, _ = build(tmp_path)

    async def running_connection() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(running_connection())
    manager._tasks["agent-01"] = task

    await manager.close()

    assert task.done()
    assert manager._tasks == {}


async def test_manager_forgets_owned_session_when_close_fails(tmp_path: Path) -> None:
    manager, _ = build(tmp_path)

    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    manager._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="session close failure"):
        await manager.close()
    assert manager._session is None


async def test_sync_replaces_an_orphaned_task_before_creating_a_channel(tmp_path: Path) -> None:
    manager, registry = build(tmp_path)
    node = make_node(
        "agent-01",
        execution_mode=ExecutionMode.AGENT_CONTAINER,
        agent_ws_url="ws://127.0.0.1:1/broker",
    )
    registry.upsert(node)

    async def orphaned_connection() -> None:
        await asyncio.Event().wait()

    orphan = asyncio.create_task(orphaned_connection())
    manager._tasks[node.node_id] = orphan

    await manager._sync()

    replacement = manager._tasks[node.node_id]
    assert replacement is not orphan
    assert orphan.done()
    await manager.close()
    assert manager._tasks == {}
