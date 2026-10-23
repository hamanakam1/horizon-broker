"""Node control-channel frame decoding and lifecycle boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import aiohttp
import pytest

from horizon_broker.config import HeartbeatConfig
from horizon_broker.contracts.errors import NodeUnreachableError
from horizon_broker.nodes.connection import NodeConnection, NodeEventType


async def test_malformed_node_frames_are_ignored() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle("not-json")
    await connection._handle("[1, 2, 3]")

    assert events == []
    assert connection.last_frame_at is None


async def test_unknown_node_frame_types_do_not_refresh_liveness() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle('{"type":"future-event","data":{"value":7}}')

    assert events == []
    assert connection.last_frame_at is None


async def test_non_string_node_frame_types_do_not_refresh_liveness() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle('{"type":1,"data":{"value":7}}')

    assert events == []
    assert connection.last_frame_at is None


async def test_non_standard_json_numbers_do_not_refresh_liveness() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle('{"type":"heartbeat","power_headroom":NaN}')

    assert events == []
    assert connection.last_frame_at is None


async def test_node_frames_without_nested_data_are_flattened() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle('{"type":"heartbeat","ts":"synthetic","load":0.5}')

    assert len(events) == 1
    assert events[0].type is NodeEventType.HEARTBEAT
    assert events[0].data == {"load": 0.5}
    assert connection.last_frame_at is not None


async def test_a_broken_event_handler_does_not_kill_frame_decoding() -> None:
    events = []
    calls = 0

    async def on_event(event) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic handler failure")
        events.append(event)

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    await connection._handle('{"type":"progress","job_id":"job-1"}')
    await connection._handle('{"type":"complete","job_id":"job-1"}')

    assert calls == 2
    assert len(events) == 1
    assert events[0].type is NodeEventType.COMPLETE


async def test_sending_before_connection_is_a_typed_failure() -> None:
    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)

    with pytest.raises(NodeUnreachableError, match="not connected"):
        await connection.send({"type": "assign"})


async def test_socket_close_cleans_up_the_ping_task_and_connection_state() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            return None

        async def receive(self) -> SimpleNamespace:
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)

        def exception(self) -> None:
            return None

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

    class Session:
        def __init__(self) -> None:
            self.socket = Socket()

        def ws_connect(self, url: str, *, heartbeat: None) -> Socket:
            return self.socket

    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://synthetic", on_event=on_event)
    connection._session = Session()  # type: ignore[assignment]

    with pytest.raises(NodeUnreachableError, match="closed the socket"):
        await connection._serve()

    assert connection.connected is False
    assert connection._ws is None


async def test_ping_failure_is_contained_by_the_ping_loop() -> None:
    async def on_event(event) -> None:
        return None

    connection = NodeConnection(
        "node-1",
        "ws://synthetic",
        on_event=on_event,
        config=replace(HeartbeatConfig(), interval_seconds=0.001),
    )

    async def fail_send(frame: dict[str, object]) -> None:
        raise RuntimeError("synthetic ping failure")

    connection.send = fail_send  # type: ignore[method-assign]
    connection._ws = SimpleNamespace(closed=False)  # type: ignore[assignment]

    await connection._ping_loop()
    assert connection._connected is False


async def test_run_emits_transport_error_and_stops_after_close_request() -> None:
    events = []

    async def on_event(event) -> None:
        events.append(event)
        connection._closing = True

    class Session:
        def ws_connect(self, url: str, *, heartbeat: None):
            raise aiohttp.ClientConnectionError("synthetic socket failure")

    connection = NodeConnection(
        "node-1",
        "ws://synthetic",
        on_event=on_event,
        config=replace(
            HeartbeatConfig(),
            reconnect_backoff_seconds=0.001,
            reconnect_backoff_max_seconds=0.001,
        ),
    )
    connection._session = Session()  # type: ignore[assignment]

    await connection.run()

    assert len(events) == 1
    assert events[0].type is NodeEventType.ERROR
    assert events[0].data["kind"] == "unreachable"


async def test_ping_loop_returns_when_socket_is_already_closed() -> None:
    async def on_event(event) -> None:
        return None

    connection = NodeConnection(
        "node-1",
        "ws://synthetic",
        on_event=on_event,
        config=replace(HeartbeatConfig(), interval_seconds=0.001),
    )

    async def fail_send(frame: dict[str, object]) -> None:
        raise AssertionError("closed sockets must not be pinged")

    connection.send = fail_send  # type: ignore[method-assign]
    connection._ws = SimpleNamespace(closed=True)  # type: ignore[assignment]

    await connection._ping_loop()


async def test_silent_socket_becomes_a_typed_unreachable_failure() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            return None

        async def receive(self) -> SimpleNamespace:
            await asyncio.sleep(1)
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{}")

        def exception(self) -> None:
            return None

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

    class Session:
        def ws_connect(self, url: str, *, heartbeat: None) -> Socket:
            return Socket()

    async def on_event(event) -> None:
        return None

    connection = NodeConnection(
        "node-1",
        "ws://synthetic",
        on_event=on_event,
        config=replace(HeartbeatConfig(), timeout_seconds=0.001),
    )
    connection._session = Session()  # type: ignore[assignment]

    with pytest.raises(NodeUnreachableError, match="silent"):
        await connection._serve()
    assert connection.connected is False
    assert connection._ws is None


async def test_socket_error_message_becomes_a_typed_unreachable_failure() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            return None

        async def receive(self) -> SimpleNamespace:
            return SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None)

        def exception(self) -> RuntimeError:
            return RuntimeError("synthetic peer error")

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

    class Session:
        def ws_connect(self, url: str, *, heartbeat: None) -> Socket:
            return Socket()

    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://synthetic", on_event=on_event)
    connection._session = Session()  # type: ignore[assignment]

    with pytest.raises(NodeUnreachableError, match="socket error"):
        await connection._serve()
    assert connection.connected is False
    assert connection._ws is None


async def test_ping_task_creation_failure_closes_ping_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            return None

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

    class Session:
        def ws_connect(self, url: str, *, heartbeat: None) -> Socket:
            return Socket()

    async def on_event(event) -> None:
        return None

    captured = []

    def fail_create_task(coroutine):
        captured.append(coroutine)
        raise RuntimeError("synthetic task creation failure")

    connection = NodeConnection("node-1", "ws://synthetic", on_event=on_event)
    connection._session = Session()  # type: ignore[assignment]
    monkeypatch.setattr(asyncio, "create_task", fail_create_task)

    with pytest.raises(RuntimeError, match="task creation failure"):
        await connection._serve()
    assert captured and captured[0].cr_frame is None
    assert connection.connected is False
    assert connection._ws is None


async def test_initial_ping_failure_clears_connection_state() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            raise RuntimeError("synthetic initial ping failure")

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

    class Session:
        def ws_connect(self, url: str, *, heartbeat: None) -> Socket:
            return Socket()

    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://synthetic", on_event=on_event)
    connection._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="initial ping failure"):
        await connection._serve()
    assert connection.connected is False
    assert connection._ws is None


async def test_sending_a_non_standard_json_number_is_rejected() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            raise AssertionError("invalid frame should not be sent")

    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)
    connection._ws = Socket()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Out of range float values"):
        await connection.send({"type": "assign", "value": float("nan")})


async def test_close_releases_owned_session_when_socket_close_fails() -> None:
    class FailingSocket:
        closed = False

        async def close(self) -> None:
            raise RuntimeError("synthetic socket close failure")

    class Session:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def on_event(event) -> None:
        return None

    session = Session()
    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)
    connection._session = session  # type: ignore[assignment]
    connection._ws = FailingSocket()  # type: ignore[assignment]
    connection._connected = True

    with pytest.raises(RuntimeError, match="socket close failure"):
        await connection.close()

    assert connection._connected is False
    assert session.closed is True
    assert connection._session is None


async def test_close_forgets_owned_session_when_session_close_fails() -> None:
    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    async def on_event(event) -> None:
        return None

    connection = NodeConnection("node-1", "ws://127.0.0.1:1", on_event=on_event)
    connection._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="session close failure"):
        await connection.close()
    assert connection._session is None
