"""Node control channel.

The broker keeps one WebSocket per agent-container node. The channel carries
four frame families:

``ping`` / ``heartbeat``
    Liveness. The broker pings every ``interval_seconds``; the node answers
    with a ``heartbeat`` frame carrying its current capability snapshot.

``assign`` / ``batch_assign`` / ``cancel``
    Work. The broker sends ``assign`` for one job or ``batch_assign`` for an
    explicitly batch-capable agent container. Each item carries its job
    envelope and resume state; the node answers with
    ``progress``/``checkpoint``/``complete`` keyed by ``job_id``.

``checkpoint``
    Recovery state: decode step, partial tokens and the KV-cache reference.

``draining``
    A polite announcement that the node is about to lose power (solar trailer
    on a low-voltage shutdown). The broker migrates the work immediately.

Any socket close or silence longer than ``timeout_seconds`` is reported as a
node loss so the watchdog can start the recovery sequence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NoReturn

import aiohttp

from horizon_broker.config import HeartbeatConfig
from horizon_broker.contracts.errors import NodeUnreachableError

LOGGER = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


class NodeEventType(StrEnum):
    """Frame types exchanged with a node."""

    HELLO = "hello"
    HEARTBEAT = "heartbeat"
    PONG = "pong"
    PROGRESS = "progress"
    CHECKPOINT = "checkpoint"
    COMPLETE = "complete"
    ERROR = "error"
    DRAINING = "draining"


@dataclass(frozen=True)
class NodeEvent:
    """A decoded frame received from a node."""

    node_id: str
    type: NodeEventType
    data: dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utcnow)


EventSink = Callable[[NodeEvent], Awaitable[None]]


class NodeConnection:
    """A resilient WebSocket client for one node."""

    def __init__(
        self,
        node_id: str,
        url: str,
        *,
        on_event: EventSink,
        config: HeartbeatConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.node_id = node_id
        self.url = url
        self.config = config or HeartbeatConfig()
        self._on_event = on_event
        self._session = session
        self._owns_session = session is None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closing = False
        self._connected = False
        self.last_frame_at: datetime | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the socket and the session (when owned)."""
        self._closing = True
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
        finally:
            self._connected = False
            if self._session is not None and self._owns_session:
                session = self._session
                try:
                    await session.close()
                finally:
                    # A failed close must not leave a poisoned session
                    # available to a later reconnect or shutdown attempt.
                    self._session = None

    async def send(self, frame: dict[str, Any]) -> None:
        """Send a JSON frame to the node."""
        if self._ws is None or self._ws.closed:
            raise NodeUnreachableError(f"node {self.node_id} is not connected")
        await self._ws.send_str(
            json.dumps(frame, separators=(",", ":"), allow_nan=False)
        )

    async def run(self) -> None:
        """Connect, serve frames, and reconnect with exponential backoff."""
        backoff = self.config.reconnect_backoff_seconds
        while not self._closing:
            try:
                await self._serve()
                backoff = self.config.reconnect_backoff_seconds
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, NodeUnreachableError, OSError, TimeoutError) as exc:
                if self._closing:
                    break
                LOGGER.warning("node %s connection lost: %s", self.node_id, exc)
                await self._emit(NodeEventType.ERROR, {"kind": "unreachable", "error": str(exc)})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.config.reconnect_backoff_max_seconds)

    async def _serve(self) -> None:
        """Open the socket and pump frames until it dies."""
        session = await self._get_session()
        async with session.ws_connect(self.url, heartbeat=None) as ws:
            self._ws = ws
            self._connected = True
            LOGGER.info("node %s connected", self.node_id)
            try:
                await self.send({"type": "ping", "ts": utcnow().isoformat()})
            except BaseException:
                # The receive-loop cleanup below has not been entered yet. A
                # failed initial ping must still leave reconnect and health
                # reporting with an honest channel state.
                self._connected = False
                self._ws = None
                raise
            ping = self._ping_loop()
            try:
                ping_task = asyncio.create_task(ping)
            except BaseException:
                ping.close()
                self._connected = False
                self._ws = None
                raise
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(
                            ws.receive(), timeout=self.config.timeout_seconds
                        )
                    except TimeoutError as exc:
                        raise NodeUnreachableError(
                            f"node {self.node_id} silent for {self.config.timeout_seconds}s"
                        ) from exc
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await self._handle(message.data)
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        raise NodeUnreachableError(f"node {self.node_id} closed the socket")
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        raise NodeUnreachableError(
                            f"node {self.node_id} socket error: {ws.exception()}"
                        )
            finally:
                ping_task.cancel()
                await asyncio.gather(ping_task, return_exceptions=True)
                self._connected = False
                self._ws = None

    async def _ping_loop(self) -> None:
        """Send periodic pings so a dead peer is noticed quickly."""
        try:
            while True:
                await asyncio.sleep(self.config.interval_seconds)
                if self._ws is None or self._ws.closed:
                    return
                await self.send({"type": "ping", "ts": utcnow().isoformat()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("ping to node %s failed: %s", self.node_id, exc)

    async def _handle(self, raw: str) -> None:
        """Decode one frame and forward it to the event sink."""
        try:
            frame = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            LOGGER.warning("node %s sent a malformed frame", self.node_id)
            return
        if not isinstance(frame, dict):
            return
        frame_type = frame.get("type")
        if not isinstance(frame_type, str):
            LOGGER.warning("node %s sent a frame without a valid type", self.node_id)
            return
        try:
            event_type = NodeEventType(frame_type)
        except ValueError:
            LOGGER.warning("node %s sent an unknown frame type %s", self.node_id, frame_type)
            return
        self.last_frame_at = utcnow()
        data = frame.get("data")
        if not isinstance(data, dict):
            data = {
                key: value
                for key, value in frame.items()
                if key not in ("type", "ts")
            }
        try:
            await self._emit(event_type, data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad event must not kill the channel
            LOGGER.exception("node %s event handler failed: %s", self.node_id, exc)

    async def _emit(self, event_type: NodeEventType, data: dict[str, Any]) -> None:
        await self._on_event(
            NodeEvent(node_id=self.node_id, type=event_type, data=data, received_at=utcnow())
        )


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")
