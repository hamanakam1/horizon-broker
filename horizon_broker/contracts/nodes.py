"""Node capability contracts.

A :class:`NodeCapability` is the broker's view of one execution target: its
role in the PD-DAG, its memory envelope, its current batch occupancy and the
power headroom that gates sleep-time compute on the solar trailer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from horizon_broker.contracts.jobs import BrokerModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class NodeRole(StrEnum):
    """Where the node sits in the prefill/decode split."""

    PREFILL = "prefill"
    DECODE = "decode"
    GENERAL = "general"


class ExecutionMode(StrEnum):
    """How the broker drives work on the node."""

    OLLAMA_DIRECT = "ollama_direct"
    """The broker speaks Ollama's HTTP API directly against the node."""

    AGENT_CONTAINER = "agent_container"
    """The broker assigns work over the node's WebSocket control channel."""


class PowerClass(StrEnum):
    """Power envelope of the node."""

    GRID = "grid"
    """Mains powered; the M3 Max in the home rack."""

    SOLAR_TRAILER = "solar_trailer"
    """Off-grid trailer; sleep-time work only runs with surplus charge."""

    BATTERY = "battery"
    """Running on battery; drain-aware scheduling applies."""


class NodeHealth(StrEnum):
    """Health as judged by the heartbeat watchdog."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DRAINING = "draining"
    """Announced shutdown (e.g. low voltage). Work must be migrated off the node."""
    SUSPECT = "suspect"
    """Missed heartbeats; not yet declared lost."""
    LOST = "lost"
    OFFLINE = "offline"


class NodeCapability(BrokerModel):
    """Capability profile of one execution node."""

    node_id: str
    role: NodeRole = NodeRole.GENERAL
    is_prefill_optimized: bool = Field(
        default=False, description="Compute-bound: fast FLOPs, wide memory, ideal for large prefills."
    )
    is_decode_optimized: bool = Field(
        default=False, description="Memory-bandwidth-bound: ideal for long token emission."
    )
    available_vram_bytes: int = Field(ge=0, description="Free memory usable right now.")
    total_vram_bytes: int = Field(ge=0)
    current_batch_size: int = Field(default=0, ge=0, description="Jobs already in flight on the node.")
    max_batch_size: int = Field(default=1, ge=1)
    model_whitelist: tuple[str, ...] = Field(
        default=(), description="Models the node can serve; empty means unrestricted."
    )
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    power_class: PowerClass = PowerClass.GRID
    power_headroom: float = Field(
        default=1.0, ge=0.0, le=1.0, description="0.0 = flat battery, 1.0 = full surplus."
    )
    health: NodeHealth = NodeHealth.HEALTHY
    execution_mode: ExecutionMode = ExecutionMode.OLLAMA_DIRECT
    endpoint: str | None = Field(default=None, description="Base HTTP endpoint (Ollama).")
    agent_ws_url: str | None = Field(default=None, description="WebSocket control channel (agent container).")
    ollama_url: str | None = Field(default=None, description="Ollama URL when execution_mode is OLLAMA_DIRECT.")
    assigned_job_ids: tuple[str, ...] = ()
    last_heartbeat_at: datetime | None = None
    quarantined_until: datetime | None = Field(
        default=None, description="Node is excluded from placement until this instant."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _check_envelope(cls, data: Any) -> Any:
        """Reject impossible memory/batch envelopes before the model is built."""
        if isinstance(data, dict):
            total = data.get("total_vram_bytes")
            available = data.get("available_vram_bytes")
            if isinstance(total, int) and isinstance(available, int) and available > total:
                data = {**data, "available_vram_bytes": total}
            max_batch = data.get("max_batch_size")
            current_batch = data.get("current_batch_size")
            if (
                isinstance(max_batch, int)
                and isinstance(current_batch, int)
                and current_batch > max_batch
            ):
                data = {**data, "current_batch_size": max_batch}
        return data

    @property
    def vram_headroom_ratio(self) -> float:
        """Fraction of memory still free (0.0 - 1.0)."""
        if self.total_vram_bytes <= 0:
            return 0.0
        return max(0.0, min(1.0, self.available_vram_bytes / self.total_vram_bytes))

    @property
    def batch_headroom(self) -> int:
        """Number of additional jobs the node can accept."""
        return max(0, self.max_batch_size - self.current_batch_size)

    def supports_model(self, model: str) -> bool:
        """True when the node is allowed to serve ``model``."""
        return not self.model_whitelist or model in self.model_whitelist

    def accepts_work(self, model: str) -> bool:
        """True when the node is healthy, unquarantined and has capacity."""
        if self.health in (NodeHealth.LOST, NodeHealth.OFFLINE, NodeHealth.DRAINING):
            return False
        if self.batch_headroom <= 0:
            return False
        return self.supports_model(model)
