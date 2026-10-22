"""The Horizon Broker.

An asynchronous inference queue and fault-tolerance manager for a
heterogeneous edge fleet. See ``docs/ARCHITECTURE.md`` for the subsystem map
and ``docs/CONTRACTS.md`` for the data contracts.
"""

from __future__ import annotations

__version__ = "0.2.0"

from horizon_broker.config import BrokerConfig, load_config
from horizon_broker.contracts import (
    ExecutionPhase,
    FaultRecoverySnapshot,
    InferenceJob,
    InferencePayload,
    JobState,
    NodeCapability,
    NodeHealth,
    NodeRole,
    PriorityTier,
)

__all__ = [
    "BrokerConfig",
    "ExecutionPhase",
    "FaultRecoverySnapshot",
    "InferenceJob",
    "InferencePayload",
    "JobState",
    "NodeCapability",
    "NodeHealth",
    "NodeRole",
    "PriorityTier",
    "__version__",
    "load_config",
]
