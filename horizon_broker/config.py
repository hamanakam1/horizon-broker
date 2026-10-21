"""Broker configuration.

Configuration is layered:

1. defaults in this module,
2. a YAML file (``config/broker.yaml``),
3. environment variables prefixed with ``HORIZON_BROKER_``.

The environment always wins, so a container can be re-pointed at a different
ledger or Ollama endpoint without rewriting a file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from horizon_broker.contracts.nodes import ExecutionMode, NodeCapability, NodeRole, PowerClass

ENV_PREFIX = "HORIZON_BROKER_"


@dataclass(frozen=True)
class HeartbeatConfig:
    """Timing of the node heartbeat and the failure detector."""

    interval_seconds: float = 2.0
    """How often the broker pings a node."""

    timeout_seconds: float = 6.0
    """Silence after which a node is considered unreachable."""

    missed_beats_before_loss: int = 3
    """Consecutive missed heartbeats before a node is declared LOST."""

    reconnect_backoff_seconds: float = 0.5
    reconnect_backoff_max_seconds: float = 30.0


@dataclass(frozen=True)
class WindowConfig:
    """Flex-tier windowing and batching policy."""

    window_seconds: float = 45.0
    """Maximum time a Flex batch is held open while it accumulates jobs."""

    min_batch_size: int = 4
    """Batch size that justifies early dispatch once the dwell time elapsed."""

    max_batch_size: int = 32
    """Hard cap on batch size; a larger queue waits for the next window."""

    min_dwell_seconds: float = 5.0
    """Minimum time the window stays open after the first job arrived."""

    power_headroom_threshold: float = 0.25
    """Solar/solar-trailer nodes only take Flex work above this headroom."""

    power_gated_tiers: tuple[str, ...] = ("solar_trailer", "battery")
    """Power classes that are gated."""


@dataclass(frozen=True)
class SchedulingConfig:
    """Concurrency budgets for the two scheduling lanes."""

    interactive_concurrency: int = 4
    flex_concurrency: int = 8
    interactive_latency_threshold_ms: int = 2_000
    """Above this budget a request is a candidate for the Flex tier."""
    poll_interval_seconds: float = 0.25
    """Idle sleep of the scheduler loops."""


@dataclass(frozen=True)
class RecoveryConfig:
    """Fault-tolerance budgets."""

    max_attempts: int = 3
    recovery_budget_seconds: float = 30.0
    """Hard budget from fault detection to re-dispatch on another node."""
    checkpoint_every_tokens: int = 64
    checkpoint_every_seconds: float = 10.0
    quarantine_seconds: float = 120.0
    """How long a failed node is excluded from placement."""


@dataclass(frozen=True)
class LedgerConfig:
    """EdgeRecord connection settings."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:8787"
    collection: str = "horizon_broker.snapshots"
    timeout_seconds: float = 5.0
    commit_path: str = "/collections/{collection}/records"
    query_path: str = "/collections/{collection}/records"


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama connection settings for direct-mode nodes."""

    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 600.0
    keep_alive: str = "30m"


@dataclass(frozen=True)
class BrokerConfig:
    """Root configuration object."""

    host: str = "127.0.0.1"
    port: int = 8791
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    journal_path: str = "./var/horizon-broker/journal.jsonl"
    static_nodes: tuple[NodeCapability, ...] = ()
    log_level: str = "INFO"

    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.static_nodes)


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value if value not in (None, "") else None


def _coerce(raw: Any, current: Any) -> Any:
    """Coerce an environment string into the type of the current value."""
    if isinstance(current, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(part.strip() for part in str(raw).split(",") if part.strip())
    return raw


def _apply_env_overrides(config: BrokerConfig) -> BrokerConfig:
    """Overlay ``HORIZON_BROKER_*`` environment variables onto a config object."""
    mapping: dict[str, tuple[str, Any]] = {
        "HOST": ("host", config.host),
        "PORT": ("port", config.port),
        "LOG_LEVEL": ("log_level", config.log_level),
        "JOURNAL_PATH": ("journal_path", config.journal_path),
        "OLLAMA_BASE_URL": ("ollama", config.ollama),
        "LEDGER_BASE_URL": ("ledger", config.ledger),
    }
    for suffix, (attribute, current) in mapping.items():
        raw = _env(suffix)
        if raw is None:
            continue
        value = _coerce(raw, current)
        if isinstance(current, OllamaConfig):
            config = replace(config, ollama=replace(current, base_url=value))
        elif isinstance(current, LedgerConfig):
            config = replace(config, ledger=replace(current, base_url=value))
        else:
            config = replace(config, **{attribute: value})

    for suffix, section, attribute, current in (
        ("HEARTBEAT_INTERVAL", config.heartbeat, "interval_seconds", config.heartbeat.interval_seconds),
        ("HEARTBEAT_TIMEOUT", config.heartbeat, "timeout_seconds", config.heartbeat.timeout_seconds),
        ("WINDOW_SECONDS", config.window, "window_seconds", config.window.window_seconds),
        ("WINDOW_MIN_BATCH", config.window, "min_batch_size", config.window.min_batch_size),
        ("WINDOW_MAX_BATCH", config.window, "max_batch_size", config.window.max_batch_size),
        ("RECOVERY_BUDGET", config.recovery, "recovery_budget_seconds", config.recovery.recovery_budget_seconds),
        ("MAX_ATTEMPTS", config.recovery, "max_attempts", config.recovery.max_attempts),
    ):
        raw = _env(suffix)
        if raw is None:
            continue
        value = _coerce(raw, current)
        if suffix.startswith("HEARTBEAT"):
            config = replace(config, heartbeat=replace(config.heartbeat, **{attribute: value}))
        elif suffix.startswith("WINDOW"):
            config = replace(config, window=replace(config.window, **{attribute: value}))
        else:
            config = replace(config, recovery=replace(config.recovery, **{attribute: value}))
    return config


def _node_from_mapping(raw: dict[str, Any]) -> NodeCapability:
    """Build a :class:`NodeCapability` from a YAML/JSON mapping."""
    data = dict(raw)
    if "role" in data:
        data["role"] = NodeRole(str(data["role"]))
    if "power_class" in data:
        data["power_class"] = PowerClass(str(data["power_class"]))
    if "execution_mode" in data:
        data["execution_mode"] = ExecutionMode(str(data["execution_mode"]))
    if "model_whitelist" in data:
        data["model_whitelist"] = tuple(data["model_whitelist"])
    return NodeCapability.model_validate(data)


def load_config(path: str | Path | None = None) -> BrokerConfig:
    """Build a :class:`BrokerConfig` from an optional YAML file plus the environment."""
    data: dict[str, Any] = {}
    config_path = Path(path) if path else None
    if config_path is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration file {config_path} must contain a mapping")
        data = loaded

    config = BrokerConfig()

    heartbeat = data.get("heartbeat")
    if isinstance(heartbeat, dict):
        config = replace(config, heartbeat=replace(config.heartbeat, **heartbeat))

    window = data.get("window")
    if isinstance(window, dict):
        config = replace(config, window=replace(config.window, **window))

    scheduling = data.get("scheduling")
    if isinstance(scheduling, dict):
        config = replace(config, scheduling=replace(config.scheduling, **scheduling))

    recovery = data.get("recovery")
    if isinstance(recovery, dict):
        config = replace(config, recovery=replace(config.recovery, **recovery))

    ledger = data.get("ledger")
    if isinstance(ledger, dict):
        config = replace(config, ledger=replace(config.ledger, **ledger))

    ollama = data.get("ollama")
    if isinstance(ollama, dict):
        config = replace(config, ollama=replace(config.ollama, **ollama))

    broker = data.get("broker")
    if isinstance(broker, dict):
        config = replace(
            config,
            host=broker.get("host", config.host),
            port=int(broker.get("port", config.port)),
            log_level=broker.get("log_level", config.log_level),
            journal_path=broker.get("journal_path", config.journal_path),
        )

    raw_nodes = data.get("nodes") or []
    if raw_nodes:
        config = replace(
            config, static_nodes=tuple(_node_from_mapping(item) for item in raw_nodes)
        )

    return _apply_env_overrides(config)


def dump_config(config: BrokerConfig) -> str:
    """Render the effective configuration as JSON (used by the diagnostics endpoint)."""
    return json.dumps(
        {
            "host": config.host,
            "port": config.port,
            "log_level": config.log_level,
            "journal_path": config.journal_path,
            "heartbeat": config.heartbeat.__dict__,
            "window": config.window.__dict__,
            "scheduling": config.scheduling.__dict__,
            "recovery": config.recovery.__dict__,
            "ledger": config.ledger.__dict__,
            "ollama": config.ollama.__dict__,
            "static_nodes": [json.loads(node.model_dump_json()) for node in config.static_nodes],
        },
        indent=2,
        sort_keys=True,
    )
