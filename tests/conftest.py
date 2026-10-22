"""Shared fixtures and builders for the Horizon Broker test suite."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from horizon_broker.config import (
    BrokerConfig,
    LedgerConfig,
    RecoveryConfig,
    SchedulingConfig,
    WindowConfig,
)
from horizon_broker.contracts.jobs import (
    InferenceJob,
    InferencePayload,
    PriorityTier,
    SamplingParams,
)
from horizon_broker.contracts.nodes import (
    ExecutionMode,
    NodeCapability,
    NodeRole,
    PowerClass,
)


def make_payload(
    model: str = "qwen2.5:7b",
    prompt: str = "hello",
    *,
    output_tokens: int | None = 64,
    prompt_tokens: int | None = None,
) -> InferencePayload:
    """Build an inference payload."""
    return InferencePayload(
        model=model,
        prompt=prompt,
        expected_output_tokens=output_tokens,
        prompt_tokens_hint=prompt_tokens,
        parameters=SamplingParams(num_predict=output_tokens),
    )


def make_job(
    *,
    tier: PriorityTier = PriorityTier.INTERACTIVE,
    model: str = "qwen2.5:7b",
    prompt: str = "hello",
    output_tokens: int | None = 64,
    prompt_tokens: int | None = None,
    max_latency_ms: int = 1_000,
    **overrides: Any,
) -> InferenceJob:
    """Build an inference job."""
    return InferenceJob(
        priority_tier=tier,
        payload=make_payload(
            model=model, prompt=prompt, output_tokens=output_tokens, prompt_tokens=prompt_tokens
        ),
        max_acceptable_latency_ms=max_latency_ms,
        **overrides,
    )


def make_node(
    node_id: str = "node-a",
    *,
    role: NodeRole = NodeRole.GENERAL,
    prefill: bool = False,
    decode: bool = False,
    total_vram: int = 16_000_000_000,
    available_vram: int = 16_000_000_000,
    max_batch: int = 4,
    current_batch: int = 0,
    models: tuple[str, ...] = ("qwen2.5:7b",),
    power_class: PowerClass = PowerClass.GRID,
    power_headroom: float = 1.0,
    execution_mode: ExecutionMode = ExecutionMode.OLLAMA_DIRECT,
    ollama_url: str | None = None,
    agent_ws_url: str | None = None,
    tokens_per_second: float | None = 20.0,
) -> NodeCapability:
    """Build a node capability profile."""
    return NodeCapability(
        node_id=node_id,
        role=role,
        is_prefill_optimized=prefill,
        is_decode_optimized=decode,
        available_vram_bytes=available_vram,
        total_vram_bytes=total_vram,
        current_batch_size=current_batch,
        max_batch_size=max_batch,
        model_whitelist=models,
        tokens_per_second=tokens_per_second,
        power_class=power_class,
        power_headroom=power_headroom,
        execution_mode=execution_mode,
        ollama_url=ollama_url,
        agent_ws_url=agent_ws_url,
    )


@pytest.fixture
def config(tmp_path) -> BrokerConfig:
    """A broker configuration that never touches the network."""
    base = BrokerConfig()
    return replace(
        base,
        host="127.0.0.1",
        port=0,
        journal_path=str(tmp_path / "journal.jsonl"),
        ledger=LedgerConfig(enabled=False, base_url="http://127.0.0.1:1"),
        window=WindowConfig(
            window_seconds=0.2,
            min_batch_size=2,
            max_batch_size=4,
            min_dwell_seconds=0.05,
            power_headroom_threshold=0.25,
            idle_flush=True,
            idle_flush_seconds=0.05,
        ),
        scheduling=SchedulingConfig(
            interactive_concurrency=2,
            flex_concurrency=2,
            interactive_latency_threshold_ms=2_000,
            poll_interval_seconds=0.02,
        ),
        recovery=RecoveryConfig(
            max_attempts=3,
            recovery_budget_seconds=30.0,
            checkpoint_every_tokens=8,
            checkpoint_every_seconds=0.05,
            quarantine_seconds=60.0,
        ),
    )


async def wait_for(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.02,
) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()
