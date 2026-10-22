"""Arithmetic intensity model.

The router decides *where* a job runs by looking at the shape of the work:

* **Prefill** (prompt ingestion) is compute-bound. Every token of a large
  context window costs FLOPs, and the arithmetic intensity (FLOPs per byte
  moved) is high.
* **Decode** (token emission) is memory-bandwidth-bound. Each step streams the
  whole weight matrix through the memory subsystem, so the arithmetic
  intensity is roughly ``batch_size`` FLOPs per byte.

This module turns an :class:`~horizon_broker.contracts.jobs.InferencePayload`
into those numbers. The model constants are deliberately coarse: they are used
for *placement decisions*, not for capacity planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from horizon_broker.contracts.jobs import InferencePayload

BYTES_PER_PARAM_BF16 = 2
"""Weight bytes per parameter for a bf16 checkpoint."""


@dataclass(frozen=True)
class RoutingThresholds:
    """The token and intensity thresholds that separate the compute phases."""

    prefill_prompt_threshold: int = 4_096
    decode_output_threshold: int = 2_048
    prefill_ratio: float = 4.0
    decode_ratio: float = 0.25


class ComputePhase(StrEnum):
    """Dominant phase of a job, from an arithmetic-intensity point of view."""

    PREFILL_BOUND = "prefill_bound"
    DECODE_BOUND = "decode_bound"
    BALANCED = "balanced"


@dataclass(frozen=True)
class ModelProfile:
    """Coarse hardware/model constants used by the router."""

    name: str
    parameters_b: float = 7.0
    """Parameter count in billions."""
    hidden_size: int = 4096
    kv_bytes_per_token: int = 131_072
    """KV cache bytes per token (128 KiB is a reasonable f32 estimate for 7B-class models)."""
    memory_bandwidth_gbps: float = 400.0
    peak_tflops: float = 10.0
    supports_kv_handoff: bool = False
    """True when the engine can hand a KV cache to another node."""

    @property
    def weight_bytes(self) -> int:
        """Resident weight footprint in bytes."""
        return int(self.parameters_b * 1e9 * BYTES_PER_PARAM_BF16)

    def prefill_flops(self, tokens: int) -> float:
        """FLOPs to ingest ``tokens`` tokens of context.

        Projection terms dominate: ``2 * P * tokens``. The attention term adds
        ``4 * L^2 * d`` which is what makes very long prompts superlinear.
        """
        tokens = max(0, tokens)
        projection = 2.0 * self.parameters_b * 1e9 * tokens
        attention = 4.0 * tokens * tokens * self.hidden_size
        return projection + attention

    def prefill_bytes(self, tokens: int) -> float:
        """Bytes moved during prefill: KV writes plus the prompt itself."""
        tokens = max(0, tokens)
        return tokens * (self.kv_bytes_per_token + 4)

    def decode_flops(self, tokens: int) -> float:
        """FLOPs to emit ``tokens`` tokens."""
        tokens = max(0, tokens)
        return 2.0 * self.parameters_b * 1e9 * tokens

    def decode_bytes(self, tokens: int, batch_size: int = 1) -> float:
        """Bytes moved during decode for ``tokens`` steps at ``batch_size`` concurrency.

        The weight matrix dominates and is amortised across the batch, which is
        precisely why batching raises MFU.
        """
        tokens = max(0, tokens)
        batch = max(1, batch_size)
        weights = self.weight_bytes / batch
        kv = tokens * self.kv_bytes_per_token
        return weights * tokens + kv


@dataclass(frozen=True)
class ArithmeticProfile:
    """The arithmetic characterisation of one job."""

    model: str
    prompt_tokens: int
    output_tokens: int
    prefill_flops: float
    prefill_bytes: float
    decode_flops: float
    decode_bytes: float
    phase: ComputePhase
    intensity_ratio: float
    """prefill FLOPs divided by decode FLOPs."""
    supports_kv_handoff: bool = False
    """True when the engine can hand a KV cache to another node."""

    @property
    def prefill_intensity(self) -> float:
        """FLOPs per byte during prefill (high => compute bound)."""
        if self.prefill_bytes <= 0:
            return float("inf")
        return self.prefill_flops / self.prefill_bytes

    @property
    def decode_intensity(self) -> float:
        """FLOPs per byte during decode (low => memory bound)."""
        if self.decode_bytes <= 0:
            return float("inf")
        return self.decode_flops / self.decode_bytes

    @property
    def estimated_memory_bytes(self) -> int:
        """Resident memory needed to serve the job: weights + KV for the whole context."""
        return int(self.prefill_bytes + self.decode_bytes)

    def summary(self) -> str:
        """One-line human readable characterisation."""
        return (
            f"{self.phase.value}: prompt={self.prompt_tokens}t output={self.output_tokens}t "
            f"prefill_intensity={self.prefill_intensity:.1f} "
            f"decode_intensity={self.decode_intensity:.1f}"
        )


DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "qwen2.5:7b": ModelProfile(
        name="qwen2.5:7b",
        parameters_b=7.6,
        hidden_size=3584,
        kv_bytes_per_token=98_304,
        memory_bandwidth_gbps=400.0,
        peak_tflops=15.0,
    ),
    "qwen2.5:14b": ModelProfile(
        name="qwen2.5:14b",
        parameters_b=14.8,
        hidden_size=5120,
        kv_bytes_per_token=163_840,
        memory_bandwidth_gbps=400.0,
        peak_tflops=15.0,
    ),
    "qwen2.5:32b": ModelProfile(
        name="qwen2.5:32b",
        parameters_b=32.8,
        hidden_size=5120,
        kv_bytes_per_token=163_840,
        memory_bandwidth_gbps=400.0,
        peak_tflops=15.0,
    ),
    "qwen3:8b": ModelProfile(
        name="qwen3:8b",
        parameters_b=8.2,
        hidden_size=4096,
        kv_bytes_per_token=131_072,
        memory_bandwidth_gbps=400.0,
        peak_tflops=15.0,
    ),
    "qwen3:30b-a3b": ModelProfile(
        name="qwen3:30b-a3b",
        parameters_b=30.5,
        hidden_size=2048,
        kv_bytes_per_token=65_536,
        memory_bandwidth_gbps=400.0,
        peak_tflops=15.0,
    ),
}


def profile_for(model: str) -> ModelProfile:
    """Return the profile for ``model``, falling back to a generic 7B estimate."""
    if model in DEFAULT_PROFILES:
        return DEFAULT_PROFILES[model]
    for name, profile in DEFAULT_PROFILES.items():
        if model.startswith(name.split(":")[0]):
            return profile
    return DEFAULT_PROFILES["qwen2.5:7b"]


def analyse(
    payload: InferencePayload,
    batch_size: int = 1,
    thresholds: RoutingThresholds | None = None,
) -> ArithmeticProfile:
    """Characterise the arithmetic shape of ``payload``."""
    profile = profile_for(payload.model)
    prompt_tokens = payload.estimated_prompt_tokens
    output_tokens = payload.estimated_output_tokens

    prefill_flops = profile.prefill_flops(prompt_tokens)
    prefill_bytes = profile.prefill_bytes(prompt_tokens)
    decode_flops = profile.decode_flops(output_tokens)
    decode_bytes = profile.decode_bytes(output_tokens, batch_size=batch_size)

    intensity_ratio = (prefill_flops / decode_flops) if decode_flops > 0 else float("inf")
    phase = _classify(prompt_tokens, output_tokens, intensity_ratio, thresholds)
    return ArithmeticProfile(
        model=payload.model,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefill_flops=prefill_flops,
        prefill_bytes=prefill_bytes,
        decode_flops=decode_flops,
        decode_bytes=decode_bytes,
        phase=phase,
        intensity_ratio=intensity_ratio,
        supports_kv_handoff=profile.supports_kv_handoff,
    )


def _classify(
    prompt_tokens: int,
    output_tokens: int,
    intensity_ratio: float,
    thresholds: RoutingThresholds | None = None,
) -> ComputePhase:
    """Classify the dominant phase from token counts and the FLOP ratio."""
    limits = thresholds or RoutingThresholds()
    if prompt_tokens >= limits.prefill_prompt_threshold and intensity_ratio >= limits.prefill_ratio:
        return ComputePhase.PREFILL_BOUND
    if output_tokens >= limits.decode_output_threshold and intensity_ratio <= limits.decode_ratio:
        return ComputePhase.DECODE_BOUND
    if prompt_tokens >= 2 * limits.prefill_prompt_threshold and output_tokens >= 2 * limits.decode_output_threshold:
        # Both phases are heavy: a split is only worth it when the fleet has
        # dedicated prefill and decode nodes.
        return ComputePhase.PREFILL_BOUND
    return ComputePhase.BALANCED
