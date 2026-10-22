"""Node placement.

Given the arithmetic character of a job, pick the node that should run it.
Scoring favours role affinity, idle memory, free batch slots and (for sleep-time
compute) power headroom.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from horizon_broker.contracts.nodes import NodeCapability, NodeHealth, NodeRole, PowerClass
from horizon_broker.routing.arithmetic import ArithmeticProfile, ModelProfile


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class PlacementScore:
    """Why a node was (or was not) chosen."""

    node_id: str
    total: float
    reasons: tuple[str, ...]


class NodeSelector:
    """Selects an execution node for a job.

    The selector is deliberately pure: it never mutates the registry and never
    performs I/O, which keeps routing decisions testable and reproducible.
    """

    def __init__(
        self,
        *,
        power_gated_classes: tuple[PowerClass, ...] = (),
        power_headroom_threshold: float = 0.25,
    ) -> None:
        self.power_gated_classes = power_gated_classes
        self.power_headroom_threshold = power_headroom_threshold

    def eligible(
        self,
        nodes: list[NodeCapability],
        model: str,
        *,
        required_role: NodeRole | None = None,
        excluded: frozenset[str] = frozenset(),
        require_memory_bytes: int = 0,
        flex_tier: bool = False,
        now: datetime | None = None,
    ) -> list[NodeCapability]:
        """Filter the fleet down to nodes that may accept the job."""
        current = now or utcnow()
        candidates: list[NodeCapability] = []
        for node in nodes:
            if node.node_id in excluded:
                continue
            if node.health in (NodeHealth.LOST, NodeHealth.OFFLINE):
                continue
            # The quarantine is a cooling-off period: once it has expired the
            # node is part of the fleet again.
            if node.quarantined_until is not None and node.quarantined_until > current:
                continue
            if not node.supports_model(model):
                continue
            if node.batch_headroom <= 0:
                continue
            if require_memory_bytes and node.available_vram_bytes < require_memory_bytes:
                continue
            if required_role is not None and node.role is not required_role:
                continue
            if flex_tier and self._power_gated(node):
                # Sleep-time work waits for the sun; interactive work does not.
                if node.power_headroom < self.power_headroom_threshold:
                    continue
            candidates.append(node)
        return candidates

    def _power_gated(self, node: NodeCapability) -> bool:
        """True when this node's power class is gated for sleep-time work."""
        return node.power_class in self.power_gated_classes

    def score(
        self,
        node: NodeCapability,
        profile: ArithmeticProfile,
        *,
        flex_tier: bool,
        model_profile: ModelProfile,
    ) -> PlacementScore:
        """Score a candidate node for a job with a given arithmetic profile."""
        reasons: list[str] = []
        total = 0.0

        if profile.phase.value == "prefill_bound" and node.is_prefill_optimized:
            total += 12.0
            reasons.append("prefill-optimised")
        if profile.phase.value == "decode_bound" and node.is_decode_optimized:
            total += 12.0
            reasons.append("decode-optimised")
        if node.role is NodeRole.GENERAL:
            total += 2.0
            reasons.append("general-purpose")

        idle = 1.0 - (node.current_batch_size / max(1, node.max_batch_size))
        total += 3.0 * idle
        reasons.append(f"batch-idle={idle:.2f}")

        headroom = node.vram_headroom_ratio
        total += 2.0 * headroom
        reasons.append(f"vram-headroom={headroom:.2f}")

        if node.tokens_per_second:
            total += min(3.0, node.tokens_per_second / 50.0)
            reasons.append(f"tps={node.tokens_per_second:.1f}")

        if flex_tier:
            total += 2.0 * node.power_headroom
            reasons.append(f"power-headroom={node.power_headroom:.2f}")
            if self._power_gated(node) and node.power_headroom < self.power_headroom_threshold:
                total -= 100.0
                reasons.append("power-gated-off")

        required = model_profile.weight_bytes
        if node.total_vram_bytes:
            fit = required / node.total_vram_bytes
            total -= max(0.0, fit - 1.0) * 5.0
            if fit > 1.0:
                reasons.append(f"weights-do-not-fit({fit:.2f}x)")

        return PlacementScore(node_id=node.node_id, total=total, reasons=tuple(reasons))

    def select(
        self,
        nodes: list[NodeCapability],
        profile: ArithmeticProfile,
        model_profile: ModelProfile,
        *,
        flex_tier: bool,
        required_role: NodeRole | None = None,
        excluded: frozenset[str] = frozenset(),
    ) -> NodeCapability | None:
        """Pick the best node, or ``None`` when the fleet cannot serve the job."""
        candidates = self.eligible(
            nodes,
            profile.model,
            required_role=required_role,
            excluded=excluded,
            require_memory_bytes=0,
            flex_tier=flex_tier,
        )
        if not candidates:
            return None
        scored = [
            (self.score(node, profile, flex_tier=flex_tier, model_profile=model_profile), node)
            for node in candidates
        ]
        scored.sort(key=lambda item: (-item[0].total, item[1].node_id))
        return scored[0][1]
