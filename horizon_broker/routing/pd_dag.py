"""PD-DAG router.

Prefill/Decode Disaggregation: the router inspects a job's arithmetic profile
and decides whether the whole job runs on one node, or whether the prefill
phase is placed on a compute-bound node and the decode phase on a
memory-bandwidth-bound node.

    massive prompt + short output  -> Node Type A (prefill-optimised)
    short prompt   + massive output-> Node Type B (decode-optimised)

When both a dedicated prefill node and a dedicated decode node exist and the
engine can hand a KV cache across nodes, the router emits a ``PD_SPLIT``
decision: phase ``PREFILL`` runs first and hands its ``kv_cache_state_id`` to
the decode phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from horizon_broker.config import RoutingConfig
from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import InferenceJob, PriorityTier
from horizon_broker.contracts.nodes import NodeCapability, NodeRole
from horizon_broker.routing.arithmetic import (
    ArithmeticProfile,
    ComputePhase,
    ModelProfile,
    RoutingThresholds,
    analyse,
    profile_for,
)
from horizon_broker.routing.placement import NodeSelector


class RoutingMode(StrEnum):
    """How the job is distributed across the mesh."""

    DIRECT = "direct"
    """One node runs the whole job."""

    PD_SPLIT = "pd_split"
    """Prefill on node A, decode on node B, KV state handed over."""


@dataclass(frozen=True)
class RoutingDecision:
    """The router's verdict for one job."""

    mode: RoutingMode
    primary_node_id: str
    profile: ArithmeticProfile = field(repr=False)
    prefill_node_id: str | None = None
    decode_node_id: str | None = None
    score: float = 0.0
    rationale: str = ""
    excluded: frozenset[str] = frozenset()

    @property
    def is_split(self) -> bool:
        return self.mode is RoutingMode.PD_SPLIT


class PDDagRouter:
    """Routes jobs to nodes based on arithmetic intensity."""

    def __init__(
        self,
        selector: NodeSelector | None = None,
        config: RoutingConfig | None = None,
    ) -> None:
        self.selector = selector or NodeSelector()
        self.config = config or RoutingConfig()
        self.thresholds = RoutingThresholds(
            prefill_prompt_threshold=self.config.prefill_prompt_threshold,
            decode_output_threshold=self.config.decode_output_threshold,
            prefill_ratio=self.config.prefill_intensity_ratio,
            decode_ratio=self.config.decode_intensity_ratio,
        )

    def _model_profile(self, model: str) -> ModelProfile:
        """Return the model profile, honouring the configured KV handoff list."""
        profile = profile_for(model)
        if model in self.config.kv_handoff_models and not profile.supports_kv_handoff:
            profile = replace(profile, supports_kv_handoff=True)
        return profile

    def route(
        self,
        job: InferenceJob,
        nodes: list[NodeCapability],
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> RoutingDecision:
        """Place ``job`` on the mesh.

        Raises :class:`NoCapacityError` when no node can accept the job; the
        scheduler responds by leaving the job in the queue.
        """
        profile = analyse(job.payload, thresholds=self.thresholds)
        model_profile = self._model_profile(job.payload.model)
        if model_profile.supports_kv_handoff:
            profile = replace(profile, supports_kv_handoff=True)
        flex = job.priority_tier is PriorityTier.FLEX

        if profile.phase is ComputePhase.PREFILL_BOUND:
            decision = self._route_prefill(job, nodes, profile, model_profile, flex, excluded)
        elif profile.phase is ComputePhase.DECODE_BOUND:
            decision = self._route_decode(job, nodes, profile, model_profile, flex, excluded)
        else:
            decision = self._route_balanced(job, nodes, profile, model_profile, flex, excluded)

        if decision is None:
            raise NoCapacityError(
                f"no node in the mesh can accept job {job.job_id} "
                f"(model={job.payload.model}, phase={profile.phase.value})"
            )
        return decision

    # -- phase specific strategies -----------------------------------------

    def _route_prefill(
        self,
        job: InferenceJob,
        nodes: list[NodeCapability],
        profile: ArithmeticProfile,
        model_profile: ModelProfile,
        flex: bool,
        excluded: frozenset[str],
    ) -> RoutingDecision | None:
        """Heavy prefill: prefer a compute-bound node, split when it pays off."""
        prefill_node = self.selector.select(
            nodes,
            profile,
            model_profile,
            flex_tier=flex,
            required_role=NodeRole.PREFILL,
            excluded=excluded,
        )
        if prefill_node is None:
            prefill_node = self.selector.select(
                nodes, profile, model_profile, flex_tier=flex, excluded=excluded
            )
        if prefill_node is None:
            return None

        decode_node = None
        if self._split_is_worthwhile(profile, model_profile, nodes, excluded):
            decode_node = self.selector.select(
                nodes,
                profile,
                model_profile,
                flex_tier=flex,
                required_role=NodeRole.DECODE,
                excluded=excluded | {prefill_node.node_id},
            )

        if decode_node is not None:
            return RoutingDecision(
                mode=RoutingMode.PD_SPLIT,
                primary_node_id=prefill_node.node_id,
                prefill_node_id=prefill_node.node_id,
                decode_node_id=decode_node.node_id,
                profile=profile,
                score=24.0,
                rationale=(
                    f"prompt={profile.prompt_tokens}t/output={profile.output_tokens}t "
                    f"-> prefill on {prefill_node.node_id}, decode on {decode_node.node_id}"
                ),
                excluded=excluded,
            )

        return RoutingDecision(
            mode=RoutingMode.DIRECT,
            primary_node_id=prefill_node.node_id,
            prefill_node_id=prefill_node.node_id,
            profile=profile,
            score=12.0,
            rationale=(
                f"prompt={profile.prompt_tokens}t/output={profile.output_tokens}t "
                f"-> compute-bound prefill on {prefill_node.node_id}"
            ),
            excluded=excluded,
        )

    def _route_decode(
        self,
        job: InferenceJob,
        nodes: list[NodeCapability],
        profile: ArithmeticProfile,
        model_profile: ModelProfile,
        flex: bool,
        excluded: frozenset[str],
    ) -> RoutingDecision | None:
        """Heavy decode: prefer a memory-bandwidth-rich node."""
        decode_node = self.selector.select(
            nodes,
            profile,
            model_profile,
            flex_tier=flex,
            required_role=NodeRole.DECODE,
            excluded=excluded,
        )
        if decode_node is None:
            decode_node = self.selector.select(
                nodes, profile, model_profile, flex_tier=flex, excluded=excluded
            )
        if decode_node is None:
            return None
        return RoutingDecision(
            mode=RoutingMode.DIRECT,
            primary_node_id=decode_node.node_id,
            decode_node_id=decode_node.node_id,
            profile=profile,
            score=12.0,
            rationale=(
                f"prompt={profile.prompt_tokens}t/output={profile.output_tokens}t "
                f"-> memory-bound decode on {decode_node.node_id}"
            ),
            excluded=excluded,
        )

    def _route_balanced(
        self,
        job: InferenceJob,
        nodes: list[NodeCapability],
        profile: ArithmeticProfile,
        model_profile: ModelProfile,
        flex: bool,
        excluded: frozenset[str],
    ) -> RoutingDecision | None:
        """Neither phase dominates: pick the least loaded general-purpose node."""
        node = self.selector.select(
            nodes, profile, model_profile, flex_tier=flex, excluded=excluded
        )
        if node is None:
            return None
        return RoutingDecision(
            mode=RoutingMode.DIRECT,
            primary_node_id=node.node_id,
            profile=profile,
            score=6.0,
            rationale=(
                f"prompt={profile.prompt_tokens}t/output={profile.output_tokens}t "
                f"-> balanced workload on {node.node_id}"
            ),
            excluded=excluded,
        )

    def _split_is_worthwhile(
        self,
        profile: ArithmeticProfile,
        model_profile: ModelProfile,
        nodes: list[NodeCapability],
        excluded: frozenset[str],
    ) -> bool:
        """A split needs a long decode phase, KV handoff support and both node types."""
        if not profile.supports_kv_handoff:
            return False
        if profile.output_tokens < 1_024:
            return False
        has_prefill = any(
            node.role is NodeRole.PREFILL and node.node_id not in excluded for node in nodes
        )
        has_decode = any(
            node.role is NodeRole.DECODE and node.node_id not in excluded for node in nodes
        )
        return has_prefill and has_decode
