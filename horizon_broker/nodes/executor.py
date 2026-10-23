"""Job supervisor.

The supervisor owns the lifecycle of a dispatched job:

* it starts the execution (Ollama HTTP stream, or an assignment frame over
  the node's control channel),
* it tracks progress and emits recovery checkpoints while the generation is
  still running,
* it finalises the job, and
* it reacts to node loss by asking the recovery coordinator to rebuild the job
  on a different node.

Checkpoint cadence is controlled by ``checkpoint_every_tokens`` and
``checkpoint_every_seconds``: whichever fires first triggers a
:class:`FaultRecoverySnapshot` commit to EdgeRecord.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from horizon_broker.config import RecoveryConfig
from horizon_broker.contracts.errors import NoCapacityError
from horizon_broker.contracts.jobs import ExecutionPhase, InferenceJob, JobState
from horizon_broker.contracts.nodes import ExecutionMode, NodeCapability, NodeHealth
from horizon_broker.contracts.recovery import FaultRecoverySnapshot, PartialToken, ResumePoint
from horizon_broker.nodes.connection import NodeEvent, NodeEventType
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.observability.metrics import MetricsRegistry
from horizon_broker.ollama.client import OllamaClient
from horizon_broker.recovery.recovery import RecoveryOutcome
from horizon_broker.recovery.snapshot_store import SnapshotStore
from horizon_broker.routing.pd_dag import RoutingDecision

LOGGER = logging.getLogger(__name__)

MAX_PARTIAL_TOKENS = 512
"""How many trailing tokens a snapshot carries."""

RecoverCallback = Callable[[InferenceJob, str, str], Awaitable[RecoveryOutcome]]
AssignCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
OllamaFactory = Callable[[NodeCapability], OllamaClient]
FinishedCallback = Callable[[InferenceJob, tuple[int, int] | None], None]
"""Called when a job reaches a terminal state, with the measured token usage."""

ChunkCallback = Callable[[InferenceJob, str], None]
"""Called with every text piece a running job produces."""

MAX_TRACKED_USAGE = 1_000
"""How many finished token counts are kept for the usage report."""


@dataclass
class ExecutionContext:
    """Live state of one dispatched job."""

    job: InferenceJob
    decision: RoutingDecision
    node_id: str
    started_at: float
    attempt: int
    phase: ExecutionPhase = ExecutionPhase.MONOLITHIC
    tokens_generated: int = 0
    prompt_tokens: int | None = None
    """Prompt tokens the engine actually counted, when it reports them."""
    completion_tokens: int | None = None
    """Completion tokens the engine actually counted, when it reports them."""
    last_checkpoint_at: float = 0.0
    last_checkpoint_tokens: int = 0
    kv_cache_state_id: str | None = None
    chunks: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    batch_member: bool = False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


class JobSupervisor:
    """Runs dispatched jobs and keeps their recovery state fresh."""

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        store: SnapshotStore,
        config: RecoveryConfig | None = None,
        ollama_factory: OllamaFactory | None = None,
        assign: AssignCallback | None = None,
        recover: RecoverCallback | None = None,
        metrics: MetricsRegistry | None = None,
        on_finished: FinishedCallback | None = None,
        on_chunk: ChunkCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.store = store
        self.config = config or RecoveryConfig()
        self._ollama_factory = ollama_factory
        self._assign = assign
        self._recover = recover
        self.metrics = metrics or MetricsRegistry()
        self._on_finished = on_finished
        self._on_chunk = on_chunk
        self._clock = clock
        self._contexts: dict[str, ExecutionContext] = {}
        self._usage: dict[str, tuple[int, int]] = {}

    # -- introspection ------------------------------------------------------

    def context(self, job_id: str) -> ExecutionContext | None:
        return self._contexts.get(job_id)

    def active_jobs(self) -> list[str]:
        return list(self._contexts.keys())

    def jobs_on(self, node_id: str) -> list[str]:
        return [ctx.job.job_id for ctx in self._contexts.values() if ctx.node_id == node_id]

    # -- execution ----------------------------------------------------------

    async def start(self, job: InferenceJob, decision: RoutingDecision) -> ExecutionContext:
        """Start ``job`` on the node chosen by ``decision``."""
        if job.job_id in self._contexts:
            raise ValueError(f"job {job.job_id} already has an active execution")
        node = self.registry.get(decision.primary_node_id)
        if node is None:
            raise KeyError(f"node {decision.primary_node_id} is not registered")
        if not node.accepts_work(job.payload.model):
            raise NoCapacityError(
                f"node {node.node_id} no longer accepts job {job.job_id}"
            )
        context = ExecutionContext(
            job=job,
            decision=decision,
            node_id=node.node_id,
            started_at=self._clock(),
            attempt=job.attempt,
            kv_cache_state_id=job.kv_cache_state_id,
        )
        self._contexts[job.job_id] = context
        job.state = JobState.DISPATCHED
        job.assigned_node_id = node.node_id
        self.registry.assign_job(node.node_id, job.job_id)

        if decision.is_split and self._split_is_executable(decision):
            context.phase = ExecutionPhase.PREFILL
            job.phase = ExecutionPhase.PREFILL
            prefill_node_id = decision.prefill_node_id or decision.primary_node_id
            try:
                await self._start_phase(context, prefill_node_id, ExecutionPhase.PREFILL)
            except BaseException:
                self._rollback_admission(context)
                raise
            return context

        if decision.is_split:
            LOGGER.warning(
                "job %s wants a prefill/decode split but the engine cannot hand a KV cache; "
                "running it whole on %s",
                job.job_id,
                node.node_id,
            )
            self.metrics.incr("horizon_splits_degraded_total")
        context.phase = ExecutionPhase.MONOLITHIC
        job.phase = ExecutionPhase.MONOLITHIC
        try:
            await self._start_phase(context, node.node_id, ExecutionPhase.MONOLITHIC)
        except BaseException:
            self._rollback_admission(context)
            raise
        return context

    def _rollback_admission(self, context: ExecutionContext) -> bool:
        """Release local ownership when the node rejects a just-started job."""
        job = context.job
        active = self._contexts.get(job.job_id) is context
        if active:
            self._contexts.pop(job.job_id, None)
        reserved_nodes = {context.node_id, context.decision.primary_node_id}
        if context.decision.prefill_node_id:
            reserved_nodes.add(context.decision.prefill_node_id)
        for node_id in reserved_nodes:
            self.registry.release_job(node_id, job.job_id)
        if not active:
            # A synchronous node callback may have completed or failed the
            # job before its send operation reported an error. Terminal state
            # is authoritative and must not be rewritten as a retry.
            self._refresh_batch_gauge()
            return False
        job.state = JobState.QUEUED
        # A queued job has no active execution phase. Leaving PREFILL or
        # DECODE visible after admission fails makes status and recovery
        # consumers mistake stale work for a live phase.
        job.phase = ExecutionPhase.MONOLITHIC
        job.assigned_node_id = None
        self._refresh_batch_gauge()
        return True

    async def start_batch(
        self,
        jobs: tuple[InferenceJob, ...],
        decisions: tuple[RoutingDecision, ...],
    ) -> tuple[ExecutionContext, ...]:
        """Advertise a group of compatible jobs in one agent assignment.

        The node protocol keeps progress and completion keyed by ``job_id``,
        so recovery remains per job even though admission is one frame. Direct
        Ollama nodes do not use this path: their API has no batch endpoint.
        """
        if not jobs or len(jobs) != len(decisions):
            raise ValueError("batch jobs and decisions must be non-empty and parallel")
        node_id = decisions[0].primary_node_id
        node = self.registry.get(node_id)
        if node is None or not node.supports_batch:
            raise ValueError("batch assignment requires an agent-container node")
        if any(decision.primary_node_id != node_id or decision.is_split for decision in decisions):
            raise ValueError("batch assignments require one node and monolithic decisions")
        models = {job.payload.model for job in jobs}
        if len(models) != 1 or any(
            decision.profile.model != job.payload.model
            for job, decision in zip(jobs, decisions, strict=True)
        ):
            raise ValueError("batch assignments require one model and matching decisions")
        if not all(node.accepts_work(job.payload.model) for job in jobs):
            self.metrics.incr("horizon_batches_rejected_total")
            self.metrics.incr(
                "horizon_batch_members_total", value=len(jobs), outcome="rejected"
            )
            raise NoCapacityError(f"node {node_id} no longer accepts this batch")
        if node.batch_headroom < len(jobs):
            self.metrics.incr("horizon_batches_rejected_total")
            self.metrics.incr(
                "horizon_batch_members_total", value=len(jobs), outcome="rejected"
            )
            raise NoCapacityError(
                f"node {node_id} has capacity for {node.batch_headroom} of {len(jobs)} batch jobs"
            )
        if self._assign is None:
            raise RuntimeError("no assignment callback configured for agent containers")
        job_ids = [job.job_id for job in jobs]
        if len(set(job_ids)) != len(job_ids) or any(job_id in self._contexts for job_id in job_ids):
            raise ValueError("batch assignments require unique, inactive job ids")

        contexts: list[ExecutionContext] = []
        for job, decision in zip(jobs, decisions, strict=True):
            context = ExecutionContext(
                job=job,
                decision=decision,
                node_id=node_id,
                started_at=self._clock(),
                attempt=job.attempt,
                phase=ExecutionPhase.MONOLITHIC,
                kv_cache_state_id=job.kv_cache_state_id,
                batch_member=True,
            )
            self._contexts[job.job_id] = context
            job.state = JobState.DISPATCHED
            job.phase = ExecutionPhase.MONOLITHIC
            job.assigned_node_id = node_id
            self.registry.assign_job(node_id, job.job_id)
            contexts.append(context)
        try:
            await self._assign(
                node_id,
                {
                    "type": "batch_assign",
                    "jobs": [
                        {
                            "job": context.job.model_dump(mode="json"),
                            "phase": context.phase.value,
                            "resume_point": (
                                context.job.resume_point.model_dump(mode="json")
                                if context.job.resume_point
                                else None
                            ),
                            "kv_cache_state_id": context.kv_cache_state_id,
                        }
                        for context in contexts
                    ],
                },
            )
        except BaseException:
            # Admission is atomic: a node that rejects the batch has not
            # accepted any member. A synchronous test double or node adapter
            # may nevertheless emit a terminal event before reporting its
            # send failure; never roll that already-terminal context back.
            rolled_back = 0
            for context in contexts:
                if self._contexts.get(context.job.job_id) is not context:
                    continue
                self._contexts.pop(context.job.job_id, None)
                self.registry.release_job(node_id, context.job.job_id)
                context.job.state = JobState.QUEUED
                context.job.assigned_node_id = None
                rolled_back += 1
            self._refresh_batch_gauge()
            self.metrics.incr("horizon_batches_rejected_total")
            if rolled_back:
                self.metrics.incr(
                    "horizon_batch_members_total", value=rolled_back, outcome="rejected"
                )
            accepted_before_failure = len(contexts) - rolled_back
            if accepted_before_failure:
                self.metrics.incr(
                    "horizon_batch_members_total",
                    value=accepted_before_failure,
                    outcome="admitted",
                )
            raise
        self.metrics.incr("horizon_batches_admitted_total")
        self.metrics.incr("horizon_batch_members_total", value=len(contexts), outcome="admitted")
        self.metrics.set_gauge(
            "horizon_batch_active_members",
            self._active_batch_members(),
        )
        return tuple(contexts)

    def _active_batch_members(self) -> int:
        """Count admitted batch members still owned by the supervisor."""
        return sum(1 for context in self._contexts.values() if context.batch_member)

    def _refresh_batch_gauge(self) -> None:
        """Publish the current active-member count after lifecycle changes."""
        self.metrics.set_gauge("horizon_batch_active_members", self._active_batch_members())

    def _split_is_executable(self, decision: RoutingDecision) -> bool:
        """True when the split can actually run.

        The prefill phase must end in a KV-cache reference that the decode
        node can consume, which only an agent container can provide: a direct
        Ollama endpoint has no KV export.
        """
        if not decision.profile.supports_kv_handoff:
            return False
        if not decision.prefill_node_id or not decision.decode_node_id:
            return False
        for node_id in (decision.prefill_node_id, decision.decode_node_id):
            node = self.registry.get(node_id)
            if node is None or node.execution_mode is not ExecutionMode.AGENT_CONTAINER:
                return False
        return True

    async def _start_phase(
        self,
        context: ExecutionContext,
        node_id: str,
        phase: ExecutionPhase,
    ) -> None:
        """Run one phase of a job on one node."""
        node = self.registry.get(node_id)
        if node is None:
            raise KeyError(f"node {node_id} is not registered")
        context.node_id = node_id
        context.phase = phase
        context.job.phase = phase
        context.job.assigned_node_id = node_id
        self.registry.assign_job(node_id, context.job.job_id)

        if node.execution_mode is ExecutionMode.AGENT_CONTAINER:
            if self._assign is None:
                raise RuntimeError("no assignment callback configured for agent containers")
            await self._assign(
                node_id,
                {
                    "type": "assign",
                    "job": context.job.model_dump(mode="json"),
                    "phase": phase.value,
                    "resume_point": (
                        context.job.resume_point.model_dump(mode="json")
                        if context.job.resume_point
                        else None
                    ),
                    "kv_cache_state_id": context.kv_cache_state_id,
                },
            )
        else:
            generation = self._run_ollama(context, node)
            try:
                context.task = asyncio.create_task(generation)
            except BaseException:
                generation.close()
                raise

    async def _run_ollama(self, context: ExecutionContext, node: NodeCapability) -> None:
        """Stream a generation from an Ollama endpoint and checkpoint as it runs."""
        job = context.job
        client = self._ollama_factory(node) if self._ollama_factory else None
        if client is None:
            await self._fail(context, "no ollama client configured")
            return
        try:
            async for chunk in client.generate(job.payload):
                context.chunks.append(chunk.text)
                self._notify_chunk(context.job, chunk.text)
                context.tokens_generated += 1
                if chunk.prompt_eval_count is not None:
                    context.prompt_tokens = chunk.prompt_eval_count
                if chunk.eval_count is not None:
                    context.completion_tokens = chunk.eval_count
                if job.state is JobState.DISPATCHED:
                    job.state = JobState.RUNNING
                await self._maybe_checkpoint(context)
                if chunk.done:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("generation failed on node %s: %s", node.node_id, exc)
            await self._fail(context, str(exc))
            return
        await self._complete(context, context.text)

    # -- events from agent containers ---------------------------------------

    async def handle_event(self, event: NodeEvent) -> None:
        """Handle one frame received from a node."""
        if event.type is NodeEventType.HEARTBEAT:
            self.registry.record_heartbeat(event.node_id, event.received_at)
            self._apply_capabilities(event.node_id, event.data)
            return
        if event.type is NodeEventType.HELLO:
            self.registry.record_heartbeat(event.node_id, event.received_at)
            self._apply_capabilities(event.node_id, event.data)
            return
        if event.type is NodeEventType.DRAINING:
            LOGGER.warning("node %s announced draining", event.node_id)
            self.registry.mark_health(event.node_id, self._health_from(event.data))
            return

        job_id = event.data.get("job_id")
        context: ExecutionContext | None = None
        if isinstance(job_id, str) and job_id:
            context = self._contexts.get(job_id)
        else:
            # A node that forgets the job id must not hang the job: when it
            # owns exactly one job, the frame is unambiguous.
            candidates = [
                ctx for ctx in self._contexts.values() if ctx.node_id == event.node_id
            ]
            if len(candidates) == 1:
                context = candidates[0]
            elif candidates:
                LOGGER.warning(
                    "node %s sent a %s frame without a job id while running %d jobs; ignoring",
                    event.node_id,
                    event.type.value,
                    len(candidates),
                )
                return
        if context is None:
            return
        if event.node_id != context.node_id:
            # Frames may arrive after recovery or a prefill/decode handoff.
            # Only the current owner may update the replacement execution.
            LOGGER.warning(
                "ignoring %s frame for job %s from node %s: current owner is %s",
                event.type.value,
                context.job.job_id,
                event.node_id,
                context.node_id,
            )
            return
        if "attempt" in event.data:
            event_attempt = event.data["attempt"]
            if (
                isinstance(event_attempt, bool)
                or not isinstance(event_attempt, int)
                or event_attempt < 0
            ):
                LOGGER.warning("ignoring malformed attempt on %s frame for job %s", event.type.value, context.job.job_id)
                return
            if event_attempt != context.attempt:
                LOGGER.warning(
                    "ignoring %s frame for job %s from attempt %d: current attempt is %d",
                    event.type.value,
                    context.job.job_id,
                    event_attempt,
                    context.attempt,
                )
                return

        if event.type is NodeEventType.PROGRESS:
            tokens = event.data.get("tokens_generated")
            if tokens is not None and (
                not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
            ):
                LOGGER.warning("ignoring malformed progress for job %s", context.job.job_id)
                return
            context.job.state = JobState.RUNNING
            if isinstance(tokens, int):
                context.tokens_generated = max(context.tokens_generated, tokens)
            text = event.data.get("text")
            if isinstance(text, str):
                context.chunks.append(text)
                self._notify_chunk(context.job, text)
            await self._maybe_checkpoint(context)
        elif event.type is NodeEventType.CHECKPOINT:
            await self._apply_checkpoint(context, event.data)
        elif event.type is NodeEventType.COMPLETE:
            text = event.data.get("text")
            if context.phase is not ExecutionPhase.PREFILL and not isinstance(text, str):
                LOGGER.warning("ignoring malformed completion for job %s", context.job.job_id)
                return
            if isinstance(text, str) and text:
                context.chunks.append(text)
                self._notify_chunk(context.job, text)
            if context.phase is ExecutionPhase.PREFILL:
                await self._handoff_to_decode(context, event.data)
                return
            await self._complete(context, context.text)
        elif event.type is NodeEventType.ERROR:
            if event.data.get("kind") == "unreachable":
                # The socket dropped; the connection retries with a backoff and
                # the watchdog decides whether the node is lost. Failing the
                # job here would throw away work the node may still be doing.
                LOGGER.warning(
                    "node %s is unreachable; the watchdog decides its fate", event.node_id
                )
                return
            await self._fail(context, str(event.data.get("error", "node reported an error")))

    @staticmethod
    def _health_from(data: dict[str, Any]) -> NodeHealth:
        """Read a health verdict from a frame.

        A verdict is an enum: storing the raw text would make every
        identity check (``health is NodeHealth.DRAINING``) fail.
        """
        raw = data.get("health", "draining")
        if isinstance(raw, NodeHealth):
            return raw
        if isinstance(raw, str):
            for health in NodeHealth:
                if health.value == raw:
                    return health
        return NodeHealth.DRAINING

    def _apply_capabilities(self, node_id: str, data: dict[str, Any]) -> None:
        """Update the registry from a heartbeat payload."""
        node = self.registry.get(node_id)
        if node is None:
            return
        updates: dict[str, Any] = {}
        for field_name in ("available_vram_bytes", "total_vram_bytes"):
            value = data.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                updates[field_name] = value
        for field_name, minimum in (("current_batch_size", 0), ("max_batch_size", 1)):
            value = data.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
                updates[field_name] = value
        tokens = data.get("tokens_per_second")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            try:
                token_rate = float(tokens)
            except (OverflowError, ValueError):
                token_rate = None
            if token_rate is not None and math.isfinite(token_rate) and token_rate >= 0:
                updates["tokens_per_second"] = token_rate
        headroom = data.get("power_headroom")
        if isinstance(headroom, (int, float)) and not isinstance(headroom, bool):
            try:
                power_headroom = float(headroom)
            except (OverflowError, ValueError):
                power_headroom = None
            if power_headroom is not None and math.isfinite(power_headroom):
                updates["power_headroom"] = max(0.0, min(1.0, power_headroom))
        total = updates.get("total_vram_bytes", node.total_vram_bytes)
        available = updates.get("available_vram_bytes", node.available_vram_bytes)
        updates["available_vram_bytes"] = min(available, total)
        if updates:
            self.registry.upsert(
                node.model_copy(update=updates),
                preserve_reported_batch_size="current_batch_size" not in updates,
            )

    # -- checkpoints --------------------------------------------------------

    async def _maybe_checkpoint(self, context: ExecutionContext) -> None:
        """Commit a recovery snapshot when the cadence requires it."""
        now = self._clock()
        tokens_since = context.tokens_generated - context.last_checkpoint_tokens
        seconds_since = now - context.last_checkpoint_at
        if context.last_checkpoint_at and (
            tokens_since < self.config.checkpoint_every_tokens
            and seconds_since < self.config.checkpoint_every_seconds
        ):
            return
        await self._commit_snapshot(context)

    async def _commit_snapshot(self, context: ExecutionContext) -> None:
        """Write a :class:`FaultRecoverySnapshot` for ``context``."""
        job = context.job
        tokens = [
            PartialToken(index=index, text=chunk)
            for index, chunk in enumerate(context.chunks[-MAX_PARTIAL_TOKENS:])
        ]
        resume = ResumePoint(
            decode_step=context.tokens_generated,
            generated_token_count=context.tokens_generated,
            kv_cache_ref=context.kv_cache_state_id,
            kv_cache_state_id=context.kv_cache_state_id,
            prompt_hash=job.payload.prompt_hash,
            model=job.payload.model,
        )
        snapshot = FaultRecoverySnapshot(
            job_id=job.job_id,
            node_id=context.node_id,
            attempt=context.attempt,
            model=job.payload.model,
            step=context.tokens_generated,
            partial_tokens=tokens,
            resume_point=resume,
        )
        context.last_checkpoint_at = self._clock()
        context.last_checkpoint_tokens = context.tokens_generated
        try:
            await self.store.commit(snapshot)
        except Exception as exc:  # noqa: BLE001 - a failed checkpoint must not kill the job
            LOGGER.warning("checkpoint commit failed for job %s: %s", job.job_id, exc)
            return
        if context.kv_cache_state_id:
            job.kv_cache_state_id = context.kv_cache_state_id

    async def _apply_checkpoint(self, context: ExecutionContext, data: dict[str, Any]) -> None:
        """Apply a checkpoint frame sent by an agent container."""
        raw_count = data.get("tokens_generated", context.tokens_generated)
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            LOGGER.warning("ignoring malformed checkpoint for job %s", context.job.job_id)
            return
        kv_refs = [
            value
            for key in ("kv_cache_ref", "kv_cache_state_id")
            if isinstance(value := data.get(key), str)
            and value
            and value == value.strip()
        ]
        if len(set(kv_refs)) > 1:
            LOGGER.warning("ignoring conflicting checkpoint cache references for job %s", context.job.job_id)
            return
        kv_ref = kv_refs[0] if kv_refs else None
        raw_tokens = data.get("partial_tokens")
        tokens: list[PartialToken] = []
        if raw_tokens is not None and not isinstance(raw_tokens, list):
            LOGGER.warning("ignoring malformed checkpoint for job %s", context.job.job_id)
            return
        if raw_tokens is not None:
            try:
                for index, item in enumerate(raw_tokens):
                    if isinstance(item, str):
                        tokens.append(PartialToken(index=index, text=item))
                    elif isinstance(item, dict):
                        raw_index = item.get("index", index)
                        raw_token_id = item.get("token_id")
                        raw_text = item.get("text", "")
                        raw_logprob = item.get("logprob")
                        if raw_logprob is not None and isinstance(raw_logprob, (int, float)):
                            try:
                                finite_logprob = math.isfinite(float(raw_logprob))
                            except (OverflowError, ValueError):
                                finite_logprob = False
                        else:
                            finite_logprob = raw_logprob is None
                        if (
                            not isinstance(raw_index, int)
                            or isinstance(raw_index, bool)
                            or raw_index < 0
                            or (
                                raw_token_id is not None
                                and (
                                    not isinstance(raw_token_id, int)
                                    or isinstance(raw_token_id, bool)
                                )
                            )
                            or not isinstance(raw_text, str)
                            or (
                                raw_logprob is not None
                                and (
                                    not isinstance(raw_logprob, (int, float))
                                    or isinstance(raw_logprob, bool)
                                    or not finite_logprob
                                )
                            )
                        ):
                            raise ValueError("checkpoint partial token has invalid fields")
                        tokens.append(
                            PartialToken(
                                index=raw_index,
                                token_id=raw_token_id,
                                text=raw_text,
                                logprob=raw_logprob,
                            )
                        )
                    else:
                        raise ValueError("checkpoint partial token is not an object or string")
            except (TypeError, ValueError):
                LOGGER.warning("ignoring malformed checkpoint for job %s", context.job.job_id)
                return
            if any(token.index != index for index, token in enumerate(tokens)):
                LOGGER.warning("ignoring non-sequential checkpoint for job %s", context.job.job_id)
                return
        context.tokens_generated = max(context.tokens_generated, raw_count)
        if isinstance(kv_ref, str) and kv_ref:
            context.kv_cache_state_id = kv_ref
            context.job.kv_cache_state_id = kv_ref
        if raw_tokens is not None:
            context.chunks = [token.text for token in tokens]
        await self._commit_snapshot(context)

    async def _handoff_to_decode(self, context: ExecutionContext, data: dict[str, Any]) -> None:
        """Move a finished prefill phase to the decode node.

        The prefill node reports the KV-cache reference it produced; the
        decode node consumes it. When no reference comes back the split
        cannot be honoured and the job is re-run whole on the decode node.
        """
        decision = context.decision
        decode_node_id = decision.decode_node_id
        kv_ref = data.get("kv_cache_ref") or data.get("kv_cache_state_id")
        if not isinstance(kv_ref, str) or not kv_ref:
            LOGGER.error(
                "prefill phase of job %s finished without a KV cache reference; "
                "falling back to a whole-job run",
                context.job.job_id,
            )
            self.metrics.incr("horizon_splits_degraded_total")
            if decode_node_id is None:
                await self._fail(context, "prefill phase produced no KV cache reference")
                return
            self.registry.release_job(context.node_id, context.job.job_id)
            context.chunks = []
            try:
                await self._start_phase(context, decode_node_id, ExecutionPhase.MONOLITHIC)
            except asyncio.CancelledError:
                self._rollback_admission(context)
                raise
            except BaseException as exc:
                active = self._rollback_admission(context)
                if active:
                    await self._fail(context, f"degraded decode assignment failed: {exc}")
            return

        context.kv_cache_state_id = kv_ref
        context.job.kv_cache_state_id = kv_ref
        self.registry.release_job(context.node_id, context.job.job_id)
        if decode_node_id is None:
            await self._fail(context, "split decision has no decode node")
            return
        LOGGER.info(
            "job %s: prefill on %s produced %s, handing over to %s",
            context.job.job_id,
            context.node_id,
            kv_ref,
            decode_node_id,
        )
        self.metrics.incr("horizon_splits_handed_off_total")
        try:
            await self._start_phase(context, decode_node_id, ExecutionPhase.DECODE)
        except asyncio.CancelledError:
            self._rollback_admission(context)
            raise
        except BaseException as exc:
            active = self._rollback_admission(context)
            if active:
                await self._fail(context, f"decode assignment failed: {exc}")

    # -- completion and failure ---------------------------------------------

    async def _complete(self, context: ExecutionContext, text: str) -> None:
        job = context.job
        if self._contexts.get(job.job_id) is not context or job.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        ):
            # A task can cross its final await just as cancellation removes
            # its context. Only the current execution may publish a result.
            return
        job.state = JobState.COMPLETED
        job.result_text = text
        job.error = None
        self.registry.release_job(context.node_id, job.job_id)
        if self._contexts.get(job.job_id) is context:
            self._contexts.pop(job.job_id, None)
        self._refresh_batch_gauge()
        usage = self._record_usage(job.job_id, context)
        self._notify_finished(job, usage)

    def _record_usage(self, job_id: str, context: ExecutionContext) -> tuple[int, int] | None:
        """Keep the token counts the engine reported, for the usage report."""
        if context.prompt_tokens is None and context.completion_tokens is None:
            return None
        usage = (context.prompt_tokens or 0, context.completion_tokens or 0)
        if len(self._usage) >= MAX_TRACKED_USAGE:
            oldest = next(iter(self._usage))
            self._usage.pop(oldest, None)
        self._usage[job_id] = usage
        return usage

    def usage(self, job_id: str) -> tuple[int, int] | None:
        """The measured (prompt, completion) token counts of a finished job."""
        return self._usage.get(job_id)

    async def _fail(self, context: ExecutionContext, message: str) -> None:
        job = context.job
        if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED) or (
            self._contexts.get(job.job_id) is not context and job.state is not JobState.QUEUED
        ):
            # A late transport failure must not resurrect a cancelled or
            # already-finished execution. A queued job is allowed here because
            # split handoff rollback deliberately clears its context before
            # reporting the handoff failure.
            return
        job.state = JobState.FAILED
        job.error = message
        self.registry.release_job(context.node_id, job.job_id)
        if self._contexts.get(job.job_id) is context:
            self._contexts.pop(job.job_id, None)
        self._refresh_batch_gauge()
        self._notify_finished(job)

    def _notify_finished(self, job: InferenceJob, usage: tuple[int, int] | None = None) -> None:
        """Tell the engine that the job reached a terminal state."""
        if self._on_finished is None:
            return
        try:
            self._on_finished(job, usage)
        except Exception as exc:  # noqa: BLE001 - a broken waiter must not fail the job
            LOGGER.warning("completion callback failed for job %s: %s", job.job_id, exc)

    def _notify_chunk(self, job: InferenceJob, text: str) -> None:
        """Tell the engine about one piece of the answer, as it arrives."""
        if self._on_chunk is None or not text:
            return
        try:
            self._on_chunk(job, text)
        except Exception as exc:  # noqa: BLE001 - a broken listener must not fail the job
            LOGGER.warning("chunk callback failed for job %s: %s", job.job_id, exc)

    # -- fault handling -----------------------------------------------------

    async def on_node_lost(self, node_id: str, reason: str) -> None:
        """Recover every job that was running on a node that just died."""
        # The watchdog normally records this verdict first, but recovery is
        # also invoked directly by connection-loss and operator paths. Keep
        # the registry truthful in every entry path so queued retries cannot
        # immediately select the node that just failed.
        self.registry.mark_health(node_id, NodeHealth.LOST, reason)
        self.registry.quarantine(node_id, self.config.quarantine_seconds)
        contexts = [ctx for ctx in self._contexts.values() if ctx.node_id == node_id]
        for context in contexts:
            job = context.job
            LOGGER.error(
                "job %s lost with node %s (%s)", job.job_id, node_id, reason
            )
            job.state = JobState.RECOVERING
            if context.task is not None:
                context.task.cancel()
            self.registry.release_job(node_id, job.job_id)
            if self._contexts.get(job.job_id) is context:
                self._contexts.pop(job.job_id, None)
            self._refresh_batch_gauge()
            if self._recover is None:
                job.state = JobState.FAILED
                job.error = f"node {node_id} lost and no recovery coordinator is configured"
                self._notify_finished(job)
                continue
            try:
                outcome = await self._recover(job, node_id, reason)
                if not outcome.dispatched and job.state is JobState.FAILED:
                    # Exhausting the retry budget returns normally; callers and
                    # the journal still need the same terminal notification.
                    self._notify_finished(job)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("recovery failed for job %s: %s", job.job_id, exc)
                if job.state in (JobState.COMPLETED, JobState.CANCELLED):
                    # A recovery callback may publish a terminal result before
                    # a contradictory acknowledgement/error reaches here.
                    # Preserve that observable result instead of rewriting it.
                    continue
                job.state = JobState.FAILED
                job.error = str(exc)
                self._notify_finished(job)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        context = self._contexts.get(job_id)
        if context is None:
            return False
        if context.task is not None:
            context.task.cancel()
        send_cancelled = False
        node = self.registry.get(context.node_id)
        if (
            node is not None
            and node.execution_mode is ExecutionMode.AGENT_CONTAINER
            and self._assign is not None
        ):
            try:
                await self._assign(node.node_id, {"type": "cancel", "job_id": job_id})
            except asyncio.CancelledError:
                send_cancelled = True
            except Exception as exc:  # noqa: BLE001 - the local state is cancelled anyway
                LOGGER.warning("could not send cancel frame to node %s: %s", node.node_id, exc)
        if context.job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            # Completion can race the cancel acknowledgement. Preserve the
            # node's terminal result instead of claiming cancellation won.
            return False
        context.job.state = JobState.CANCELLED
        self.registry.release_job(context.node_id, job_id)
        if self._contexts.get(job_id) is context:
            self._contexts.pop(job_id, None)
        self._refresh_batch_gauge()
        if send_cancelled:
            raise asyncio.CancelledError
        return True
