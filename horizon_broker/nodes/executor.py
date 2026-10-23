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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from horizon_broker.config import RecoveryConfig
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
        node = self.registry.get(decision.primary_node_id)
        if node is None:
            raise KeyError(f"node {decision.primary_node_id} is not registered")
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
            await self._start_phase(context, prefill_node_id, ExecutionPhase.PREFILL)
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
        await self._start_phase(context, node.node_id, ExecutionPhase.MONOLITHIC)
        return context

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
            context.task = asyncio.create_task(self._run_ollama(context, node))

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

        if event.type is NodeEventType.PROGRESS:
            context.job.state = JobState.RUNNING
            tokens = event.data.get("tokens_generated")
            if isinstance(tokens, int):
                context.tokens_generated = tokens
            text = event.data.get("text")
            if isinstance(text, str):
                context.chunks.append(text)
                self._notify_chunk(context.job, text)
            await self._maybe_checkpoint(context)
        elif event.type is NodeEventType.CHECKPOINT:
            await self._apply_checkpoint(context, event.data)
        elif event.type is NodeEventType.COMPLETE:
            text = event.data.get("text")
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
        for field_name in (
            "available_vram_bytes",
            "total_vram_bytes",
            "current_batch_size",
            "max_batch_size",
        ):
            value = data.get(field_name)
            if isinstance(value, int):
                updates[field_name] = value
        tokens = data.get("tokens_per_second")
        if isinstance(tokens, (int, float)):
            updates["tokens_per_second"] = float(tokens)
        headroom = data.get("power_headroom")
        if isinstance(headroom, (int, float)):
            updates["power_headroom"] = float(headroom)
        if updates:
            self.registry.upsert(node.model_copy(update=updates))

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
        context.tokens_generated = int(data.get("tokens_generated", context.tokens_generated))
        kv_ref = data.get("kv_cache_ref")
        if isinstance(kv_ref, str) and kv_ref:
            context.kv_cache_state_id = kv_ref
            context.job.kv_cache_state_id = kv_ref
        raw_tokens = data.get("partial_tokens")
        tokens: list[PartialToken] = []
        if isinstance(raw_tokens, list):
            for index, item in enumerate(raw_tokens):
                if isinstance(item, str):
                    tokens.append(PartialToken(index=index, text=item))
                elif isinstance(item, dict):
                    tokens.append(
                        PartialToken(
                            index=int(item.get("index", index)),
                            token_id=item.get("token_id"),
                            text=str(item.get("text", "")),
                        )
                    )
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
            await self._start_phase(context, decode_node_id, ExecutionPhase.MONOLITHIC)
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
        await self._start_phase(context, decode_node_id, ExecutionPhase.DECODE)

    # -- completion and failure ---------------------------------------------

    async def _complete(self, context: ExecutionContext, text: str) -> None:
        job = context.job
        job.state = JobState.COMPLETED
        job.result_text = text
        job.error = None
        self.registry.release_job(context.node_id, job.job_id)
        if self._contexts.get(job.job_id) is context:
            self._contexts.pop(job.job_id, None)
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
        job.state = JobState.FAILED
        job.error = message
        self.registry.release_job(context.node_id, job.job_id)
        if self._contexts.get(job.job_id) is context:
            self._contexts.pop(job.job_id, None)
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
        node = self.registry.get(context.node_id)
        if (
            node is not None
            and node.execution_mode is ExecutionMode.AGENT_CONTAINER
            and self._assign is not None
        ):
            try:
                await self._assign(node.node_id, {"type": "cancel", "job_id": job_id})
            except Exception as exc:  # noqa: BLE001 - the local state is cancelled anyway
                LOGGER.warning("could not send cancel frame to node %s: %s", node.node_id, exc)
        context.job.state = JobState.CANCELLED
        self.registry.release_job(context.node_id, job_id)
        if self._contexts.get(job_id) is context:
            self._contexts.pop(job_id, None)
        return True
