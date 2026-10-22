"""The Broker engine.

Wires the four sub-systems together and owns the two scheduling lanes:

``_interactive_loop``
    Latency-bound work. Jobs are dispatched as soon as they are popped from the
    interactive lane, bounded by ``interactive_concurrency``.

``_flex_loop``
    Sleep-time compute. Jobs are held in a per-model window until the batch is
    large enough (or the window expires), then dispatched together.

The engine also owns the fault path: the watchdog reports a lost node, the
supervisor asks the recovery coordinator to rebuild the job from the last
EdgeRecord snapshot, and the rebuilt job is re-queued and dispatched on a
different node.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any

import aiohttp

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.errors import (
    BrokerDrainingError,
    NoCapacityError,
    QueueFullError,
)
from horizon_broker.contracts.jobs import InferenceJob, JobState
from horizon_broker.contracts.nodes import NodeHealth, PowerClass
from horizon_broker.contracts.recovery import utcnow
from horizon_broker.nodes.executor import JobSupervisor
from horizon_broker.nodes.manager import NodeConnectionManager
from horizon_broker.nodes.probe import NodeHealthProbe
from horizon_broker.nodes.registry import NodeRegistry
from horizon_broker.nodes.watchdog import FaultWatchdog
from horizon_broker.observability.metrics import MetricsRegistry
from horizon_broker.ollama.client import OllamaClient
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.recovery.ledger import EdgeRecordClient
from horizon_broker.recovery.recovery import RecoveryCoordinator, RecoveryOutcome
from horizon_broker.recovery.snapshot_store import SnapshotStore
from horizon_broker.routing.pd_dag import PDDagRouter
from horizon_broker.routing.placement import NodeSelector
from horizon_broker.scheduler.batcher import BatchPlanner
from horizon_broker.scheduler.dispatch import Dispatcher
from horizon_broker.scheduler.priority_queue import JobPriorityQueue
from horizon_broker.scheduler.window import FlexBatch, WindowPolicy

LOGGER = logging.getLogger(__name__)

MAX_TRACKED_JOBS = 10_000
"""How many finished jobs stay queryable through the status endpoint."""

TERMINAL_STATES = (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)


class BrokerEngine:
    """The Horizon Broker runtime."""

    def __init__(
        self,
        config: BrokerConfig,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config
        self.metrics = MetricsRegistry()
        self.metrics.describe("horizon_jobs_submitted_total", "Jobs accepted by the ingress API.")
        self.metrics.describe("horizon_jobs_completed_total", "Jobs that finished successfully.")
        self.metrics.describe("horizon_jobs_failed_total", "Jobs that ended in failure.")
        self.metrics.describe("horizon_jobs_recovered_total", "Jobs resumed after a node loss.")
        self.metrics.describe(
            "horizon_jobs_restored_total", "Jobs replayed from the journal on a cold start."
        )
        self.metrics.describe("horizon_flex_batch_size", "Size of the dispatched Flex batches.")
        self.metrics.describe("horizon_recovery_seconds", "Seconds from fault to re-dispatch.")
        self.metrics.describe(
            "horizon_splits_handed_off_total", "Prefill/decode splits that handed a KV cache over."
        )
        # Only the series that are always unlabelled are declared: a declared
        # zero next to a labelled series would export one metric name with
        # two shapes. Metrics with a bounded label set are declared per
        # label, so a dashboard sees them before the first event.
        for metric in (
            "horizon_jobs_completed_total",
            "horizon_jobs_failed_total",
            "horizon_jobs_deferred_total",
            "horizon_jobs_duplicate_total",
            "horizon_splits_handed_off_total",
            "horizon_splits_degraded_total",
            "horizon_jobs_restored_total",
        ):
            self.metrics.declare(metric)
        for tier in ("interactive", "flex"):
            self.metrics.declare("horizon_jobs_submitted_total", tier=tier)
        for resumed in ("true", "false"):
            self.metrics.declare("horizon_jobs_recovered_total", resumed=resumed)
        self.metrics.declare_summary("horizon_recovery_seconds")

        self.registry = NodeRegistry()
        for node in config.static_nodes:
            self.registry.upsert(node)

        self.router = PDDagRouter(
            selector=NodeSelector(
                power_gated_classes=self._power_gated_classes(config),
                power_headroom_threshold=config.window.power_headroom_threshold,
            ),
            config=config.routing,
        )
        self.queue = JobPriorityQueue()
        self.window_policy = WindowPolicy.from_config(config.window)
        self.planner = BatchPlanner(self.router)

        self.journal = LocalJournal(config.journal_path)
        self.ledger = EdgeRecordClient(config.ledger, session=session)
        self.store = SnapshotStore(
            ledger=self.ledger,
            journal=self.journal,
            ledger_config=config.ledger,
            journal_max_bytes=config.journal_max_bytes,
        )

        self.supervisor = JobSupervisor(
            registry=self.registry,
            store=self.store,
            config=config.recovery,
            ollama_factory=self._ollama_factory,
            assign=self._assign,
            recover=self._recover,
            metrics=self.metrics,
            on_finished=self._on_job_finished,
        )
        self.manager = NodeConnectionManager(
            self.registry,
            self.supervisor,
            config=config.heartbeat,
        )
        self.probe = NodeHealthProbe(self.registry, config=config.heartbeat)
        self.dispatcher = Dispatcher(self.registry, self.router, self.supervisor)
        self.recovery = RecoveryCoordinator(
            store=self.store,
            config=config.recovery,
            requeue=self._requeue,
            dispatch=self._dispatch_recovered,
        )
        self.watchdog = FaultWatchdog(
            self.registry,
            config=config.heartbeat,
            recovery_config=config.recovery,
            on_node_lost=self._on_node_lost,
            on_node_draining=self._on_node_draining,
        )

        self._jobs: dict[str, InferenceJob] = {}
        self._usage: dict[str, tuple[int, int]] = {}
        self._waiters: dict[str, set[asyncio.Future[str | None]]] = {}
        self._journal_tasks: set[asyncio.Task[None]] = set()
        self._tasks: list[asyncio.Task[None]] = []
        self._ollama_clients: dict[str, OllamaClient] = {}
        self._running = False
        self._draining = False
        self._inflight_interactive = 0
        self._inflight_flex = 0

    @staticmethod
    def _power_gated_classes(config: BrokerConfig) -> tuple[PowerClass, ...]:
        """The power classes that may not take sleep-time work when low.

        An unknown class name is ignored rather than fatal: a typo in the
        config should not stop the broker from starting.
        """
        known = {member.value: member for member in PowerClass}
        classes: list[PowerClass] = []
        for name in config.window.power_gated_tiers:
            member = known.get(name)
            if member is None:
                LOGGER.warning("unknown power class in power_gated_tiers: %s", name)
                continue
            classes.append(member)
        return tuple(classes)

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        """Start the scheduler loops, the connection manager and the watchdog."""
        if self._running:
            return
        self._running = True
        restored = await self._replay_journal()
        self._tasks = [
            asyncio.create_task(self._interactive_loop(), name="horizon-interactive"),
            asyncio.create_task(self._flex_loop(), name="horizon-flex"),
            asyncio.create_task(self.manager.run(), name="horizon-connections"),
            asyncio.create_task(self.watchdog.run(), name="horizon-watchdog"),
            asyncio.create_task(self.probe.run(), name="horizon-probe"),
        ]
        LOGGER.info(
            "horizon broker started: %d static nodes, window=%.0fs, %d job(s) replayed",
            len(self.config.static_nodes),
            self.config.window.window_seconds,
            restored,
        )

    async def stop(self, *, drain: bool = True) -> None:
        """Stop the broker.

        With ``drain`` the broker stops accepting work, lets the in-flight
        jobs finish (up to ``drain_timeout_seconds``) and then tears down.
        Without it the broker stops immediately.
        """
        if drain and self._running:
            await self.drain(timeout=self.config.drain_timeout_seconds)
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        # A cancelled job's terminal envelope is still being written; the
        # journal must not go dark under a pending commit.
        pending = [task for task in self._journal_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._journal_tasks.clear()
        await self.manager.close()
        await self.probe.close()
        await self._close_ollama_clients()
        await self.ledger.close()
        LOGGER.info("horizon broker stopped")

    async def drain(self, timeout: float) -> bool:
        """Stop accepting work and wait for the in-flight jobs to finish.

        Returns True when everything finished inside the timeout.
        """
        self._draining = True
        LOGGER.info("draining: refusing new work, waiting for in-flight jobs")
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if not self.supervisor.active_jobs() and not self._inflight_interactive:
                break
            await asyncio.sleep(0.05)
        pending = self.supervisor.active_jobs()
        if pending:
            LOGGER.warning("drain timed out with %d job(s) still running", len(pending))
            return False
        LOGGER.info("drain complete")
        return True

    # -- ingress ------------------------------------------------------------

    async def submit(self, job: InferenceJob) -> InferenceJob:
        """Accept a job and place it on the correct lane.

        Submission is idempotent: a repeated job id returns the job that is
        already known instead of starting the work twice. Callers that retry
        after a timeout get the original job back.
        """
        existing = self._jobs.get(job.job_id)
        if existing is not None:
            LOGGER.info("job %s already known; ignoring duplicate submission", job.job_id)
            self.metrics.incr("horizon_jobs_duplicate_total")
            return existing
        if self._draining:
            raise BrokerDrainingError("the broker is draining and is not accepting work")
        if self._queue_depth() >= self.config.scheduling.max_queue_depth:
            raise QueueFullError(
                f"queue depth {self._queue_depth()} reached the configured limit "
                f"{self.config.scheduling.max_queue_depth}"
            )
        self._jobs[job.job_id] = job
        try:
            await self.store.store_job_payload(job)
        except Exception as exc:  # noqa: BLE001 - journalling must not block ingress
            LOGGER.warning("could not journal job %s: %s", job.job_id, exc)
        self.queue.submit(job)
        self._prune_jobs()
        self.metrics.incr("horizon_jobs_submitted_total", tier=job.priority_tier.value)
        LOGGER.info(
            "accepted job %s tier=%s model=%s latency_budget=%dms",
            job.job_id,
            job.priority_tier.value,
            job.payload.model,
            job.max_acceptable_latency_ms,
        )
        return job

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Returns False for an unknown job id and for a job that already
        reached a terminal state: rewriting a finished job into CANCELLED
        would lie to every future status query.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.state in TERMINAL_STATES:
            return False
        cancelled = await self.supervisor.cancel(job_id)
        if not cancelled:
            self.queue.remove(job_id)
            job.state = JobState.CANCELLED
        # A cancellation is a terminal answer: the waiter must be released,
        # or an OpenAI client hangs until its own timeout.
        self._on_job_finished(job)
        return True

    async def wait_for_result(self, job_id: str, timeout: float | None = None) -> str | None:
        """Wait for a job to finish and return its text.

        Returns ``None`` when the job failed or the timeout elapsed. The
        OpenAI-compatible ingress uses this to answer a synchronous client.

        Every waiter owns its own future: ``asyncio.wait_for`` cancels the
        future it times out on, so a shared future would drag every peer
        waiter down with the one caller whose budget expired.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.state in TERMINAL_STATES:
            return job.result_text
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(job_id, set()).add(future)
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            return None
        finally:
            # Discard only this waiter's future (already cancelled on a
            # timeout); the rest of the set stays resolvable.
            entry = self._waiters.get(job_id)
            if entry is not None:
                entry.discard(future)
                if not entry:
                    del self._waiters[job_id]

    def _on_job_finished(self, job: InferenceJob, usage: tuple[int, int] | None = None) -> None:
        """Resolve the waiters when a job reaches a terminal state.

        Popping the whole set here (not in the waiters) is what keeps the
        registry bounded: it is the only moment a job can leave the waiting
        set, whether anyone was watching or not.
        """
        if usage is not None:
            self._usage[job.job_id] = usage
        if job.state is JobState.COMPLETED:
            self.metrics.incr("horizon_jobs_completed_total")
        elif job.state is JobState.FAILED:
            self.metrics.incr("horizon_jobs_failed_total")
        futures = self._waiters.pop(job.job_id, None)
        for future in futures or ():
            if not future.done():
                future.set_result(job.result_text)
        self._record_state_async(job)

    def _record_state_async(self, job: InferenceJob) -> None:
        """Journal a terminal envelope without blocking the completion callback.

        The callback runs synchronously inside the supervisor; a journal
        write would have to be awaited from there. A tracked task keeps the
        write alive until ``stop`` collects it.
        """
        try:
            task = asyncio.create_task(self._journal_state(job))
        except RuntimeError:  # no running loop; nothing durable to promise
            return
        self._journal_tasks.add(task)
        task.add_done_callback(self._journal_tasks.discard)

    async def _journal_state(self, job: InferenceJob) -> None:
        """Persist the job envelope so a cold restart sees its newest state."""
        try:
            await self.store.store_job_payload(job)
        except Exception as exc:  # noqa: BLE001 - journalling must not break a finish
            LOGGER.warning("could not journal job %s state: %s", job.job_id, exc)

    async def _replay_journal(self) -> int:
        """Re-queue the jobs a previous process left non-terminal.

        The journal is read in write order, so the last envelope per job id
        is its newest state. A job whose newest record is terminal did its
        work and stays retired; anything else re-enters the queue and is
        rejected for good if its attempt budget was already spent.
        """
        try:
            jobs = await self.store.load_jobs()
        except Exception as exc:  # noqa: BLE001 - a broken journal must not block startup
            LOGGER.warning("could not read the journal for replay: %s", exc)
            return 0
        restored = 0
        for job in jobs:
            previous_state = job.state.value
            if job.state in TERMINAL_STATES:
                continue
            if job.attempt >= self.config.recovery.max_attempts:
                job.state = JobState.FAILED
                job.error = "recovery attempts exhausted before the restart"
                continue
            job.state = JobState.QUEUED
            job.not_before = None
            job.assigned_node_id = None
            self._jobs[job.job_id] = job
            self.queue.submit(job)
            restored += 1
            self.metrics.incr("horizon_jobs_restored_total")
            LOGGER.info(
                "replayed job %s into the queue (journalled state was %s)",
                job.job_id,
                previous_state,
            )
        return restored

    def token_usage(self, job_id: str) -> tuple[int, int] | None:
        """The measured (prompt, completion) token counts of a finished job."""
        return self._usage.get(job_id)

    def status(self, job_id: str) -> dict[str, Any] | None:
        """Return the JSON view of a job."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        payload = job.model_dump(mode="json")
        payload["assigned_node_id"] = job.assigned_node_id
        return payload

    def list_jobs(self, limit: int = 100) -> dict[str, Any]:
        """Return the tracked jobs, newest first.

        Operators look for the slow ones and the stuck ones, so the newest
        submission first is the useful order. ``limit`` bounds one response;
        the store itself is already bounded by ``MAX_TRACKED_JOBS``.
        """
        newest_first = sorted(
            self._jobs.values(),
            key=lambda job: job.submitted_at,
            reverse=True,
        )
        sliced = newest_first[: max(0, limit)]
        return {
            "total": len(self._jobs),
            "count": len(sliced),
            "jobs": [job.model_dump(mode="json") for job in sliced],
        }

    def stats(self) -> dict[str, Any]:
        """Return queue, fleet and in-flight counters."""
        queue_stats = self.queue.stats()
        nodes = self.registry.all()
        return {
            "queue": queue_stats.as_dict(),
            "nodes": {
                "total": len(nodes),
                "healthy": sum(
                    1
                    for node in nodes
                    if node.health not in (NodeHealth.LOST, NodeHealth.OFFLINE)
                ),
                "by_health": {
                    health.value: sum(1 for node in nodes if node.health is health)
                    for health in NodeHealth
                },
            },
            "inflight": {
                "interactive": self._inflight_interactive,
                "flex": self._inflight_flex,
            },
            "draining": self._draining,
            "tracked_jobs": len(self._jobs),
        }

    # -- scheduling loops ---------------------------------------------------

    async def _interactive_loop(self) -> None:
        """Dispatch latency-bound work immediately."""
        while self._running:
            try:
                # Jobs that ran out of latency budget are terminal answers:
                # they are released to their waiters before anything else.
                for expired in self.queue.drain_expired():
                    self._on_job_finished(expired)
                if self._inflight_interactive < self.config.scheduling.interactive_concurrency:
                    job = self.queue.pop_interactive(time.time())
                    if job is not None:
                        self._inflight_interactive += 1
                        asyncio.create_task(self._run_interactive(job))
                        continue
                await self._wait_for_work(self.config.scheduling.poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("interactive loop error: %s", exc)
                await asyncio.sleep(self.config.scheduling.poll_interval_seconds)

    async def _flex_loop(self) -> None:
        """Hold Flex work in a window, then dispatch it in batches."""
        while self._running:
            try:
                if self._inflight_flex < self.config.scheduling.flex_concurrency:
                    batches = self.queue.plan_flex(
                        time.time(),
                        self.window_policy,
                        power_ok=self._flex_power_ok(),
                        fleet_idle=self._fleet_idle(),
                    )
                    for batch in batches:
                        self._inflight_flex += 1
                        asyncio.create_task(self._run_flex(batch))
                    if batches:
                        continue
                await self._wait_for_work(self.config.scheduling.poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("flex loop error: %s", exc)
                await asyncio.sleep(self.config.scheduling.poll_interval_seconds)

    async def _wait_for_work(self, timeout: float) -> None:
        """Wait until the queue raises its wakeup event, or the interval ends.

        The event is the fast path: an interactive caller must not pay the
        poll interval for work that is already waiting. The interval stays
        the safety net for the time-based conditions (windows, deadlines)
        that no event announces.
        """
        try:
            await asyncio.wait_for(self.queue.wakeup.wait(), timeout=timeout)
        except TimeoutError:
            return
        finally:
            self.queue.wakeup.clear()

    async def _run_interactive(self, job: InferenceJob) -> None:
        try:
            await self._execute(job)
        finally:
            self._inflight_interactive -= 1

    async def _run_flex(self, batch: FlexBatch) -> None:
        try:
            plans = self.planner.plan(list(batch.jobs), self.registry.all())
            if not plans:
                LOGGER.warning(
                    "flex batch %s could not be routed; returning %d jobs to the queue",
                    batch.model,
                    batch.size,
                )
                for job in batch.jobs:
                    self.queue.requeue_front(job, reason="unroutable")
                return
            self.metrics.observe("horizon_flex_batch_size", float(batch.size))
            self.metrics.incr(
                "horizon_flex_batches_total", reason=batch.reason, model=batch.model
            )
            LOGGER.info(
                "dispatching flex batch model=%s size=%d reason=%s",
                batch.model,
                batch.size,
                batch.reason,
            )
            for plan in plans:
                for job, decision in zip(plan.jobs, plan.decisions, strict=True):
                    await self._execute(job, decision)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("flex batch failed: %s", exc)
        finally:
            self._inflight_flex -= 1

    async def _execute(self, job: InferenceJob, decision: Any = None) -> None:
        """Route and start one job."""
        try:
            if decision is None:
                decision = self.dispatcher.route(job)
            await self.dispatcher.dispatch(job, decision)
        except NoCapacityError as exc:
            LOGGER.warning("deferring job %s: %s", job.job_id, exc)
            job.state = JobState.QUEUED
            job.not_before = utcnow() + timedelta(seconds=2)
            self.queue.requeue_front(job, reason="no-capacity")
            self.metrics.incr("horizon_jobs_deferred_total")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("job %s failed: %s", job.job_id, exc)
            job.state = JobState.FAILED
            job.error = str(exc)
            # A failed dispatch is terminal too: the waiter is released and
            # the terminal envelope is journalled so a restart does not
            # resurrect the job.
            self._on_job_finished(job)

    # -- recovery hooks -----------------------------------------------------

    def _requeue(self, job: InferenceJob) -> InferenceJob:
        """Re-queue a job at the head of its lane."""
        self._jobs[job.job_id] = job
        return self.queue.requeue_front(job, reason="recovery")

    async def _dispatch_recovered(self, job: InferenceJob, excluded: frozenset[str]) -> None:
        """Dispatch a recovered job, avoiding the nodes that already failed it.

        Raises when the mesh cannot take the job; the recovery coordinator
        re-queues it so the scheduler retries later.
        """
        decision = self.dispatcher.route(job, excluded=excluded)
        await self.dispatcher.dispatch(job, decision)

    async def _recover(self, job: InferenceJob, node_id: str, reason: str) -> RecoveryOutcome:
        """Recover a job whose node died."""
        outcome = await self.recovery.recover(job, node_id, reason=reason)
        if outcome.dispatched:
            self.metrics.incr("horizon_jobs_recovered_total", resumed=str(outcome.resumed).lower())
        self.metrics.observe("horizon_recovery_seconds", outcome.elapsed_seconds)
        return outcome

    async def _on_node_lost(self, node_id: str, reason: str) -> None:
        """Watchdog callback: a node went silent."""
        await self.supervisor.on_node_lost(node_id, reason)

    async def _on_node_draining(self, node_id: str) -> None:
        """Watchdog callback: a node announced an imminent shutdown."""
        LOGGER.warning("migrating work off draining node %s", node_id)
        await self.supervisor.on_node_lost(node_id, "node announced draining")

    # -- helpers ------------------------------------------------------------

    def _ollama_factory(self, node: Any) -> OllamaClient:
        """Build an Ollama client for a node that executes directly.

        Clients are cached per endpoint: every dispatch would otherwise open
        (and leak) its own HTTP session.
        """
        base_url = getattr(node, "ollama_url", None) or self.config.ollama.base_url
        client = self._ollama_clients.get(base_url)
        if client is None:
            client = OllamaClient(
                base_url,
                timeout_seconds=self.config.ollama.timeout_seconds,
                keep_alive=self.config.ollama.keep_alive,
            )
            self._ollama_clients[base_url] = client
        return client

    async def _close_ollama_clients(self) -> None:
        """Close every cached Ollama client."""
        for client in self._ollama_clients.values():
            await client.close()
        self._ollama_clients.clear()

    async def _assign(self, node_id: str, frame: dict[str, Any]) -> None:
        """Send an assignment frame to an agent container."""
        await self.manager.assign(node_id, frame)

    def _flex_power_ok(self) -> bool:
        """True when at least one node may accept sleep-time work right now."""
        threshold = self.config.window.power_headroom_threshold
        gated = set(self.config.window.power_gated_tiers)
        for node in self.registry.all():
            if node.health in (NodeHealth.LOST, NodeHealth.OFFLINE):
                continue
            if node.batch_headroom <= 0:
                continue
            if node.power_class.value in gated and node.power_headroom < threshold:
                continue
            return True
        return False

    def _fleet_idle(self) -> bool:
        """True when nothing is running anywhere in the mesh.

        With an idle fleet there is no batch to amortise against, so holding
        Flex work back only adds latency.
        """
        if self._inflight_interactive or self._inflight_flex:
            return False
        return not self.supervisor.active_jobs()

    def _queue_depth(self) -> int:
        """Total number of jobs waiting in the queue."""
        stats = self.queue.stats()
        return stats.interactive + stats.flex

    def _prune_jobs(self) -> None:
        """Keep the job store bounded.

        Finished jobs are dropped oldest first: they are the ones nobody
        queries again, and an unbounded store would grow for the life of the
        broker.
        """
        if len(self._jobs) <= MAX_TRACKED_JOBS:
            return
        finished = [
            job_id
            for job_id, job in self._jobs.items()
            if job.state in TERMINAL_STATES
        ]
        if not finished:
            return
        finished.sort(key=lambda job_id: self._jobs[job_id].submitted_at)
        excess = len(self._jobs) - MAX_TRACKED_JOBS
        for job_id in finished[:excess]:
            self._jobs.pop(job_id, None)
            self._usage.pop(job_id, None)
        LOGGER.info("pruned %d finished job(s) from the job store", min(excess, len(finished)))
