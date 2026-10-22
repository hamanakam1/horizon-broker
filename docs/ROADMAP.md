# The Horizon Broker — long-term plan

**Status:** v0.2.0 (alpha) · 131 passing tests · one broker, one fleet
**Plan horizon:** v0.3 → v0.4 → v0.5 → v1.0
**Owner:** the operator of the edge fleet (a Mac, a Pi cluster, a solar trailer)

This document is the plan for how the Horizon Broker should be built from here.
It is grounded in the code as it stands today (`horizon_broker/`, 31 modules,
5 976 lines) and in a defect sweep of that code. Every work item below names
the file it touches. Every phase ends in a gate that can be measured, not
asserted.

---

## 1. Where the software stands

The broker is real. The scheduler, the router, the watchdog, the supervisor,
the checkpoint cadence and the recovery coordinator are implemented and
covered. Both ingresses work: the native `/v1/jobs` and the OpenAI-compatible
`/v1/chat/completions`, and both produce the same `InferenceJob`. A job
submitted on this machine really does run against a local Ollama engine and
really does come back with text.

What is *not* real yet:

| Claim in the docs | Reality |
| --- | --- |
| "a failed node is quarantined for `recovery.quarantine_seconds`" | the quarantine never expires — the node is lost for the life of the process (`horizon_broker/routing/placement.py:51`) |
| "the broker drains: in-flight jobs finish before exit" | the drain ignores flex work and never awaits the per-job tasks (`horizon_broker/scheduler/engine.py:200`, `:341`) |
| "the queue is recoverable from the journal" | the replay path exists but nothing calls it (`horizon_broker/recovery/snapshot_store.py:136`) |
| "the broker batches work so the node amortises the weight load" | the "batch" is N independent single-prompt HTTP requests (`horizon_broker/scheduler/engine.py:401`) |
| "covered by tests (85 of them)" | there are 131 (`README.md:206`) |
| `horizon_jobs_completed_total` | declared, exported, and never incremented — it is always `0.0` (`horizon_broker/scheduler/engine.py:75`) |

That gap between the documentation and the behaviour is the reason this
plan starts with correctness and not with features. A broker that loses a node
forever, hangs a waiting HTTP client, and exports a completion counter that
never moves cannot be given more work, however good the routing is.

---

## 2. The three properties the software must have

Everything below is in service of three properties. When a choice is open,
these decide it.

1. **Honest under failure.** A node dies at any moment. The broker must know
   within the heartbeat budget, re-dispatch within the recovery budget, and
   say what happened in a metric an operator can alert on.
2. **Worth the fleet.** The promise is throughput, not latency. If a Flex
   window does not measurably raise Model Flop Utilization over running the
   same work one prompt at a time, the window is theatre and must be removed.
3. **Safe to grow.** One broker owns one fleet today. That must stay true
   until a control plane makes it true by design rather than by accident.

---

## 3. Principles that do not change with the phase

These are the invariants from `AGENTS.md`, restated because every phase
tempts a violation of one of them.

- **The contracts are the wire format.** A field rename is a breaking change:
  bump `schema_version`, regenerate `schemas/`, update `docs/CONTRACTS.md`.
- **The ingress never touches the queue, the router or recovery.** It
  validates, categorises and hands over.
- **The scheduler never talks to a node.** It produces dispatch batches.
- **The watchdog only reports verdicts.** It never decides what happens next.
- **Recovery is idempotent.** Replaying a snapshot twice must not corrupt a
  job.
- **Every claim needs evidence.** A sentence in the README that says the
  broker recovers in 30 seconds needs a metric, a command and a test behind
  it. If the evidence does not exist, the sentence does not ship.
- **A defect is a bug until a test fails first.** No phase item closes without
  the reproducing test.

---

## 4. The phases

### Phase 0 — Truth (v0.3)

**Goal:** the broker's behaviour and its documentation say the same thing.

**Why first:** every later phase assumes that a node loss is detected, that
the job survives it, and that the metrics say so. Today none of the three is
true. Building throughput on top of that would multiply the damage.

**Work items**

| # | Item | Where | Why it matters |
| --- | --- | --- | --- |
| 0.1 | Expire the quarantine | `routing/placement.py:51`, `nodes/registry.py:76` | a node that failed once never returns to the fleet; `recovery.quarantine_seconds` is a dead setting |
| 0.2 | Store health as `NodeHealth`, not `str` | `nodes/executor.py:254`, `nodes/registry.py:59` | `model_copy(update=…)` skips validation, so a WS `draining` frame stores the string `"draining"`; the watchdog branches on identity, so the drain/migration path is unreachable |
| 0.3 | Resolve the waiter on every terminal path | `scheduler/engine.py:249`, `:284`, `:425`; `scheduler/priority_queue.py:165` | failure, cancellation and expiry never resolve the `asyncio.Future`, so an OpenAI client hangs until its timeout |
| 0.4 | Do not fail a job on a transient socket error | `nodes/connection.py:134` → `nodes/executor.py:299` | a one-second reconnect attempt is attributed to the running job and fails it instead of letting the watchdog declare the node LOST |
| 0.5 | Gate power per node, not per fleet | `scheduler/engine.py:493`, `routing/placement.py:32` | a solar trailer at 5 % headroom can still be selected while another node has headroom; `NodeSelector.power_gated_classes` is constructed empty, so the power penalty is unreachable |
| 0.6 | Keep `window.power_gated_tiers` when YAML omits it | `config.py:328` | any config file with a `window:` section silently disables power gating |
| 0.7 | Make the metrics true | `scheduler/engine.py:75`, `:87`; `observability/metrics.py:35` | increment `horizon_jobs_completed_total`; declare `horizon_recovery_seconds` and `horizon_splits_handed_off_total`; stop emitting an unlabelled and a labelled series under one metric name |
| 0.8 | Publish quantiles, not just sums | `observability/metrics.py:85` | the runbook says "recovery should stay under 30 s" but only `_count`/`_sum` are exported |
| 0.9 | Repair the documentation drift | `README.md:206`, `docs/CONTRACTS.md:221`, `docs/ARCHITECTURE.md:23` | 131 tests not 85; the checkpoint frame carries `tokens_generated`, not `step`; the contracts say to ignore unknown fields while every contract sets `extra="forbid"` |
| 0.10 | Enforce `broker.journal_max_bytes` from YAML and fix the container's journal path | `config.py:354`, `Dockerfile:20` | the setting is ignored, and the container writes its journal outside the mounted volume, so compose deployments persist no recovery state |
| 0.11 | Raise the errors that are never raised | `contracts/errors.py` | `UnknownJobError`, `JobExpiredError`, `RoutingError`, `RecoveryBudgetExceededError` and `MaxAttemptsExceededError` exist but are never raised; the failures they describe happen silently |

**Gate A — the fault-path suite is green.**

Not the unit suite: a suite that kills nodes, cuts sockets, stops the ledger
and asserts the *observable* outcome — the job completed on another node,
the metric moved, the operator was told. It must include the three cases the
current suite misses: a transient socket drop, a cancellation while a request
is waiting, and a node that recovers after its quarantine expires.

**Evidence:** `make test` green, the fault-path suite green, the metrics
diff showing the counters that now move, and the README corrected.

---

### Phase 1 — Throughput (v0.4)

**Goal:** the batching promise is real, and the broker survives its own
restarts.

**Why second:** after Phase 0 the broker is trustworthy, so it can be given
real work. Throughput is what makes the fleet worth owning.

**Work items**

| # | Item | Where | Why it matters |
| --- | --- | --- | --- |
| 1.1 | Send a real batch to the engine | `scheduler/batcher.py:4`, `scheduler/engine.py:401` | today "batching" is a grouping that changes the order of N single-prompt requests; the MFU argument only holds if the weight load is amortised across one call |
| 1.2 | Stream token by token | `ingress/app.py:216` | the whole result is buffered and emitted as one content delta; `stream: true` is not streaming |
| 1.3 | Honour `ollama.timeout_seconds` | `ollama/client.py:114` | `ClientTimeout(total=None)` disables the configured 600 s budget for the whole generation |
| 1.4 | One HTTP session for the whole broker | `scheduler/engine.py:70` | the injected session reaches only the ledger client; the connection manager, the probe and the Ollama clients each open their own pool |
| 1.5 | Wake the queue instead of polling it | `scheduler/priority_queue.py:70` | the interactive lane pays up to 250 ms of poll latency on every request; the `wakeup` event exists and is never awaited |
| 1.6 | Rebuild the queue from the journal after a restart | `recovery/snapshot_store.py:136`, `docs/RUNBOOK.md:139` | the runbook promises this; the code path is unreachable, so a broker restart loses queued work |
| 1.7 | Ship a real EdgeRecord adapter | `recovery/ledger.py:36` | the ledger port is a generic HTTP shim that accepts four different response shapes; a real client removes the guesswork and makes the hash verification meaningful |
| 1.8 | Make the OpenAI ingress agree with the native ingress | `ingress/openai.py:108` | the OpenAI path hard-codes a 120 s tier threshold while the native path uses `scheduling.interactive_latency_threshold_ms` (2 s) |
| 1.9 | Stop duplicating the system prompt | `ingress/openai.py:94`, `contracts/jobs.py:172` | the system message is copied into `payload.system` and left in `payload.messages`, so it is sent twice |
| 1.10 | Enforce the per-node batch limit end to end | `nodes/registry.py:94` | `max_batch_size` is tracked per node but never consulted when a Flex batch is grouped |

**Gate B — measured evidence from a soak.**

One week of the real fleet running real work, with the numbers attached:
Model Flop Utilization before and after the window, p50/p95/p99 queue
latency for both tiers, the recovery time distribution from the new
quantiles, and the journal-replay result from a cold restart.

**Evidence:** the soak report in `docs/`, the before/after MFU table, and a
green restart-replay test.

---

### Phase 2 — Scale (v0.5)

**Goal:** two brokers can share a fleet without double-dispatching, and the
broker survives being moved.

**Why third and last:** this is the most expensive and least valuable work
until the fleet is actually bigger than one machine can run. The current
limit is honest and documented; it is only dishonest if it is forgotten.

**Work items**

| # | Item | Where | Why it matters |
| --- | --- | --- | --- |
| 2.1 | Elect a leader per fleet | new `control/` package | two brokers on one network currently both accept and run the same job id |
| 2.2 | Move the queue out of process memory | new `control/` package | the job store is a dict in one process; a restart loses the queue until 1.6 lands |
| 2.3 | Namespace job ids by broker and fleet | `contracts/jobs.py:193` | ids are bare UUIDs, so a broker cannot tell its own work from another broker's |
| 2.4 | Lock the journal against two writers | `recovery/journal.py:33` | the journal is opened in append mode with no locking; two brokers interleave writes |
| 2.5 | Filter the ledger query by broker | `recovery/ledger.py:146` | the query is `?job_id=…` with no owner, so two brokers read each other's snapshots |
| 2.6 | Make the KV hand-off real, or delete it | `routing/pd_dag.py:85`, `nodes/executor.py:159` | the protocol is implemented and tested, but no engine can export a KV cache; either an engine appears or the split mode leaves the roadmap |
| 2.7 | Track per-job tasks and await them on drain | `scheduler/engine.py:341`, `:200` | `SIGTERM` closes the Ollama clients while generation tasks may still be streaming |

**Gate C — two brokers, one fleet, one job.**

A test (and a live drill) where two brokers point at the same fleet: the
leader holds the lease, the follower proxies, a job submitted to either
completes exactly once, and the journal and the ledger stay consistent.

**Evidence:** the leader-election drill, the journal-lock test, and the
updated "Deliberate limits" section in `docs/ARCHITECTURE.md`.

---

### Phase 3 — Proof (v1.0)

**Goal:** the broker is supported, not merely running.

**Work items**

- A seven-day edge trial on the real fleet (M3 Max, Pi cluster, solar
  trailer) with the fault injection switched on: pull the power, kill the
  network, overheat a node, let the sun go down.
- The SLOs written down and measured: recovery within 30 s at p99, no lost
  job across the trial, no duplicate execution, the recovery budget respected.
- The operator runbook rewritten from the drill: what the operator sees, what
  the broker does, what to do next, and how to tell the two apart.
- The contracts frozen at `schema_version 2.0` with the versioning rules
  written down honestly (breaking changes, the deprecation window, the
  compatibility matrix).
- The release process: a signed tag, a changelog, a rollback that is a
  `git revert` plus a config change, and a published artifact id for the
  architecture diagram that matches the code.

**Gate D — the SLOs hold under real faults.**

**Evidence:** the trial report, the SLO table with the measured numbers, and
the signed release.

---

## 5. Cross-cutting work

These run in every phase, not in one.

**Testing.** The unit suite protects the contracts, the windowing policy and
the router. The fault-path suite protects the promises. The soak protects the
performance claims. A change that touches the fault paths needs all three.

**Observability.** Every phase adds metrics before it adds behaviour, and
every metric is asserted in a test so it cannot silently stop moving. The
metric contract is: `horizon_jobs_submitted_total`, `horizon_jobs_completed_total`,
`horizon_jobs_failed_total`, `horizon_jobs_recovered_total`,
`horizon_jobs_deferred_total`, `horizon_jobs_duplicate_total`,
`horizon_flex_batches_total`, `horizon_splits_handed_off_total`,
`horizon_splits_degraded_total`, plus the latency distributions for the
queue, the recovery and the generation.

**Contracts.** `schemas/*.v1.json` is regenerated in CI and the diff is
enforced. A contract change is a version bump plus a migration note, never a
field rename in place.

**Documentation.** `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md` and
`docs/RUNBOOK.md` are part of the deliverable, not an afterthought. The
architecture diagram, the delivery workflow, the state machine and the
node-loss sequence are generated from typed specifications in
`docs/diagrams/` and are republished with every release.

**Release hygiene.** The version in `pyproject.toml`, the tag, the compose
image tag and the `entry_points.txt` artefact move together. `v0.2.0` is
current; the compose file still says `0.1.0`.

---

## 6. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| The EdgeRecord ledger is not available on the edge hardware | Recovery degrades to machine-local; a broker crash loses the resume state | the local journal is always written first (`recovery/snapshot_store.py`); the runbook states the degraded mode explicitly |
| No engine can export a KV cache | the PD split stays a protocol without a payload | Phase 2.6 forces a decision: an engine appears, or the split leaves the roadmap |
| The fleet's hardware fails in ways the watchdog cannot see | a silent node is only detected by silence | the probe covers direct Ollama nodes (`nodes/probe.py`); the remaining gap is a node that answers but does not work, which the checkpoint cadence will eventually expose |
| The operator's attention is the scarcest resource | a multi-year plan becomes a multi-year queue | each phase is independently valuable: v0.3 makes the broker trustworthy, v0.4 makes it worth the hardware, v0.5 makes it grow |
| Scope creep into a general job system | the broker becomes a platform instead of an edge broker | "Deliberate limits" stays; the three properties in section 2 are the filter for every new request |

---

## 7. What is deliberately not planned

- A hosted control plane. The fleet is local; the control plane is local.
- Multi-region failover. One fleet, one geography, one operator.
- A general-purpose durable queue. The broker's queue exists to route
  inference, not to be a message broker.
- Replacing Ollama. The broker speaks to engines; it is not an engine.
- A GUI. The operator CLI and the metrics are the interface.

---

## 8. The plan at a glance

| Version | Phase | Headline | Gate |
| --- | --- | --- | --- |
| v0.3 | Truth | the broker tells the truth about failures | fault-path suite green |
| v0.4 | Throughput | the batching promise is real, restarts are survivable | one-week soak with measured MFU |
| v0.5 | Scale | two brokers, one fleet, one job | leader-election drill |
| v1.0 | Proof | supported, signed, measured under real faults | seven-day edge trial, SLOs hold |

---

## 9. The diagrams

Four diagrams accompany this plan. Each is generated from a typed
specification in `docs/diagrams/`, validated to the showcase standard (nine
checks, zero errors, zero warnings) and published to the local dashboard at
`http://127.0.0.1:7878/#archify`.

| Diagram | Type | Artifact | Shows |
| --- | --- | --- | --- |
| Target architecture | architecture | `933154eab75945728198234c1b40bc61` | the v1.0 shape: what is built today (solid) and what Phase 2 adds (dashed) |
| Delivery workflow | workflow | `2c6f56fecbe44a07b74da30541ccee30` | how the software gets built: phase, gate, edge trial, release |
| Job lifecycle | lifecycle | `eb92809cc4f343ed986e81116ae2036a` | the `InferenceJob` state machine, including the recovery loop and the terminal exits |
| Node-loss sequence | sequence | `6d4236a2f1ef478cbbc7bd68438d06ee` | one job across a node loss, from acceptance to completion |

The live artifacts are at `/api/archify/artifacts/<artifact-id>`; the rendered
screenshots are in `docs/images/`.
