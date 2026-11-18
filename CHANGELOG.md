# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/); the
project follows [semantic versioning](https://semver.org/). Every entry names
the evidence behind it — a metric, a command or a test — because a claim
without evidence does not ship.

## [0.4.0] — 2026-11-12

The control-plane release: two brokers can share a fleet without
double-dispatching. 799 tests pass.

### Added

- **The fleet lease** (`horizon_broker/control/lease.py`): an advisory lock
  over a shared file. The operating system releases it when the process dies,
  so a crashed broker cannot strand the fleet and a follower takes over
  without a timeout. The owner is written into the file, so an operator can
  see who owns the fleet without asking a running process.
- **The follower's refusal**: a broker that does not own the fleet answers
  an admission with `503 broker is a follower` and names the owner. The fleet
  view stays open on a follower. `/readyz` reports `follower`.
- **Automatic takeover**: a follower keeps asking for the lease, and the
  first one to get it owns the fleet from then on.
- **The shared queue is claimed from the journal**: a broker that owns the
  fleet takes the unclaimed jobs from the journal and records that it is
  running them, so a broker that dies does not take the queue with it.
- **The live two-broker drill** (`scripts/two_broker_drill.py`): two real
  brokers, one lease, one submission each. The owner admits, the follower
  refuses, and the follower takes over when the owner stops.
- `docs/CONTROL-PLANE.md` — the design for the rest of Phase 2.

### Changed

- `broker.lease_path` points a broker at the shared lease file; empty
  means one broker owns its fleet unconditionally (the single-broker
  case is unchanged).
- `HORIZON_BROKER_LEASE_PATH` sets the lease path from the environment.
- The broker's own jobs are counted as live occupancy, so the broker
  never books a node beyond what it can take; the node's report corrects
  the count, and a job the broker knows is finished reduces it.
- A Flex batch reaches the engine as a wave (2.85x measured on a local
  Ollama), not one prompt after another.
- The planner books a node only as far as it can take; the rest stays in
  the queue, and a stale plan is refreshed at dispatch time.

### Known limits

- The shared queue is the journal plus claims; a real out-of-process queue
  is still Phase 2's next step.
- The KV hand-off stays an opt-in protocol that no shipped engine enables.
- The seven-day fault trial needs the real fleet.

## [0.3.0] — 2026-11-10

The "truth" release: the behaviour and the documentation say the same thing.
Phase 0 of [`docs/ROADMAP.md`](docs/ROADMAP.md) is closed; 703 tests pass.

### Fixed

- A node that failed once returns to the fleet. The quarantine honours
  `recovery.quarantine_seconds` instead of excluding the node for the life of
  the process, and the watchdog returns a node that answers again
  (`horizon_broker/routing/placement.py`, `horizon_broker/nodes/watchdog.py`).
- A `draining` frame stores a `NodeHealth`, not the raw string, so the
  drain-and-migrate path is reachable (`horizon_broker/nodes/executor.py`).
- Every terminal path releases its waiter: completion, failure, cancellation
  and an expired latency budget. An OpenAI client no longer hangs until its
  own timeout (`horizon_broker/scheduler/engine.py`).
- A dropped socket no longer fails the running job; the watchdog decides
  whether the node is lost.
- Power gating is per node: a solar trailer below
  `window.power_headroom_threshold` is skipped for Flex work only.
- A config file with a `window:` section no longer disables power gating, and
  `broker.journal_max_bytes` is read from the file.
- The container journals into the mounted volume
  (`HORIZON_BROKER_JOURNAL_PATH`), so compose deployments keep their recovery
  state.
- The EdgeRecord port speaks one envelope. The client reads
  `{"records": [...]}` and refuses the legacy shapes unless
  `ledger.accept_alternate_envelopes` says so; the returned content hash is
  verified against the payload and the record must carry this broker's owner
  tag.
- A job that already finished cannot be cancelled: `DELETE` answers `409`
  instead of rewriting a completed job as cancelled.
- One caller's timeout no longer strands its peers: every waiter owns its own
  future, because `asyncio.wait_for` cancels the future it times out on.

### Added

- `GET /v1/jobs` lists the tracked jobs, newest first (`?limit=`, capped at
  1000); the operator CLI shows them in a table and gained a `queue` command.
- The queue replays from the journal on a cold start: a job submitted before
  the crash is queued again, counted in `horizon_jobs_restored_total`.
- `stream: true` streams. The supervisor publishes every piece of the answer
  as the engine produces it and the OpenAI ingress writes one delta per
  piece.
- The metrics tell the truth: the completion and failure counters move, the
  missing metrics are declared, no metric name carries a bare and a labelled
  series, and every summary publishes p50, p95 and p99.
- The error taxonomy is used: the recovery outcome carries the typed failure,
  `UnknownJobError` drives the 404, an expired job is a `JobExpiredError`, and
  `NoCapacityError` is a `RoutingError`.
- The broker's wordmark: in the README, in `horizon-broker version` and on
  the terminal when the broker starts.
- `make check` — the pre-proposal gate (lint, strict typecheck, schema
  drift) — and a `py.typed` marker so downstream type checkers see the
  annotations.

### Changed

- The scheduler awaits the queue's wakeup event: an interactive caller no
  longer pays a poll interval for work that is already waiting. The poll
  interval stays the fallback for the time-based conditions no event
  announces.
- The generation budget is honoured: the Ollama client bounds a generation
  with the configured timeout instead of `ClientTimeout(total=None)`.
- Both ingresses agree on the tier: the OpenAI path uses
  `scheduling.interactive_latency_threshold_ms` like the native path.
- The system prompt travels in `payload.system` only; it was also left in the
  message list and therefore sent to the engine twice.
- The standard GitHub Actions are pinned to release commits.

### Known limits

- One broker owns one fleet. Leader election and a shared queue are Phase 2
  and are not implemented.
- A direct Ollama node still receives one request per prompt; the batch is a
  grouping, not a single multi-prompt call.
- The KV hand-off is an opt-in protocol that no shipped engine enables.

[0.3.0]: https://github.com/hamanakam1/horizon-broker/releases/tag/v0.3.0
