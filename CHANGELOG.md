# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/); the
project follows [semantic versioning](https://semver.org/). Every entry names
the evidence behind it — a metric, a command or a test — because a claim
without evidence does not ship.

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
