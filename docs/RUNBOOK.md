# Runbook

Operating The Horizon Broker on a fleet where hardware failure is normal.

## Health at a glance

```sh
curl -s http://127.0.0.1:8791/healthz     # liveness
curl -s http://127.0.0.1:8791/readyz      # readiness: at least one healthy node
curl -s http://127.0.0.1:8791/v1/queue    # queue occupancy and in-flight counters
curl -s http://127.0.0.1:8791/v1/nodes    # per-node capability, health and load
curl -s http://127.0.0.1:8791/v1/jobs     # the tracked jobs, newest first
curl -s http://127.0.0.1:8791/metrics     # Prometheus exposition
```

The `/metrics` response declares `horizon_ledger_fallback_total` even before
an outage occurs, with `commit`, `query` and `invalid` reason series ready for
alerts and dashboards.

`horizon_journal_replay_failures_total` increments when startup cannot read
the local journal; treat any non-zero value as a recovery incident.

`readyz` returns 503 when no node is healthy. That is the signal to stop
submitting work and investigate the fleet.

## A node died

**What the broker does automatically**

1. The socket closes, or the heartbeat goes silent for
   `heartbeat.timeout_seconds` (6 s by default).
2. The watchdog marks the node `lost` and quarantines it for
   `recovery.quarantine_seconds` (120 s).
3. Every job that was running on the node is recovered: the newest
   `FaultRecoverySnapshot` is loaded from EdgeRecord (or the local journal),
   validated against the job, and re-dispatched to a node that did not fail
   it.
4. The failed node is added to the job's exclusion set, so a retry never lands
   on the machine that just died.

**What you should see**

```sh
curl -s http://127.0.0.1:8791/v1/nodes | python3 -m json.tool | grep -A3 '"health": "lost"'
curl -s http://127.0.0.1:8791/metrics | grep horizon_jobs_recovered_total
```

**If recovery did not happen**

* Check the logs for `node <id> declared LOST` and `recovery for job <id>`.
* `horizon_recovery_seconds` should stay below 30. A value above it means the
  budget was exceeded — usually a slow ledger.
* A job that reports `max attempts exceeded` is dead. It stays queryable at
  `GET /v1/jobs/{job_id}` and can be resubmitted.

## The ledger (EdgeRecord) is down

The broker keeps working: snapshots are written to the local journal first and
the ledger commit is retried on the next checkpoint. Recovery falls back to
the journal.

```sh
curl -s http://127.0.0.1:8791/metrics | grep horizon_
grep -c '"kind": "snapshot"' var/horizon-broker/journal.jsonl
```

`horizon_ledger_fallback_total{reason="commit"}` counts snapshots kept only
in the local journal after a failed commit. `reason="query"` counts recovery
reads that could not reach the ledger, and `reason="invalid"` counts ledger
records rejected during snapshot validation.

A ledger that answers with 5xx or drops the connection is retried
(`ledger.max_retries`, growing backoff) before the commit is given up on. A
4xx means the *document* is wrong — that is a bug, not an outage: the broker
logs `ledger commit rejected` and keeps the snapshot in the journal.

The trade-off is durability, not availability: while the ledger is down, the
recovery state only exists on the broker's own disk. A broker restart can
replay that journal, but loss of the journal or its machine loses the shared
resume copy (the job itself is not lost — it fails loudly and can be
resubmitted). The journal is trimmed back to
`broker.journal_max_bytes` (64 MiB by default), keeping the newest records.

Restore the ledger, then the next checkpoint re-synchronises it.

## The ledger rejects the commits (4xx)

```sh
grep 'ledger commit rejected' /tmp/horizon-broker.log
```

A rejection is not an outage: the broker is writing a document the ledger
does not accept. Check the collection name, the document id and the payload
shape against the ledger's own contract. The snapshots are still in the
journal, so no recovery state is lost while the problem is fixed.

The broker also rejects malformed successful replies. A commit response with
invalid JSON or mismatched identity, or a query record without a document ID,
mapping payload, matching `job_id`, or valid `content_hash`, is treated as
ledger-unavailable and recovery falls back to the local journal. This is
intentional: accepting a successful but unrelated or tampered record would be
worse than using the local snapshot.

## The solar trailer is draining

A trailer node announces `draining` before it loses power. The broker
migrates its work immediately, the same way as a node loss, but with a known
cause.

```sh
curl -s http://127.0.0.1:8791/v1/nodes | grep -B4 draining
```

Flex (sleep-time) work is not dispatched to a power-gated node while
`power_headroom` is below `window.power_headroom_threshold` (0.25). Interactive
work still goes through: a human is waiting.

## The queue is backing up

```sh
curl -s http://127.0.0.1:8791/v1/queue
```

* `queue.flex` high, `queue.open_windows` > 0 — the window is still filling.
  Shorten `window.window_seconds` to dispatch sooner, or raise
  `window.max_batch_size` to dispatch more per batch.
* `queue.flex` high and `queue.expired` climbing — jobs are waiting longer
  than their latency budget. Raise the node count, or lower
  `window.min_batch_size` so partial batches go out.
* `horizon_jobs_deferred_total` climbing — the router could not place the
  work (no node serves that model, or the fleet is at capacity). Check
  `model_whitelist` on the nodes.
* `horizon_batch_active_members` should return to zero after a batch drains.
  If it remains positive, inspect the node's event stream for missing
  per-job `complete`/`error` frames; the supervisor tracks and recovers each
  member independently.

## Throughput tuning

| Symptom | Lever |
| --- | --- |
| MFU too low, agent batches too small | `window.min_batch_size` down, `window.max_batch_size` up; confirm the node advertises `metadata.supports_batch: true` |
| Flex work starts too late | `window.window_seconds` down, or `window.idle_flush_seconds` down |
| A lone Flex job waits for a batch that never comes | `window.idle_flush` is on by default; check `horizon_flex_batches_total{reason="fleet_idle"}` |
| Interactive requests queue behind Flex work | separate the lanes: `scheduling.interactive_concurrency` |
| Trailer battery drains during sleep-time runs | `window.power_headroom_threshold` up |
| Batch frames are rejected | Set `metadata.supports_batch: true` only on nodes that implement the `batch_assign` frame; inspect `horizon_batches_rejected_total` |

## Adding a node

1. Add the node to `config/broker.yaml` under `nodes:` with its role, memory,
   model whitelist and power class.
2. For an agent container, set `execution_mode: agent_container` and
   `agent_ws_url`. The broker opens the channel and sends `ping` frames; the
   node answers `heartbeat` with its capability snapshot.
3. An agent container may opt into one `batch_assign` admission frame by
   advertising `metadata.supports_batch: true`. Completion, progress,
   checkpoint and recovery frames remain keyed by individual `job_id` values.
4. For a direct Ollama node, set `execution_mode: ollama_direct` and
   `ollama_url`.
4. Restart the broker (or `POST /v1/nodes/{node_id}/heartbeat` for the HTTP
   fallback path).

## Restarting the broker

The broker is safe to restart: the *queue* is rebuilt from the journal. Job
payloads are journalled on submission, and their newest envelope (including
the terminal states `completed`, `failed` and `cancelled`) is journalled when
the state changes. On a cold start the broker re-queues every job whose
newest journalled state is not terminal and whose attempt budget is not
spent, and counts them in `horizon_jobs_restored_total`.

Snapshots survive in EdgeRecord, which is the point: the ledger is what makes
recovery survive a broker crash. `SIGTERM` drains instead of cutting: the
ingress stops accepting work (`503`), the in-flight jobs are finished (up to
`broker.drain_timeout_seconds`, 30 s by default) and only then does the
process exit. `--no-drain` skips the wait; use it only when the node is
going away anyway.

## The queue is full (429)

The ingress refuses new work above `scheduling.max_queue_depth` (10 000 by
default). That is backpressure, not a failure: the caller should retry later.

```sh
curl -s http://127.0.0.1:8791/v1/queue | python3 -m json.tool
```

Raise the limit only when the fleet can actually absorb more; a deeper queue
just moves the wait from the caller to the agents.

## Tuning the detection budget

```yaml
heartbeat:
  interval_seconds: 2.0      # ping cadence
  timeout_seconds: 6.0       # minimum silence before loss
  missed_beats_before_loss: 3 # also require this many missed intervals
recovery:
  recovery_budget_seconds: 30.0
```

Loss detection waits for the later of `timeout_seconds` and
`interval_seconds * missed_beats_before_loss`. Lowering either limit detects
loss faster but risks false positives on a congested mesh. Keep the timeout at
least as large as the intended missed-beat window.
