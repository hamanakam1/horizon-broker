# Runbook

Operating The Horizon Broker on a fleet where hardware failure is normal.

## Health at a glance

```sh
curl -s http://127.0.0.1:8791/healthz     # liveness
curl -s http://127.0.0.1:8791/readyz      # readiness: at least one healthy node
curl -s http://127.0.0.1:8791/v1/queue    # queue occupancy and in-flight counters
curl -s http://127.0.0.1:8791/v1/nodes    # per-node capability, health and load
curl -s http://127.0.0.1:8791/metrics     # Prometheus exposition
```

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

The trade-off is durability, not availability: while the ledger is down, the
recovery state only exists on the broker's own disk. A broker crash during a
ledger outage loses the resume state (the job itself is not lost — it fails
loudly and can be resubmitted).

Restore the ledger, then the next checkpoint re-synchronises it.

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

## Throughput tuning

| Symptom | Lever |
| --- | --- |
| MFU too low, batches too small | `window.min_batch_size` down, `window.max_batch_size` up |
| Flex work starts too late | `window.window_seconds` down, or `window.idle_flush_seconds` down |
| A lone Flex job waits for a batch that never comes | `window.idle_flush` is on by default; check `horizon_flex_batches_total{reason="fleet_idle"}` |
| Interactive requests queue behind Flex work | separate the lanes: `scheduling.interactive_concurrency` |
| Trailer battery drains during sleep-time runs | `window.power_headroom_threshold` up |

## Adding a node

1. Add the node to `config/broker.yaml` under `nodes:` with its role, memory,
   model whitelist and power class.
2. For an agent container, set `execution_mode: agent_container` and
   `agent_ws_url`. The broker opens the channel and sends `ping` frames; the
   node answers `heartbeat` with its capability snapshot.
3. For a direct Ollama node, set `execution_mode: ollama_direct` and
   `ollama_url`.
4. Restart the broker (or `POST /v1/nodes/{node_id}/heartbeat` for the HTTP
   fallback path).

## Restarting the broker

The broker is safe to restart: in-flight jobs are lost (they fail with
`max attempts exceeded` once the retry budget is spent), but the *queue* is
recoverable from the journal. Job payloads are journalled on submission, so a
restarted broker can rebuild the envelope of a job it was running.

Snapshots survive in EdgeRecord, which is the point: the ledger is what makes
recovery survive a broker crash.

`SIGTERM` drains instead of cutting: the ingress stops accepting work
(`503`), the in-flight jobs are finished (up to
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
  timeout_seconds: 6.0       # silence after which the node is unreachable
  missed_beats_before_loss: 3
recovery:
  recovery_budget_seconds: 30.0
```

Lowering `timeout_seconds` detects loss faster but risks false positives on a
congested mesh. The rule of thumb: `timeout_seconds >= 3 * interval_seconds`.
