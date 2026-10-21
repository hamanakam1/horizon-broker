# Architecture

## Directory tree

```text
horizon-broker/
├── horizon_broker/
│   ├── __init__.py                 # public package surface
│   ├── __main__.py                 # CLI: `python -m horizon_broker serve`
│   ├── config.py                   # dataclass config + YAML/env layering
│   │
│   ├── contracts/                  # (1) ASYNCHRONOUS DATA CONTRACTS
│   │   ├── jobs.py                 #   InferenceJob, InferencePayload, PriorityTier, JobState
│   │   ├── nodes.py                #   NodeCapability, NodeRole, NodeHealth, PowerClass
│   │   ├── recovery.py             #   FaultRecoverySnapshot, PartialToken, ResumePoint
│   │   ├── ledger.py               #   LedgerRecord, LedgerCommitReceipt
│   │   └── errors.py               #   broker error taxonomy
│   │
│   ├── ingress/                    # (2) API INGRESS
│   │   ├── app.py                  #   aiohttp routes, tier categorisation
│   │   └── server.py               #   runner lifecycle
│   │
│   ├── scheduler/                  # (3) BATCHING & QUEUE ENGINE
│   │   ├── engine.py               #   the two lanes: interactive + flex
│   │   ├── priority_queue.py       #   deadline-ordered interactive lane, per-model flex lane
│   │   ├── window.py               #   windowing policy (X seconds, min/max batch)
│   │   ├── batcher.py              #   batch assembly and per-node grouping
│   │   ├── dispatch.py             #   router -> supervisor hand-off
│   │   └── priority_queue.py
│   │
│   ├── routing/                    # (5) PD-DAG ROUTER
│   │   ├── arithmetic.py           #   arithmetic intensity model (prefill vs decode)
│   │   ├── pd_dag.py               #   the router itself
│   │   └── placement.py            #   node scoring and selection
│   │
│   ├── nodes/                      # (4) HEARTBEAT & FAULT DETECTION
│   │   ├── registry.py             #   the fleet view, health, quarantine
│   │   ├── connection.py           #   WebSocket control channel + reconnect loop
│   │   ├── manager.py              #   owns one channel per node
│   │   ├── watchdog.py             #   silence -> LOST -> recovery
│   │   └── executor.py             #   job supervision, checkpoint cadence
│   │
│   ├── recovery/                   # (6) STATE RECOVERY / EDGERECORD HOOK
│   │   ├── ledger.py               #   EdgeRecord HTTP port
│   │   ├── journal.py              #   local write-ahead journal (fallback)
│   │   ├── snapshot_store.py       #   write path + read path with fallback
│   │   └── recovery.py             #   the recovery sequence
│   │
│   ├── ollama/
│   │   └── client.py               #   streaming Ollama client
│   │
│   └── observability/
│       ├── metrics.py              #   in-process Prometheus registry
│       └── logging.py              #   structured JSON logging
│
├── schemas/                        # generated JSON Schema (contract wire format)
├── config/broker.example.yaml      # annotated configuration
├── docs/                           # architecture, contracts, runbook
├── scripts/generate_schemas.py
└── tests/
```

The four concerns the design separates deliberately:

* **Ingress** (`ingress/`) never touches the queue. It validates, categorises
  and hands over.
* **Batching queue logic** (`scheduler/`) never talks to a node directly. It
  produces dispatch batches and hands them to the dispatcher.
* **Fault detection** (`nodes/watchdog.py`, `nodes/connection.py`) only reports
  verdicts. It does not decide what happens next.
* **State recovery** (`recovery/`) owns the EdgeRecord contract and the
  resume protocol. It is reachable from the supervisor without importing the
  scheduler.

## The two lanes

```text
                    submit
                      │
                      ▼
              ┌───────────────┐
              │  JobPriority  │
              │    Queue      │
              └───────┬───────┘
        interactive   │   flex
                      │
    ┌─────────────────┴──────────────────┐
    ▼                                    ▼
INTERACTIVE LANE                     FLEX LANE (sleep-time)
pop by deadline,                     hold in a per-model window
dispatch immediately                  until: max batch reached
                                     or: min batch + dwell reached
                                     or: window expires
                                            │
                                            ▼
                                     BatchPlanner routes each job,
                                     groups by node, dispatches together
```

Both lanes share one `Dispatcher`, one `PDDagRouter` and one `JobSupervisor`.

### Windowing

The window opens when the first Flex job for a model arrives. It closes when
the batch reaches `max_batch_size`, when the batch reaches `min_batch_size`
*and* the window has been open for `min_dwell_seconds`, or when
`window_seconds` elapse. Windows are per model: prompts for different models
cannot share a weight load, so mixing them would defeat the purpose.

The payoff is the MFU argument. Streaming weights out of unified memory once
for a batch of 16 costs the same as streaming them once for a single prompt;
only the per-token compute scales with batch size.

## PD-DAG routing

The router turns a payload into an arithmetic profile:

```text
prefill_flops  = 2 * P * L  +  4 * L^2 * d          (compute bound)
decode_flops   = 2 * P * N                          (memory bound)
decode_bytes   = (2 * P / batch) * N + kv_bytes * N
```

| Shape | Phase | Placement |
| --- | --- | --- |
| massive prompt, short output | `PREFILL_BOUND` | Node Type A — prefill-optimised (M3 Max) |
| short prompt, massive output | `DECODE_BOUND` | Node Type B — decode-optimised (Pi cluster) |
| both heavy, both node types present | `PREFILL_BOUND` + split | prefill on A, decode on B, KV handed over |
| neither dominates | `BALANCED` | general-purpose node, least loaded wins |

The split mode requires the engine to support KV-cache hand-off
(`ModelProfile.supports_kv_handoff`). The contract carries
`kv_cache_state_id` and `InferenceJob.phase` for exactly this, but Ollama
does not expose a KV-cache export endpoint yet — so the split is currently
negotiated but not executed. The direct modes are fully live.

## Fault timeline

The requirement: a node may die at any moment, and the work must be running
elsewhere within 30 seconds.

```text
 t+0.0s   node dies (thermal kill, power loss, cable)
 t+0.0s   socket closes, or the heartbeat goes silent
 t+6.0s   watchdog declares the node LOST, quarantines it
          (2 s heartbeat interval, 3 missed beats)
 t+6.0s   supervisor asks the recovery coordinator for every job on the node
 t+6.1s   coordinator loads the newest FaultRecoverySnapshot from EdgeRecord
          (falls back to the local journal when the ledger is down)
 t+6.2s   coordinator validates the snapshot against the job
          (prompt hash + model must match, else the job restarts clean)
 t+6.3s   job is re-dispatched to a node that did not fail it
 t+6.5s   generation resumes from the recorded decode step
          ────────────────────────────────
          well inside the 30 s budget
```

The checkpoint that makes this possible is written *continuously*, not at
failure time: every `checkpoint_every_tokens` (64) or
`checkpoint_every_seconds` (10) the supervisor commits a
`FaultRecoverySnapshot` carrying the decode step, the partial tokens and the
KV-cache reference.

### Two durability layers

| Layer | Survives | Used when |
| --- | --- | --- |
| Local journal (`var/horizon-broker/journal.jsonl`) | Broker crash, ledger outage | Always written first |
| EdgeRecord ledger | Node death, broker death, fleet rebuild | Shared source of truth |

Write order is journal first, ledger second. Read order is ledger first,
journal second. A ledger outage therefore degrades durability to
machine-local, never to "lost".

## Node control channel

Two execution modes are supported:

* `ollama_direct` — the broker streams the generation over Ollama's HTTP API
  (used by the M3 Max and by Pi nodes running their own Ollama).
* `agent_container` — the broker assigns the job over the node's WebSocket
  and the container reports progress, checkpoints and completion back.

```text
broker ──ping──▶ node
node   ──heartbeat──▶ broker   (capability snapshot, power headroom)
broker ──assign──▶ node       (job envelope + resume point)
node   ──progress──▶ broker
node   ──checkpoint──▶ broker (decode step, partial tokens, kv_cache_ref)
node   ──complete──▶ broker
node   ──draining──▶ broker   (low voltage: migrate the work now)
```

Any socket close, or silence longer than `heartbeat.timeout_seconds`, is a
node loss.

## Recovery semantics

The recovery coordinator is deliberately conservative:

* the snapshot must match the job's prompt hash and model, otherwise it is
  discarded and the job restarts from scratch on the new node — degraded, but
  correct;
* the recovered job is dispatched **directly**, not pushed back into the
  shared queue. Re-queueing would let the scheduler lane pick it up a second
  time and dispatch it to the node that just failed;
* the failed node is added to the job's exclusion set, so attempt N+1 never
  lands on the machine that killed attempt N;
* the retry budget is finite (`recovery.max_attempts`). After that the job
  fails loudly instead of looping forever.

## Architecture diagram

The interactive version of this map is published to the dashboard
(`http://127.0.0.1:7878/#archify`, artifact
`7c81f2978e534e9c874d6f9bd185b4da`). The typed specification lives at
`docs/diagrams/horizon-broker.architecture.json` and the delivery receipt at
`docs/diagrams/delivery-receipt.json`.

## Deliberate limits

* One broker owns one fleet. There is no broker-to-broker consensus; a
  second broker on the same network would double-dispatch. The multi-broker
  story needs a leader election and a shared queue, which is out of scope
  here.
* KV-cache hand-off between nodes is a contract without an engine behind it
  yet.
* The broker assumes it is the only writer for a given job id. Job ids are
  UUIDs, so collisions are not a practical concern.
