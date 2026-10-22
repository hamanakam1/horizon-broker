# Data contracts

All contracts are Pydantic v2 models with `extra="forbid"` and assignment
validation: a payload that contains an unknown field is rejected rather than
silently ignored. The JSON Schema for each contract is generated into
`schemas/*.v1.json` by `make schemas`.

## `InferenceJob`

`horizon_broker/contracts/jobs.py` · `schemas/inference_job.v1.json`

The unit of work tracked by the broker. Four fields define the scheduling
contract:

| Field | Meaning |
| --- | --- |
| `priority_tier` | `interactive` (latency bound) or `flex` (sleep-time compute) |
| `payload` | the inference request itself |
| `max_acceptable_latency_ms` | the latency budget, measured from submission |
| `kv_cache_state_id` | EdgeRecord document id of the resume state, if any |

The rest of the envelope:

```json
{
  "job_id": "0f2c6b5e-8f36-4a9f-9a1d-6b0f0d9c1f21",
  "priority_tier": "flex",
  "payload": {
    "model": "qwen2.5:7b",
    "prompt": "Map every call site of the ingestion pipeline",
    "messages": [],
    "system": null,
    "parameters": {
      "temperature": 0.8, "top_p": 0.95, "top_k": 40,
      "num_predict": 4096, "seed": null, "stop": [], "repeat_penalty": 1.1
    },
    "expected_output_tokens": 4096,
    "prompt_tokens_hint": null
  },
  "max_acceptable_latency_ms": 3600000,
  "kv_cache_state_id": null,
  "state": "windowing",
  "phase": "monolithic",
  "attempt": 0,
  "submitted_at": "2026-10-21T15:54:02.113Z",
  "deadline_at": "2026-10-21T16:54:02.113Z",
  "not_before": null,
  "assigned_node_id": null,
  "caller": "taskrivet",
  "tags": [],
  "resume_point": null,
  "result_text": null,
  "error": null,
  "schema_version": "1.0"
}
```

`deadline_at` is derived from `max_acceptable_latency_ms` when it is not
supplied. `state` moves through `queued → windowing → dispatched → running →
completed`, with `recovering` and `failed` as the fault branches.

### Token estimation

`payload.estimated_prompt_tokens` is `prompt_tokens_hint` when the caller
knows it, otherwise `len(prompt + system + messages) / 4`.
`payload.estimated_output_tokens` is `expected_output_tokens`, else
`parameters.num_predict`, else 512. These numbers drive the PD-DAG router, so
an honest hint is worth more than a precise one.

## `NodeCapability`

`horizon_broker/contracts/nodes.py` · `schemas/node_capability.v1.json`

The broker's view of one execution node.

```json
{
  "node_id": "m3-max",
  "role": "prefill",
  "is_prefill_optimized": true,
  "is_decode_optimized": false,
  "available_vram_bytes": 100000000000,
  "total_vram_bytes": 128000000000,
  "current_batch_size": 0,
  "max_batch_size": 8,
  "model_whitelist": ["qwen2.5:32b", "qwen2.5:14b"],
  "tokens_per_second": 42.0,
  "power_class": "grid",
  "power_headroom": 1.0,
  "health": "healthy",
  "execution_mode": "ollama_direct",
  "endpoint": null,
  "agent_ws_url": null,
  "ollama_url": "http://127.0.0.1:11434",
  "assigned_job_ids": [],
  "last_heartbeat_at": "2026-10-21T15:54:02.113Z",
  "quarantined_until": null,
  "metadata": {}
}
```

Notes:

* `role` is `prefill`, `decode` or `general`; the two `is_*_optimized` flags
  describe the hardware and may disagree with the role.
* `power_class` is `grid`, `solar_trailer` or `battery`. `power_headroom`
  (0.0–1.0) gates Flex work: a trailer node below
  `window.power_headroom_threshold` does not take sleep-time work.
* `health` is `healthy`, `degraded`, `draining`, `suspect`, `lost` or
  `offline`. `draining` is a polite announcement that the node is about to
  lose power; the broker migrates the work immediately.
* `available_vram_bytes` above `total_vram_bytes` (and a batch size above the
  maximum) is clamped on validation rather than rejected — a node reporting
  bad numbers should not take the fleet down.

## `FaultRecoverySnapshot`

`horizon_broker/contracts/recovery.py` · `schemas/fault_recovery_snapshot.v1.json`

The payload committed to EdgeRecord when a job is interrupted.

```json
{
  "snapshot_id": "b0a5...-...",
  "job_id": "0f2c6b5e-8f36-4a9f-9a1d-6b0f0d9c1f21",
  "node_id": "pi-grid-01",
  "attempt": 0,
  "model": "qwen2.5:7b",
  "step": 64,
  "partial_tokens": [
    {"index": 0, "token_id": 3112, "text": "The", "logprob": -0.31}
  ],
  "resume_point": {
    "decode_step": 64,
    "generated_token_count": 64,
    "kv_cache_ref": "edge-doc-8812",
    "kv_cache_state_id": "edge-doc-8812",
    "prompt_hash": "9f2b...c41d",
    "model": "qwen2.5:7b",
    "checksum_algorithm": "sha256"
  },
  "created_at": "2026-10-21T15:58:41.902Z",
  "ledger_document_id": "b0a5...-...",
  "content_hash": "afbc3053d88631df385c778c...",
  "schema_version": "1.0"
}
```

* `step` is the decode iteration reached when the checkpoint was taken.
* `partial_tokens` are the tokens already emitted, in order.
* `resume_point.kv_cache_ref` is the same identifier surfaced as
  `InferenceJob.kv_cache_state_id`.
* `resume_point.prompt_hash` is the sha256 of the canonical prompt. A
  snapshot whose prompt hash does not match the job is discarded: the
  recovered output would be incoherent.
* `content_hash` covers the canonical payload and is verified on read; a
  record that fails verification is ignored.

## `LedgerRecord` / `LedgerCommitReceipt`

`horizon_broker/contracts/ledger.py` · `schemas/ledger_record.v1.json`

The envelope the broker writes to EdgeRecord. `LedgerRecord.payload` holds the
snapshot body; `content_hash` guards the payload; `LedgerCommitReceipt` is the
proof of a durable commit.

## The node control channel

Frames exchanged over the node WebSocket:

| Frame | Direction | Payload |
| --- | --- | --- |
| `ping` | broker → node | liveness probe |
| `heartbeat` | node → broker | capability snapshot, power headroom |
| `assign` | broker → node | the job envelope, `phase` (`prefill`, `decode`, `monolithic`), `resume_point`, `kv_cache_state_id` |
| `progress` | node → broker | `job_id`, `tokens_generated`, optional text |
| `checkpoint` | node → broker | `job_id`, `step`, `partial_tokens`, `kv_cache_ref` |
| `complete` | node → broker | `job_id`, `text`, and for a prefill phase `kv_cache_ref` |
| `cancel` | broker → node | `job_id` |
| `draining` | node → broker | the node is about to lose power |

A frame without a `job_id` is attributed to the node's only active job; when
the node runs more than one job the frame is ignored and logged, because
guessing would corrupt the wrong job.

## The EdgeRecord HTTP port

The broker does not depend on EdgeRecord internals; it depends on this narrow
contract, whose paths are configurable:

```text
POST {base_url}{ledger.commit_path}
     body:    {"document_id", "payload", "content_hash", "tags"}
     reply:   {"document_id", "commit_url", "content_hash"}

GET  {base_url}{ledger.query_path}?job_id=...&limit=1
     reply:   {"records": [{"document_id", "payload", "content_hash"}]}
```

Defaults: `commit_path=/collections/{collection}/records`,
`query_path=/collections/{collection}/records`,
`collection=horizon_broker.snapshots`.

Any transport failure raises `LedgerUnavailableError`, which the snapshot
store answers by reading the local journal instead.

## Versioning

Every envelope carries `schema_version`. Within a major version, fields may
be added but never removed or re-typed; a consumer must ignore unknown
fields. The JSON Schema files are named `<contract>.v1.json`.
