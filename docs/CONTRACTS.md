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
`assigned_node_id` identifies the node that currently owns dispatched work and
is cleared whenever work returns to a queue or recovery lane. For terminal
jobs, the value is retained as execution provenance; it is not an active
reservation. `phase` describes active execution and returns to `monolithic`
when a job is re-queued or restored after a broker restart.

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
* `last_heartbeat_at` and `quarantined_until`, when present, must be
  timezone-aware because watchdog and placement compare them with UTC now.
* `endpoint` and `ollama_url` must be absolute HTTP(S) URLs, while
  `agent_ws_url` must be an absolute WS(S) URL. All three reject embedded
  credentials and ports outside `1–65535` before the node can enter routing.
* `available_vram_bytes` above `total_vram_bytes` (and a batch size above the
  maximum) is clamped on validation rather than rejected — a node reporting
  bad numbers should not take the fleet down.
* `current_batch_size` is the node's total reported occupancy, including work
  not owned by this broker. `assigned_job_ids` is only this broker's ownership
  ledger; releasing one of those jobs must not erase the reported occupancy
  baseline.
* `tokens_per_second` and `power_headroom`, when present, must be finite
  numbers; non-finite telemetry is rejected before routing.
* `metadata` must contain only JSON-safe values so capability diagnostics and
  wire responses cannot fail on arbitrary objects or non-standard numbers.

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
* `step`, `resume_point.decode_step`, and
  `resume_point.generated_token_count` must be identical; the broker rejects
  snapshots with ambiguous progress coordinates.
* `partial_tokens` may be truncated for storage, but their count cannot exceed
  the recorded decode step.
* `partial_tokens` are the tokens already emitted, in order.
* `resume_point.kv_cache_ref` is the same identifier surfaced as
  `InferenceJob.kv_cache_state_id`.
* `resume_point.prompt_hash` is the sha256 of the canonical prompt. A
  snapshot whose prompt hash does not match the job is discarded: the
  recovered output would be incoherent.
* `content_hash` covers the canonical payload and is verified on read; a
  record that fails verification is ignored.
* `created_at` must be timezone-aware, so recovery timestamps remain
  comparable across machines.

## `LedgerRecord` / `LedgerCommitReceipt`

`horizon_broker/contracts/ledger.py` · `schemas/ledger_record.v1.json`

The envelope the broker writes to EdgeRecord. `LedgerRecord.payload` holds the
snapshot body; `content_hash` guards the payload; `LedgerCommitReceipt` is the
proof of a durable commit. `LedgerRecord.created_at` and
`LedgerCommitReceipt.committed_at` must be timezone-aware, matching the
recovery snapshot timestamp contract.

## The OpenAI-compatible surface

`POST /v1/chat/completions` accepts an OpenAI chat-completions body:

```json
{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 128,
  "temperature": 0.2,
  "top_p": 0.95,
  "stop": ["END"],
  "seed": 7,
  "stream": false
}
```

and answers with the OpenAI completion object, or with server-sent events
when `stream` is true:

```text
data: {"id": "chatcmpl-...","object": "chat.completion.chunk", ...}
data: {"id": "chatcmpl-...","object": "chat.completion.chunk", "choices": [{"delta": {"content": "..."}}]}
data: {"id": "chatcmpl-...","object": "chat.completion.chunk", "choices": [{"delta": {}, "finish_reason": "stop"}]}
data: [DONE]
```

Two extension fields steer the broker and are ignored by other clients:

| Field | Meaning |
| --- | --- |
| `horizon_priority_tier` | `interactive` or `flex` |
| `horizon_max_acceptable_latency_ms` | The latency budget; above 120 s the request becomes Flex work |
| `horizon_timeout_seconds` | How long the caller waits for the answer |

The `system` message is preserved as a separate field, and a payload with
chat messages is sent to the engine's chat endpoint so the model's own chat
template applies. Flattening a conversation by hand would silently change
the prompt.

The `usage` block reports the token counts the engine actually measured
(`prompt_eval_count` / `eval_count`), not an estimate; the estimate is only
used when the engine does not report counts.

## The node control channel

Frames exchanged over the node WebSocket:

| Frame | Direction | Payload |
| --- | --- | --- |
| `ping` | broker → node | liveness probe |
| `heartbeat` | node → broker | capability snapshot, power headroom |
| `assign` | broker → node | the job envelope, `phase` (`prefill`, `decode`, `monolithic`), `resume_point`, `kv_cache_state_id` |
| `batch_assign` | broker → node | `jobs`: one or more compatible `assign` payloads; only nodes with `metadata.supports_batch=true` may receive it |
| `progress` | node → broker | `job_id`, `tokens_generated`, optional text |
| `checkpoint` | node → broker | `job_id`, `tokens_generated`, `partial_tokens`, `kv_cache_ref` |
| `complete` | node → broker | `job_id`, `text`, and for a prefill phase `kv_cache_ref` |
| `cancel` | broker → node | `job_id` |
| `draining` | node → broker | the node is about to lose power |

A frame without a `job_id` is attributed to the node's only active job; when
the node runs more than one job the frame is ignored and logged, because
guessing would corrupt the wrong job.

`batch_assign` is an admission optimization, not a new recovery identity:
each item retains its own `job_id`, phase, resume point and terminal lifecycle.
The broker sends one batch only when all members target the same node, use the
same model, have matching routing decisions, and fit the node's remaining
batch capacity. Direct Ollama nodes continue to receive individual requests.

## The EdgeRecord HTTP port

The broker does not depend on EdgeRecord internals; it depends on this narrow
contract, whose paths are configurable:

```text
POST {base_url}{ledger.commit_path}
     body:    {"document_id", "payload", "content_hash", "tags"}
     reply:   {"document_id", "commit_url", "content_hash"}

GET  {base_url}{ledger.query_path}?job_id=...&owner_id=...&limit=1
     reply:   {"records": [{"document_id", "payload", "content_hash", "tags"}]}
```

Defaults: `commit_path=/collections/{collection}/records`,
`query_path=/collections/{collection}/records`,
`collection=horizon_broker.snapshots`, and `owner_id=horizon-broker`.
Both paths must be absolute path templates with exactly one `{collection}`
placeholder and no query or fragment component; the collection value is
percent-encoded when the request URL is built. This keeps route interpolation
deterministic and prevents collection delimiters from changing the path.
Committed recovery records also carry the `broker:<owner_id>` tag, so a shared
EdgeRecord collection can enforce the same ownership boundary on writes and
reads. A returned record must carry the configured owner tag; a missing or
conflicting owner is rejected as unavailable recovery state.

Transient failures (network errors and 5xx) are retried with a growing
backoff (`ledger.max_retries`, `ledger.retry_backoff_seconds`); a 4xx is
permanent and fails immediately, because retrying a rejected document only
delays the error. When `EDGERECORD_TOKEN` (configurable through
`ledger.auth_token_env`) is set, it is sent as
`Authorization: Bearer <token>`; the value is read per request so it can be
rotated without restarting the broker. Every request applies
`ledger.timeout_seconds` independently, including when the broker injects its
shared HTTP session, and redirects are rejected rather than followed.

The adapter treats a successful response as untrusted input. Commit replies
must be valid standards-compliant JSON objects with a non-empty `document_id`
matching the submitted record; an optional `content_hash` must also match
when present. An optional `commit_url` must be an absolute HTTP(S) URL without
embedded credentials or surrounding whitespace. Query replies may use an
object containing a `records`, `items`, `results`, or `documents` list, a
direct record object, or a direct list of records. The selected record must
have a non-empty `document_id`, a mapping `payload`, and `payload.job_id` equal
to the requested job. These checks prevent a malformed or cross-job ledger
response from being used as recovery state. When a query record includes
`content_hash`, it must also match the canonical hash of the returned payload;
the broker ultimately requires a valid content hash before using the record
for recovery.

Any transport failure raises `LedgerUnavailableError`, which the snapshot
store answers by reading the local journal instead.

## Versioning

Every envelope carries `schema_version`. Within a major version, fields may
be added but never removed or re-typed. A consumer must reject an envelope
that carries a field it does not know: every contract sets `extra="forbid"`,
so an unknown field is an error rather than something to ignore. Adding a
field therefore means a new `schema_version` and a new file in `schemas/`.
