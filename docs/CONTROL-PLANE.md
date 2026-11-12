# The control plane: one broker owns a fleet

Phase 2 of the plan: two brokers on one network must not both accept a
job id and run it twice. This document is the design. The lease is built
and tested (`horizon_broker/control/lease.py`); what follows is how the
rest of the phase fits around it.

## The problem, precisely

The broker assumes it is the only writer for a given job id. Job ids are
UUIDs, so a collision is not the risk: **the risk is two brokers admitting
the same submission**, because each sees the id as new. The symptom is
not a crash but a silent duplicate execution - the fleet does the work
twice and the operator sees nothing.

Today that is prevented by deployment discipline (one broker per fleet).
Phase 2 makes it a property of the software.

## The decision: a lease over a shared file

One file, `fleet.lease`, in a directory both brokers can reach. The
broker that holds the advisory lock on it owns the fleet.

Why this and not a consensus protocol:

- **The operating system releases the lock when the process dies.** A
  broker that crashes cannot strand the fleet; a follower takes over as
  soon as the kernel says the lock is free. There is no lease expiry to
  tune, no clock to trust, and no split-brain window from a stale timer.
- **The lock is held by a file descriptor**, so a broker that stalls
  keeps the fleet. The owner is not lost by a slow heartbeat.
- **The owner is written into the file**, so an operator can see who owns
  the fleet without asking a running process.

The constraint is honest and must be stated: the lease must live on a
filesystem with real `flock` semantics. A local disk is fine; a network
filesystem that only approximates `flock` is not, because two brokers
could then both believe they own the fleet. A shared disk with real
locks (or one broker's host exporting a directory over a lock-correct
mount) is the deployment shape this design assumes.

## The protocol

```
startup     acquire the lease (never blocking)
            ├─ acquired   → this broker owns the fleet; it admits work
            └─ held by X  → this broker is a follower; it admits nothing
renew       every few seconds: re-assert the lock, refresh the owner note
shutdown    release the lease, so the follower takes over immediately
crash       the kernel releases the lock; nothing to clean up
```

A follower is not idle: it keeps its journal, its node registry and its
probe running (so the fleet view is right everywhere), but it refuses
admissions with `503 broker is a follower` until it owns the lease. This
is implemented: `broker.lease_path` points a broker at the shared file,
and the refusal names the owner.

## What a follower does with a request

Three options, in order of preference:

1. **Refuse** (`503`, with the owner's id). The caller retries and the
   owner answers. Simple, and correct for every caller that can retry.
2. **Proxy** the request to the owner. Needs the owner's address and a
   second auth path; deferred until the fleet actually runs two brokers.
3. **Queue it into the shared journal** and let the owner pick it up.
   This is the interesting one, and it is what the journal replay already
   half provides.

## The queue out of process (roadmap 2.2)

The journal is already a durable, ordered log of submissions and terminal
states, and a cold broker already replays it. The step from there to a
shared queue is small:

- a submission is appended to the journal by the broker that accepted it;
- the owner claims work from the journal and records the claim;
- a job is claimed by writing the owner's tag beside it, so a crash
  leaves a stale claim that the next owner can take over.

The claim needs the same lease discipline: a claim is a short-lived lease
on the job's record, held by the broker that is running it. The journal's
`flock` (already shipped) covers the file; the per-job claim needs its
own primitive, and it is the same one.

## Job ids and ownership (roadmap 2.3)

The roadmap wanted namespaced job ids. That is no longer needed: the
owner is already recorded twice - as the `broker:<owner_id>` tag on every
committed recovery record (shipped) and in the ledger query scope
(shipped). A job id stays a bare UUID, which keeps the wire format
unchanged and the contract's `schema_version` where it is.

## The drill

The phase gate is a test and a live drill:

- **Test:** two brokers, one lease directory, one submission. Exactly one
  admits it; the other refuses with `503`. The owner dies; the follower
  takes the lease and admits the next submission. A job submitted before
  the crash is queued and runs after it.
- **Live drill:** two brokers on the same host against the same fleet,
  a real submission, and the operator watching one answer while the other
  refuses.

## The order of the work

| Step | What | Gate |
| --- | --- | --- |
| 2.1a | the lease (built ✓) and the follower refusal | two-broker test |
| 2.1b | the follower's refusal (**done**) - a follower answers admissions with `503 broker is a follower` and names the owner; `/readyz` reports `follower` | operator can see who owns the fleet |
| 2.2 | the journal as the shared queue, with per-job claims | a submission before a crash runs after it |
| 2.3 | ~~namespaced ids~~ - already covered by the owner tag | - |
| 2.4 | ~~journal locking~~ - shipped | - |
| 2.5 | ~~ledger scoping~~ - shipped | - |

Until 2.2 lands, one broker owns a fleet and the others refuse work: the
duplicate execution is gone, at the cost of a follower that cannot help.
