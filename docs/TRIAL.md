# The fault trial

Phase 3 of the plan asks for a seven-day trial on the real fleet with the
fault injection switched on, and for the SLOs to be measured rather than
asserted. This document is the trial: the fleet, the fault schedule, the
SLOs, and how each one is measured.

The point of the trial is not that the broker survives the faults. It is
that the broker **says** what happened, in a number an operator can alert
on, and that the numbers come from the broker's own metrics.

## The fleet

| Node | Role | Execution mode | How it is faulted |
| --- | --- | --- | --- |
| `m3-max` | prefill | `ollama_direct` | the Ollama server is stopped: the socket closes |
| `pi-01` | decode | `ollama_direct` | the Ollama server is stopped: the socket closes |
| `trailer-01` | decode, solar | `agent_container` | the machine is powered off: silence |

The commands live in `config/trial.example.json`; the harness never has to
know what a Raspberry Pi looks like. Copy the file, point the commands at
the real fleet, and keep it out of the repository if it holds credentials.

## The harness

```sh
# sample the broker while the trial runs (in its own terminal, or under tmux)
python scripts/trial_harness.py record --config config/trial.example.json \
    --out var/trial/samples.jsonl --interval 5

# fault a node, and bring it back
python scripts/trial_harness.py fault --config config/trial.example.json --node pi-01 --stop
python scripts/trial_harness.py fault --config config/trial.example.json --node pi-01 --restore

# build the SLO table from the samples
python scripts/trial_harness.py report --config config/trial.example.json \
    --samples var/trial/samples.jsonl --out docs/trial-report.md
```

The recorder keeps the last known state of every job the broker has seen, so
a **lost job** is a job that never reached a terminal state - not a job that
is missing from a listing.

## The schedule

Seven days, one fault class per day, each one repeated often enough to
matter and spaced far enough apart to be survivable.

| Day | Fault | What it exercises |
| --- | --- | --- |
| 1 | baseline: no faults | the fleet at rest; the wave's throughput baseline |
| 2 | stop the Ollama server on `m3-max`, every 30 min | the socket-closes path; the job must resume elsewhere |
| 3 | stop the Ollama server on `pi-01`, every 30 min | the same, on the decode node of a split |
| 4 | power off `trailer-01` for 10 min, four times | a node that dies without a goodbye; the recovery budget |
| 5 | pull the network on `pi-01` (interface down) | silence without a close: the watchdog's verdict, not a socket error |
| 6 | a hot day: all three faults in one shift | overlap: a job must not land on a node that just failed |
| 7 | no faults; drain and restart the broker | the cold start: the queue replays from the journal |

Day 7 is the one that needs no fault at all and is the easiest to forget:
a restart must not lose queued work.

## The SLOs

| Measure | SLO | Measured from |
| --- | --- | --- |
| Recovery completes within 30 s at p99 | `<= 30 s` | `horizon_recovery_seconds` (the summary's p99) |
| No job is lost across the trial | `0` | the recorder's job ledger: a job that never reached a terminal state |
| No job is executed twice | `0` | `horizon_jobs_completed_total` against `horizon_jobs_submitted_total` |
| Flex queue latency at p99 | measured | the recorder's queue samples |
| The wave amortises the weight load | `>= 1.5x` | the same four prompts, one after another and as a wave |
| The recovery budget is respected | no recovery over 30 s | `horizon_recovery_seconds` |

`scripts/trial_harness.py report` renders the first four; the last two are
measured by hand and written into the report.

### What a failure means

A failed SLO is a finding, not an embarrassment. Record it with the
metrics window around it: the failing number, the node that died, the
metric that moved, and the job that shows it. A trial that finds one
honest defect in seven days has paid for itself.

## The evidence

Attach to the release notes:

1. `docs/trial-report.md` - the generated SLO table.
2. The samples file (`var/trial/samples.jsonl`, trimmed) - the raw record.
3. The broker's own metrics for the trial window (`/metrics`).
4. The MFU measurement: the same prompts, one after another and as a wave.
5. The timeline: which fault, when, and what the operator saw.

## The rollback

The trial changes nothing about how the broker runs, so the rollback is
small: stop the trial, and `git revert` the trial commits. The broker's
configuration is unchanged by the trial; only `config/trial.example.json`
and the harness are new, and neither is read by the broker.

## Before the trial starts

- The broker runs the release under test (`v0.3.0` or later), not a branch.
- The fleet serves real work, not synthetic prompts: the MFU number means
  nothing on a toy workload.
- The recorder runs on a machine that survives a powered-off node - the
  laptop running the broker, not the Pi.
- The fault commands are tested once by hand each, so a trial day is not
  lost to a typo in a shell command.
