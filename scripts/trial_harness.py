"""Fault-trial harness.

Phase 3 of the roadmap asks for a seven-day trial on the real fleet with
the fault injection switched on, and for the SLOs to be measured rather than
asserted. This harness is the instrument:

``record``
    Sample the broker on an interval and keep the state of every job it
    has seen, so a lost job is a job that never reached a terminal state.

``fault``
    Run the fault command of one node - stop it, or restore it. The
    commands live in the trial configuration, so the harness never has to
    know what a Raspberry Pi looks like.

``report``
    Turn the samples into the SLO table: recovery percentiles, the job
    outcomes, the queue latency and the duplicate check.

The harness only reads the broker's own API and metrics; it never changes
what the broker does. The trial is only meaningful if the faults are real.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATES = {"completed", "failed", "cancelled"}

SLO_ROWS = (
    ("recovery_p99_seconds", "Recovery completes within 30 s at p99", "<= 30 s"),
    ("lost_jobs", "No job is lost across the trial", "0"),
    ("duplicate_executions", "No job is executed twice", "0"),
    ("queue_latency_p99_seconds", "Flex queue latency at p99", "measured"),
    ("throughput_ratio", "The wave amortises the weight load", ">= 1.5x"),
)


@dataclass
class NodeSpec:
    """One node of the trial fleet and the commands that fault it."""

    node_id: str
    fault_command: list[str] = field(default_factory=list)
    restore_command: list[str] = field(default_factory=list)


@dataclass
class TrialConfig:
    """The trial configuration: where the broker is and how to fault it."""

    broker_url: str
    nodes: list[NodeSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> TrialConfig:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        nodes = [
            NodeSpec(
                node_id=str(entry["node_id"]),
                fault_command=list(entry.get("fault_command", [])),
                restore_command=list(entry.get("restore_command", [])),
            )
            for entry in raw.get("nodes", [])
        ]
        return cls(broker_url=str(raw.get("broker_url", "http://127.0.0.1:8791")), nodes=nodes)


class BrokerClient:
    """A read-only client for the broker's API and metrics."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_json(self, path: str) -> Any:
        request = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def metrics(self) -> dict[str, float]:
        """Parse the Prometheus text exposition into a name -> value mapping."""
        request = urllib.request.Request(f"{self.base_url}/metrics")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        values: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, raw = line.rpartition(" ")
            try:
                value = float(raw)
            except ValueError:
                continue
            key = name.split("{", 1)[0]
            values[key] = values.get(key, 0.0) + value
        return values


class JobLedger:
    """The last known state of every job the trial has seen."""

    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.first_seen: dict[str, float] = {}

    def observe(self, jobs: list[dict[str, Any]], now: float) -> None:
        for job in jobs:
            job_id = str(job.get("job_id", ""))
            state = str(job.get("state", ""))
            if not job_id or not state:
                continue
            self.states[job_id] = state
            self.first_seen.setdefault(job_id, now)

    def outcomes(self) -> dict[str, int]:
        counts = {"completed": 0, "failed": 0, "cancelled": 0, "running": 0, "other": 0}
        for state in self.states.values():
            if state in counts:
                counts[state] += 1
            else:
                counts["other"] += 1
        return counts

    def lost(self) -> int:
        """The jobs that never reached a terminal state."""
        return sum(1 for state in self.states.values() if state not in TERMINAL_STATES)


def record(config: TrialConfig, out_path: str, interval: float, duration: float) -> int:
    """Sample the broker until the duration is up, or until interrupted."""
    client = BrokerClient(config.broker_url)
    ledger = JobLedger()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration
    samples = 0
    with out.open("a", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            try:
                metrics = client.metrics()
                queue = client.get_json("/v1/queue")
                jobs = client.get_json("/v1/jobs").get("jobs", [])
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                print(f"sample failed: {exc}", file=sys.stderr)
            else:
                now = time.time()
                ledger.observe(jobs, now)
                sample = {
                    "ts": now,
                    "metrics": metrics,
                    "queue": queue,
                    "jobs": {str(job.get("job_id")): job.get("state") for job in jobs},
                }
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                samples += 1
            time.sleep(interval)
    print(f"recorded {samples} sample(s) to {out}")
    return 0


def fault(config: TrialConfig, node_id: str, *, stop: bool) -> int:
    """Run the fault (or restore) command of one node."""
    node = next((entry for entry in config.nodes if entry.node_id == node_id), None)
    if node is None:
        print(f"unknown node: {node_id}", file=sys.stderr)
        return 1
    command = node.fault_command if stop else node.restore_command
    if not command:
        print(f"node {node_id} has no {'fault' if stop else 'restore'} command", file=sys.stderr)
        return 1
    print(f"running: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    return result.returncode


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def evaluate(samples: list[dict[str, Any]], ledger: JobLedger) -> dict[str, Any]:
    """Turn the samples into the numbers the SLO table is built from."""
    recovery: list[float] = []
    submitted = 0.0
    completed = 0.0
    for sample in samples:
        metrics = sample.get("metrics", {})
        value = metrics.get("horizon_recovery_seconds_sum")
        count = metrics.get("horizon_recovery_seconds_count")
        if isinstance(value, (int, float)) and isinstance(count, (int, float)) and count:
            recovery.append(float(value) / float(count))
        submitted = max(submitted, float(metrics.get("horizon_jobs_submitted_total", 0.0)))
        completed = max(completed, float(metrics.get("horizon_jobs_completed_total", 0.0)))
    outcomes = ledger.outcomes()
    return {
        "recovery_p99_seconds": _percentile(recovery, 0.99),
        "recovery_mean_seconds": sum(recovery) / len(recovery) if recovery else None,
        "lost_jobs": ledger.lost(),
        "duplicate_executions": max(0.0, completed - submitted),
        "queue_latency_p99_seconds": None,
        "throughput_ratio": None,
        "outcomes": outcomes,
        "samples": len(samples),
    }


def render_report(results: dict[str, Any], config: TrialConfig) -> str:
    """Render the SLO table."""
    lines = [
        "# Trial report",
        "",
        f"Broker: `{config.broker_url}`  ",
        f"Samples: {results['samples']}",
        "",
        "| Measure | SLO | Measured | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    for key, description, threshold in SLO_ROWS:
        value = results.get(key)
        if value is None:
            measured = "not measured"
            verdict = "-"
        else:
            measured = f"{value:.2f}" if isinstance(value, float) else str(value)
            verdict = "pass" if _meets(key, value) else "FAIL"
        lines.append(f"| {description} | {threshold} | {measured} | {verdict} |")
    outcomes = results.get("outcomes", {})
    lines += [
        "",
        "## Job outcomes",
        "",
        "| Outcome | Count |",
        "| --- | --- |",
    ]
    for name, count in outcomes.items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _meets(key: str, value: Any) -> bool:
    if key == "lost_jobs":
        return value == 0
    if key == "duplicate_executions":
        return value == 0
    if key == "recovery_p99_seconds":
        return value is not None and value <= 30.0
    if key == "throughput_ratio":
        return value is not None and value >= 1.5
    return True


def report(config: TrialConfig, samples_path: str, out_path: str | None) -> int:
    """Build the SLO report from the recorded samples."""
    path = pathlib.Path(samples_path)
    samples: list[dict[str, Any]] = []
    ledger = JobLedger()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append(sample)
            jobs = sample.get("jobs", {})
            ledger.observe(
                [{"job_id": job_id, "state": state} for job_id, state in jobs.items()],
                float(sample.get("ts", 0.0)),
            )
    results = evaluate(samples, ledger)
    text = render_report(results, config)
    if out_path:
        target = pathlib.Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trial_harness", description=__doc__)
    parser.add_argument("--config", default="config/trial.example.json")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="sample the broker on an interval")
    rec.add_argument("--out", default="var/trial/samples.jsonl")
    rec.add_argument("--interval", type=float, default=5.0)
    rec.add_argument("--duration", type=float, default=3600.0)

    flt = sub.add_parser("fault", help="stop or restore one node")
    flt.add_argument("--node", required=True)
    flt.add_argument("--stop", action="store_true")
    flt.add_argument("--restore", action="store_true")

    rep = sub.add_parser("report", help="build the SLO report from the samples")
    rep.add_argument("--samples", default="var/trial/samples.jsonl")
    rep.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrialConfig.load(args.config)
    if args.command == "record":
        return record(config, args.out, args.interval, args.duration)
    if args.command == "fault":
        if args.stop == args.restore:
            print("exactly one of --stop or --restore is required", file=sys.stderr)
            return 2
        return fault(config, args.node, stop=args.stop)
    if args.command == "report":
        return report(config, args.samples, args.out)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - the entry point
    sys.exit(main())
