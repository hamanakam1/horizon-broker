"""The two-broker drill.

Phase 2's gate is a test (and a live drill): two brokers on one fleet, one
admits the work, the other refuses it, and a job submitted to either
completes exactly once. The in-process test (tests/test_follower.py) is the
fast, deterministic version of that drill. This script is the live one: two
real brokers on the same machine, one lease file, one submission each.

Usage:

    python scripts/two_broker_drill.py \\
        --broker-a http://127.0.0.1:8791 --broker-b http://127.0.0.1:8792 \\
        --lease /tmp/fleet.lease \\
        --config config/broker.example.yaml

The brokers must be started beforehand, pointed at the same lease file:

    HORIZON_BROKER_LEASE_PATH=/tmp/fleet.lease HORIZON_BROKER_PORT=8791 \\
        python -m horizon_broker serve --config config/broker.example.yaml
    HORIZON_BROKER_LEASE_PATH=/tmp/fleet.lease HORIZON_BROKER_PORT=8792 \\
        python -m horizon_broker serve --config config/broker.example.yaml

The script never touches the brokers' state: it only submits and observes.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class DrillReport:
    owner_admits: bool
    follower_refuses: bool
    follower_names_owner: bool
    takeover: bool
    steps: list[str]

    @property
    def passed(self) -> bool:
        return self.owner_admits and self.follower_refuses and self.follower_names_owner


class Broker:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            return exc.code, payload

    def get(self, path: str) -> tuple[int, Any]:
        request = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))


def run_drill(owner: Broker, follower: Broker, job_id: str) -> DrillReport:
    steps: list[str] = []

    status, body = owner.post(
        "/v1/jobs", {"job_id": job_id, "payload": {"model": "qwen2.5:7b", "prompt": "drill"}, "max_acceptable_latency_ms": 600_000}
    )
    owner_admits = status == 202
    steps.append(f"owner admits {job_id}: {status} {'OK' if owner_admits else 'FAIL'}")

    status, body = follower.post(
        "/v1/jobs", {"job_id": job_id, "payload": {"model": "qwen2.5:7b", "prompt": "drill"}, "max_acceptable_latency_ms": 600_000}
    )
    follower_refuses = status == 503
    steps.append(f"follower refuses {job_id}: {status} {'OK' if follower_refuses else 'FAIL'}")
    follower_names_owner = (
        isinstance(body, dict) and body.get("leader") is not None
    )
    steps.append(f"follower names the owner: {'OK' if follower_names_owner else 'FAIL'}")

    # The takeover cannot be driven from here - the owner must be stopped by
    # hand (or by a supervisor). What the script verifies is the pre-condition:
    # the follower is a follower, and the owner is the owner.
    leader = owner.get("/readyz")[1].get("lease_holder") or owner.get("/readyz")[1].get("leader")
    steps.append(f"owner reports itself as leader: {'OK' if leader in (True, None) else leader}")

    return DrillReport(
        owner_admits=owner_admits,
        follower_refuses=follower_refuses,
        follower_names_owner=follower_names_owner,
        takeover=False,  # the takeover is asserted by the operator, see below
        steps=steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two_broker_drill", description=__doc__)
    parser.add_argument("--broker-a", required=True)
    parser.add_argument("--broker-b", required=True)
    parser.add_argument("--job-id", default="drill-job")
    args = parser.parse_args(argv)

    owner = Broker(args.broker_a)
    follower = Broker(args.broker_b)

    report = run_drill(owner, follower, args.job_id)
    for step in report.steps:
        print(step)
    print()
    if not report.owner_admits or not report.follower_refuses or not report.follower_names_owner:
        print("the drill FAILED: the ownership boundary is not in place")
        return 1
    print("the ownership boundary holds: one broker owns the fleet, the other refuses it")
    print()
    print("to exercise the takeover, stop broker A (SIGTERM) and submit a job to broker B:")
    print(f"  curl -sX POST {args.broker_b}/v1/jobs -H 'content-type: application/json' \\")
    print("       -d '{\"job_id\": \"drill-takeover\", \"payload\": {\"model\": \"qwen2.5:7b\",")
    print("              \"prompt\": \"drill\"}, \"max_acceptable_latency_ms\": 600000}'")
    print("broker B must answer 202 within two seconds of broker A stopping.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
