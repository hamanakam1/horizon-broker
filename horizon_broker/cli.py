"""Operator command line client.

    horizon-broker-cli job submit --model qwen2.5:7b --prompt "..." --tier flex
    horizon-broker-cli job status <job-id>
    horizon-broker-cli job list
    horizon-broker-cli queue
    horizon-broker-cli nodes
    horizon-broker-cli health

The client talks to a running broker over its HTTP API. It never touches the
broker's state directly, so it can be run from another machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, NoReturn

import aiohttp

DEFAULT_BASE_URL = "http://127.0.0.1:8791"


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _base_url(args: argparse.Namespace) -> str:
    return str(args.base_url).rstrip("/")


def _positive_timeout(raw: str) -> float:
    """Parse a finite, positive operator request timeout."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return value


async def _request(
    args: argparse.Namespace,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    url = f"{_base_url(args)}{path}"
    timeout = aiohttp.ClientTimeout(total=float(args.timeout))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, json=payload) as response:
            text = await response.text()
            try:
                body = (
                    json.loads(text, parse_constant=_reject_json_constant) if text else None
                )
            except (json.JSONDecodeError, ValueError):
                body = text
            return response.status, body


def _print(body: Any) -> None:
    if isinstance(body, (dict, list)):
        print(json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(body)


def _display_power(value: Any) -> str:
    """Render untrusted node telemetry without crashing the operator CLI."""
    if isinstance(value, bool):
        return "-"
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return "-"
    return f"{numeric:.2f}" if math.isfinite(numeric) else "-"


def _display_models(value: Any) -> str:
    """Render an untrusted model whitelist without assuming its shape."""
    if not isinstance(value, list):
        return "any"
    models: list[str] = []
    for model in value:
        if (
            isinstance(model, str)
            and model
            and model == model.strip()
            and model not in models
        ):
            models.append(model)
    return ",".join(models) or "any"


def _is_positive_finite_number(value: Any) -> bool:
    """Recognize numeric counters without boolean or overflow coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, ValueError):
        return False


async def cmd_job_submit(args: argparse.Namespace) -> int:
    """Submit a job."""
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
    }
    if args.output_tokens:
        payload["expected_output_tokens"] = args.output_tokens
    body: dict[str, Any] = {
        "payload": payload,
        "max_acceptable_latency_ms": args.latency_ms,
    }
    if args.tier:
        body["priority_tier"] = args.tier
    status, response = await _request(args, "POST", "/v1/jobs", body)
    if status != 202:
        print(f"the broker refused the job (HTTP {status})", file=sys.stderr)
        _print(response)
        return 1
    _print(response)
    return 0


async def cmd_job_status(args: argparse.Namespace) -> int:
    """Show one job."""
    status, response = await _request(args, "GET", f"/v1/jobs/{args.job_id}")
    if status != 200:
        print(f"unknown job {args.job_id} (HTTP {status})", file=sys.stderr)
        return 1
    _print(response)
    return 0


async def cmd_job_cancel(args: argparse.Namespace) -> int:
    """Cancel a job."""
    status, response = await _request(args, "DELETE", f"/v1/jobs/{args.job_id}")
    if status != 200:
        print(f"could not cancel {args.job_id} (HTTP {status})", file=sys.stderr)
        _print(response)
        return 1
    _print(response)
    return 0


async def cmd_job_list(args: argparse.Namespace) -> int:
    """List the tracked jobs, newest first.

    The header carries ``total`` vs ``count`` so an operator notices when a
    response was truncated by the limit instead of assuming the mesh is idle.
    """
    status, response = await _request(args, "GET", "/v1/jobs")
    if status != 200:
        print(f"the broker did not answer (HTTP {status})", file=sys.stderr)
        _print(response)
        return 1
    if not isinstance(response, dict):
        _print(response)
        return 0
    raw_total = response.get("total", 0)
    try:
        total = 0 if isinstance(raw_total, bool) else max(0, int(raw_total))
    except (TypeError, ValueError, OverflowError):
        total = 0
    jobs = response.get("jobs", [])
    if not isinstance(jobs, list):
        jobs = []
    print(f"{len(jobs)} of {total} job(s), newest first")
    print(f"{'JOB':<20} {'STATE':<11} {'TIER':<12} {'MODEL':<16} NODE")
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id", ""))[:20]
        state = str(job.get("state", ""))
        tier = str(job.get("priority_tier", ""))
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        model = str(payload.get("model", "")) if payload else ""
        node = str(job.get("assigned_node_id") or "-")
        print(f"{job_id:<20} {state:<11} {tier:<12} {model:<16} {node}")
        if job.get("error"):
            print(f"job {job_id} carries an error; see `job status {job_id}`")
    return 0


async def cmd_queue(args: argparse.Namespace) -> int:
    """Show the queue occupancy and the in-flight counters."""
    status, response = await _request(args, "GET", "/v1/queue")
    if status != 200:
        print(f"the broker did not answer (HTTP {status})", file=sys.stderr)
        return 1
    if isinstance(response, dict):
        batches = response.get("batches")
        if isinstance(batches, dict):
            rejected = batches.get("rejected", 0)
            print(
                "batch: "
                f"active={batches.get('active_members', 0)} "
                f"admitted={batches.get('admitted', 0)} "
                f"rejected={rejected}"
            )
            if _is_positive_finite_number(rejected):
                print("warning: one or more batch admissions were rejected", file=sys.stderr)
    _print(response)
    return 0


async def cmd_nodes(args: argparse.Namespace) -> int:
    """Show the fleet."""
    status, response = await _request(args, "GET", "/v1/nodes")
    if status != 200:
        print(f"the broker did not answer (HTTP {status})", file=sys.stderr)
        return 1
    nodes = response.get("nodes", []) if isinstance(response, dict) else []
    if not nodes:
        print("no nodes are registered")
        return 0
    print(f"{'NODE':<20} {'ROLE':<9} {'HEALTH':<9} {'BATCH':<7} {'POWER':<7} MODELS")
    for node in nodes:
        if not isinstance(node, dict):
            continue
        batch = f"{node.get('current_batch_size', 0)}/{node.get('max_batch_size', 0)}"
        power = _display_power(node.get("power_headroom", 0))
        models = _display_models(node.get("model_whitelist"))
        print(
            f"{node.get('node_id', ''):<20} {node.get('role', ''):<9} "
            f"{node.get('health', ''):<9} {batch:<7} {power:<7} {models}"
        )
    return 0


async def cmd_health(args: argparse.Namespace) -> int:
    """Show the broker health."""
    status, response = await _request(args, "GET", "/readyz")
    _print(response)
    return 0 if status == 200 else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(prog="horizon-broker-cli", description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="broker base URL")
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=30.0,
        help="request timeout in seconds",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    job = subparsers.add_parser("job", help="work with jobs")
    job_sub = job.add_subparsers(dest="job_command", required=True)

    submit = job_sub.add_parser("submit", help="submit a job")
    submit.add_argument("--model", default="qwen2.5:7b")
    submit.add_argument("--prompt", required=True)
    submit.add_argument("--tier", choices=["interactive", "flex"], default=None)
    submit.add_argument("--latency-ms", type=int, default=600_000)
    submit.add_argument("--output-tokens", type=int, default=None)
    submit.set_defaults(handler=cmd_job_submit)

    job_status = job_sub.add_parser("status", help="show one job")
    job_status.add_argument("job_id")
    job_status.set_defaults(handler=cmd_job_status)

    job_cancel = job_sub.add_parser("cancel", help="cancel a job")
    job_cancel.add_argument("job_id")
    job_cancel.set_defaults(handler=cmd_job_cancel)

    job_list = job_sub.add_parser("list", help="list the tracked jobs, newest first")
    job_list.set_defaults(handler=cmd_job_list)

    nodes = subparsers.add_parser("nodes", help="show the fleet")
    nodes.set_defaults(handler=cmd_nodes)

    queue = subparsers.add_parser("queue", help="show the queue occupancy")
    queue.set_defaults(handler=cmd_queue)

    health = subparsers.add_parser("health", help="show the broker health")
    health.set_defaults(handler=cmd_health)

    return parser


async def run(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the requested command.

    This is the async entry point; :func:`main` wraps it for the shell.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("no command")
        return 2
    try:
        result = await handler(args)
        return int(result)
    except aiohttp.ClientError as exc:
        print(f"could not reach the broker at {_base_url(args)}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    import asyncio

    try:
        return asyncio.run(run(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
