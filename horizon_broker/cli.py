"""Operator command line client.

    horizon-broker job submit --model qwen2.5:7b --prompt "..." --tier flex
    horizon-broker job status <job-id>
    horizon-broker job list
    horizon-broker nodes
    horizon-broker queue

The client talks to a running broker over its HTTP API. It never touches the
broker's state directly, so it can be run from another machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import aiohttp

DEFAULT_BASE_URL = "http://127.0.0.1:8791"


def _base_url(args: argparse.Namespace) -> str:
    return str(args.base_url).rstrip("/")


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
                body = json.loads(text) if text else None
            except json.JSONDecodeError:
                body = text
            return response.status, body


def _print(body: Any) -> None:
    if isinstance(body, (dict, list)):
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        print(body)


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
    """Show the queue occupancy."""
    status, response = await _request(args, "GET", "/v1/queue")
    if status != 200:
        print(f"the broker did not answer (HTTP {status})", file=sys.stderr)
        return 1
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
        batch = f"{node.get('current_batch_size', 0)}/{node.get('max_batch_size', 0)}"
        power = f"{float(node.get('power_headroom', 0)):.2f}"
        models = ",".join(node.get("model_whitelist") or []) or "any"
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
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
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

    job_list = job_sub.add_parser("list", help="show the queue occupancy")
    job_list.set_defaults(handler=cmd_job_list)

    nodes = subparsers.add_parser("nodes", help="show the fleet")
    nodes.set_defaults(handler=cmd_nodes)

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
