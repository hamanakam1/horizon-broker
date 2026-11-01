"""The operator command line client."""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.cli import (
    _display_models,
    _display_power,
    _is_positive_finite_number,
    build_parser,
    run,
)


async def start_broker() -> tuple[TestServer, str]:
    """Start a fake broker that records what the client sends."""

    received: dict[str, Any] = {}

    async def submit(request: web.Request) -> web.Response:
        received["submit"] = await request.json()
        return web.json_response({"job_id": "job-1", "state": "queued"}, status=202)

    async def job_status(request: web.Request) -> web.Response:
        received["status_id"] = request.match_info["job_id"]
        return web.json_response({"job_id": request.match_info["job_id"], "state": "completed"})

    async def cancel(request: web.Request) -> web.Response:
        received["cancel_id"] = request.match_info["job_id"]
        return web.json_response({"job_id": request.match_info["job_id"], "cancelled": True})

    async def queue(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "queue": {"interactive": 1, "flex": 2},
                "batches": {
                    "active_members": 2,
                    "admitted": 3,
                    "rejected": 1,
                    "members_admitted": 6,
                    "members_rejected": 2,
                },
            }
        )

    async def job_list(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "total": 1,
                "count": 1,
                "jobs": [
                    {
                        "job_id": "job-1",
                        "state": "running",
                        "priority_tier": "flex",
                        "payload": {"model": "qwen2.5:7b"},
                        "assigned_node_id": "m3-max",
                    }
                ],
            }
        )

    async def nodes(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "nodes": [
                    {
                        "node_id": "m3-max",
                        "role": "prefill",
                        "health": "healthy",
                        "current_batch_size": 1,
                        "max_batch_size": 4,
                        "power_headroom": 1.0,
                        "model_whitelist": ["qwen2.5:7b"],
                    }
                ]
            }
        )

    async def readyz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ready", "healthy_nodes": 1})

    app = web.Application()
    app.router.add_post("/v1/jobs", submit)
    app.router.add_get("/v1/jobs", job_list)
    app.router.add_get("/v1/jobs/{job_id}", job_status)
    app.router.add_delete("/v1/jobs/{job_id}", cancel)
    app.router.add_get("/v1/queue", queue)
    app.router.add_get("/v1/nodes", nodes)
    app.router.add_get("/readyz", readyz)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


async def test_the_client_submits_a_job(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(
            [
                "--base-url",
                url,
                "job",
                "submit",
                "--model",
                "qwen2.5:7b",
                "--prompt",
                "scan the repo",
                "--tier",
                "flex",
                "--latency-ms",
                "3600000",
                "--output-tokens",
                "2048",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "job-1" in out
    finally:
        await server.close()


async def test_the_queue_command_shows_batch_health(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "queue"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "batch: active=2 admitted=3 rejected=1" in out
        assert '"active_members": 2' in out
        assert '"rejected": 1' in out
        assert "warning: one or more batch admissions were rejected" in captured.err
    finally:
        await server.close()


async def test_the_client_shows_a_job(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "job", "status", "abc"])
        assert code == 0
        assert "abc" in capsys.readouterr().out
    finally:
        await server.close()


async def test_the_client_cancels_a_job(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "job", "cancel", "abc"])
        assert code == 0
        assert "cancelled" in capsys.readouterr().out
    finally:
        await server.close()


async def test_the_client_lists_the_jobs(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "job", "list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "job-1" in out
        assert "running" in out
        assert "qwen2.5:7b" in out
        assert "1 of 1 job(s)" in out
    finally:
        await server.close()


async def test_the_client_shows_the_queue_occupancy(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "queue"])
        assert code == 0
        assert "flex" in capsys.readouterr().out
    finally:
        await server.close()


async def test_the_client_shows_the_fleet(capsys) -> None:
    server, url = await start_broker()
    try:
        code = await run(["--base-url", url, "nodes"])
        assert code == 0
        out = capsys.readouterr().out
        assert "m3-max" in out
        assert "qwen2.5:7b" in out
    finally:
        await server.close()


async def test_the_client_reports_an_unreachable_broker(capsys) -> None:
    code = await run(["--base-url", "http://127.0.0.1:1", "nodes"])
    assert code == 1
    assert "could not reach the broker" in capsys.readouterr().err


async def test_the_client_reports_an_unknown_job(capsys) -> None:
    async def unknown(request: web.Request) -> web.Response:
        return web.json_response({"error": "unknown job"}, status=404)

    app = web.Application()
    app.router.add_get("/v1/jobs/{job_id}", unknown)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url("")).rstrip("/")
    try:
        code = await run(["--base-url", url, "job", "status", "nope"])
        assert code == 1
        assert "unknown job" in capsys.readouterr().err
    finally:
        await server.close()


def test_cli_rejects_non_finite_or_non_positive_timeouts() -> None:
    parser = build_parser()

    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--timeout", value, "health"])


@pytest.mark.parametrize(
    "value", ["bad", float("nan"), float("inf"), 10**1000, True, False]
)
def test_cli_renders_invalid_power_telemetry_as_unknown(value: Any) -> None:
    assert _display_power(value) == "-"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "any"),
        ("qwen2.5:7b", "any"),
        (["qwen2.5:7b", 7], "qwen2.5:7b"),
        ([" qwen2.5:7b", "   "], "any"),
        (["qwen2.5:7b", "qwen2.5:7b"], "qwen2.5:7b"),
    ],
)
def test_cli_renders_malformed_model_whitelists_safely(value: Any, expected: str) -> None:
    assert _display_models(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, True), (0, False), (True, False), (float("nan"), False), (10**1000, False)],
)
def test_cli_recognizes_only_finite_positive_counters(value: Any, expected: bool) -> None:
    assert _is_positive_finite_number(value) is expected


async def test_the_submitted_body_carries_the_tier(capsys) -> None:
    received: dict[str, Any] = {}

    async def submit(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"job_id": "job-1"}, status=202)

    app = web.Application()
    app.router.add_post("/v1/jobs", submit)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url("")).rstrip("/")
    try:
        await run(
            [
                "--base-url",
                url,
                "job",
                "submit",
                "--prompt",
                "hello",
                "--tier",
                "flex",
                "--latency-ms",
                "1234",
                "--output-tokens",
                "99",
            ]
        )
        assert received["priority_tier"] == "flex"
        assert received["max_acceptable_latency_ms"] == 1234
        assert received["payload"]["expected_output_tokens"] == 99
        assert received["payload"]["prompt"] == "hello"
    finally:
        await server.close()
