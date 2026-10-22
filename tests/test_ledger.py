"""The EdgeRecord HTTP port: retries, auth and the failure taxonomy."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.config import LedgerConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.ledger import LedgerRecord
from horizon_broker.recovery.ledger import EdgeRecordClient


async def start_ledger(
    handler: Any,
) -> tuple[TestServer, str]:
    """Start a fake ledger server."""
    app = web.Application()
    app.router.add_route("*", "/collections/{collection}/records", handler)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


def build_record() -> LedgerRecord:
    return LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id="doc-1",
        payload={"job_id": "job-1", "step": 3},
    )


async def test_a_commit_returns_the_receipt() -> None:
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(
            {
                "document_id": body["document_id"],
                "content_hash": body["content_hash"],
                "commit_url": "http://ledger/doc-1",
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        receipt = await client.commit(build_record())
        assert receipt.document_id == "doc-1"
        assert receipt.commit_url == "http://ledger/doc-1"
    finally:
        await client.close()
        await server.close()


async def test_a_transient_failure_is_retried() -> None:
    attempts = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return web.Response(status=503, text="ledger warming up")
        body = await request.json()
        return web.json_response({"document_id": body["document_id"]})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, max_retries=3, retry_backoff_seconds=0.01)
    )
    try:
        receipt = await client.commit(build_record())
        assert receipt.document_id == "doc-1"
        assert attempts["count"] == 3
    finally:
        await client.close()
        await server.close()


async def test_a_permanent_rejection_is_not_retried() -> None:
    attempts = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        attempts["count"] += 1
        return web.Response(status=400, text="malformed document")

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, max_retries=3, retry_backoff_seconds=0.01)
    )
    try:
        try:
            await client.commit(build_record())
        except LedgerUnavailableError as exc:
            assert "rejected" in str(exc)
        else:
            raise AssertionError("the commit should have failed")
        assert attempts["count"] == 1
    finally:
        await client.close()
        await server.close()


async def test_the_retries_are_exhausted() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=503, text="down")

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, max_retries=1, retry_backoff_seconds=0.01)
    )
    try:
        try:
            await client.commit(build_record())
        except LedgerUnavailableError:
            pass
        else:
            raise AssertionError("the commit should have failed")
    finally:
        await client.close()
        await server.close()


async def test_the_token_is_sent_when_one_is_configured(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization", "")
        body = await request.json()
        return web.json_response({"document_id": body["document_id"]})

    server, url = await start_ledger(handler)
    monkeypatch.setenv("EDGERECORD_TOKEN", "s3cret")
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        await client.commit(build_record())
        assert seen["authorization"] == "Bearer s3cret"
    finally:
        await client.close()
        await server.close()


async def test_no_auth_header_without_a_token(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization", "")
        body = await request.json()
        return web.json_response({"document_id": body["document_id"]})

    server, url = await start_ledger(handler)
    monkeypatch.delenv("EDGERECORD_TOKEN", raising=False)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        await client.commit(build_record())
        assert seen["authorization"] == ""
    finally:
        await client.close()
        await server.close()


async def test_a_query_returns_the_newest_record() -> None:
    async def handler(request: web.Request) -> web.Response:
        assert request.query["job_id"] == "job-1"
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1", "step": 3},
                        "content_hash": "abc",
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        record = await client.latest_for_job("job-1")
        assert record is not None
        assert record.payload["step"] == 3
    finally:
        await client.close()
        await server.close()


async def test_a_query_accepts_a_bare_list() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            [
                {
                    "document_id": "doc-1",
                    "payload": {"job_id": "job-1"},
                    "content_hash": "abc",
                }
            ]
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        record = await client.latest_for_job("job-1")
        assert record is not None
        assert record.document_id == "doc-1"
    finally:
        await client.close()
        await server.close()


async def test_a_missing_record_is_none() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"records": []})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        assert await client.latest_for_job("job-1") is None
    finally:
        await client.close()
        await server.close()


async def test_a_disabled_ledger_refuses_to_commit() -> None:
    client = EdgeRecordClient(LedgerConfig(enabled=False))
    try:
        try:
            await client.commit(build_record())
        except LedgerUnavailableError as exc:
            assert "disabled" in str(exc)
        else:
            raise AssertionError("the commit should have failed")
    finally:
        await client.close()


async def test_the_committed_body_carries_the_content_hash() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured.update(await request.json())
        return web.json_response({"document_id": "doc-1"})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url))
    try:
        record = build_record()
        await client.commit(record)
        assert captured["content_hash"] == record.content_hash
        assert json.loads(json.dumps(captured["payload"]))["job_id"] == "job-1"
    finally:
        await client.close()
        await server.close()
