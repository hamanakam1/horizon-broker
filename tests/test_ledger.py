"""The EdgeRecord HTTP port: retries, auth and the failure taxonomy."""

from __future__ import annotations

import json
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.config import LedgerConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.ledger import LedgerRecord
from horizon_broker.recovery.ledger import EdgeRecordClient


class _ResponseContext:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def __aenter__(self) -> Any:
        return self.response

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class _JsonResponse:
    status = 200

    def __init__(self, body: Any) -> None:
        self.body = body

    async def json(self) -> Any:
        return self.body

    async def text(self) -> str:
        return json.dumps(self.body)


class _RecordingSession:
    def __init__(self) -> None:
        self.commit_kwargs: dict[str, Any] = {}
        self.query_kwargs: dict[str, Any] = {}

    def post(self, url: str, **kwargs: Any) -> _ResponseContext:
        self.commit_kwargs = kwargs
        return _ResponseContext(_JsonResponse({"document_id": "doc-1"}))

    def get(self, url: str, **kwargs: Any) -> _ResponseContext:
        self.query_kwargs = kwargs
        owner_id = kwargs["params"]["owner_id"]
        payload = {"job_id": "job-1", "step": 1}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return _ResponseContext(
            _JsonResponse(
                {
                    "records": [
                        {
                            "document_id": "doc-1",
                            "payload": payload,
                            "content_hash": content_hash,
                            "tags": [f"broker:{owner_id}"],
                        }
                    ]
                }
            )
        )


async def test_shared_ledger_requests_keep_their_configured_timeout() -> None:
    """The ledger budget must survive injection of the broker's shared session."""
    config = LedgerConfig(enabled=True, timeout_seconds=1.25)
    session = _RecordingSession()
    client = EdgeRecordClient(config, session=session)  # type: ignore[arg-type]

    await client.commit(build_record())
    await client.latest_for_job("job-1")

    assert session.commit_kwargs["timeout"].total == 1.25
    assert session.query_kwargs["timeout"].total == 1.25
    assert session.commit_kwargs["allow_redirects"] is False
    assert session.query_kwargs["allow_redirects"] is False


async def test_shared_ledger_requests_are_scoped_to_the_broker_owner() -> None:
    config = LedgerConfig(enabled=True, owner_id="broker-a")
    session = _RecordingSession()
    client = EdgeRecordClient(config, session=session)  # type: ignore[arg-type]

    await client.commit(build_record())
    await client.latest_for_job("job-1")

    assert session.commit_kwargs["json"]["tags"][-1] == "broker:broker-a"
    assert session.query_kwargs["params"]["owner_id"] == "broker-a"


async def test_a_query_record_for_a_different_broker_is_rejected() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["horizon_broker", "broker:broker-b"],
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, owner_id="broker-a", max_retries=0)
    )
    try:
        with pytest.raises(LedgerUnavailableError, match="different broker"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_with_conflicting_owner_tags_is_rejected() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["broker:broker-a", "broker:broker-b"],
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, owner_id="broker-a", max_retries=0)
    )
    try:
        with pytest.raises(LedgerUnavailableError, match="different broker"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_from_a_different_collection_is_rejected() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "collection": "other.collection",
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["broker:horizon-broker"],
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="different collection"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def start_ledger(
    handler: Any,
) -> tuple[TestServer, str]:
    """Start a fake ledger server."""
    app = web.Application()
    app.router.add_route("*", "/collections/{collection}/records", handler)
    server = TestServer(app)
    await server.start_server()
    return server, str(server.make_url("")).rstrip("/")


def build_record(tags: tuple[str, ...] = ()) -> LedgerRecord:
    return LedgerRecord.build(
        collection="horizon_broker.snapshots",
        document_id="doc-1",
        payload={"job_id": "job-1", "step": 3},
        tags=tags,
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


def test_ledger_url_encodes_collection_delimiters() -> None:
    from horizon_broker.recovery.ledger import _ledger_url

    assert _ledger_url("http://ledger", "/collections/{collection}/records", "a/b?c") == (
        "http://ledger/collections/a%2Fb%3Fc/records"
    )


async def test_a_mutated_record_is_not_committed_with_a_stale_hash() -> None:
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url="http://ledger", max_retries=0))
    record = build_record()
    record.payload["step"] = 4

    with pytest.raises(LedgerUnavailableError, match="content_hash"):
        await client.commit(record)


async def test_a_non_json_mutation_is_not_committed() -> None:
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url="http://ledger", max_retries=0))
    record = build_record()
    record.payload["bad"] = object()

    with pytest.raises(LedgerUnavailableError, match="content_hash"):
        await client.commit(record)


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


async def test_a_redirected_commit_is_not_treated_as_success() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"document_id": "doc-1", "commit_url": "http://ledger/doc-1"},
            status=302,
            headers={"Location": "http://other-ledger/records"},
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="redirected"):
            await client.commit(build_record())
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


@pytest.mark.parametrize("body", [b"not-json", b"[]"])
async def test_a_malformed_commit_response_is_a_typed_failure(body: bytes) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="application/json")

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="commit returned"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize("body", [b"{}", b'{"content_hash":"hash"}'])
async def test_a_commit_without_identity_is_a_typed_failure(body: bytes) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="application/json")

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="no document_id"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


async def test_a_commit_with_a_padded_document_id_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"document_id": " doc-1"})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="non-canonical document_id"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize("content_hash", [b"", b"123"])
async def test_a_commit_with_an_invalid_content_hash_is_a_typed_failure(
    content_hash: bytes,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(
            {"document_id": body["document_id"], "content_hash": content_hash.decode()}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="(invalid|mismatched) content_hash"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize(
    "commit_url",
    [
        {"href": "wrong"},
        "",
        "relative/doc-1",
        "javascript:alert(1)",
        " http://ledger/doc-1",
        "https://user:secret@ledger/doc-1",
        "http://[malformed/doc-1",
        "http://ledger:not-a-port/doc-1",
        "http://:80/doc-1",
    ],
)
async def test_a_commit_with_an_invalid_url_is_a_typed_failure(commit_url: Any) -> None:
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(
            {"document_id": body["document_id"], "commit_url": commit_url}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid commit_url"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize(
    "body",
    [b"not-json", b"{}", b'{"records":"wrong"}', b'[{"document_id":"doc-1"}, 1]'],
)
async def test_a_malformed_query_response_is_a_typed_failure(body: bytes) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="application/json")

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="query returned"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [("document_id", "wrong-document"), ("content_hash", "wrong-hash")],
)
async def test_a_commit_identity_mismatch_is_rejected(field: str, value: str) -> None:
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        response = {"document_id": body["document_id"], "content_hash": body["content_hash"]}
        response[field] = value
        return web.json_response(response)

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="mismatched"):
            await client.commit(build_record())
    finally:
        await client.close()
        await server.close()


async def test_a_commit_with_a_conflicting_broker_tag_is_rejected() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"document_id": "doc-1"})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="conflicting broker"):
            await client.commit(build_record(tags=("broker:other",)))
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
        payload = {"job_id": "job-1", "step": 3}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["broker:horizon-broker"],
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


async def test_a_redirected_query_is_not_treated_as_success() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": []},
            status=302,
            headers={"Location": "http://other-ledger/records"},
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="redirected"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_without_a_broker_owner_tag_is_rejected() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["horizon_broker"],
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(
        LedgerConfig(enabled=True, base_url=url, owner_id="broker-a", max_retries=0)
    )
    try:
        with pytest.raises(LedgerUnavailableError, match="different broker"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_accepts_a_bare_list() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            [
                {
                    "document_id": "doc-1",
                    "payload": payload,
                    "content_hash": content_hash,
                    "tags": ["broker:horizon-broker"],
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


async def test_a_query_record_without_an_identity_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"records": [{"payload": {"job_id": "job-1"}}]})

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="document_id"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_with_a_padded_document_id_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": [{"document_id": " doc-1", "payload": {"job_id": "job-1"}}]}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="non-canonical document_id"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_with_a_non_string_identity_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": [{"document_id": 123, "payload": {"job_id": "job-1"}}]}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="document_id"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_without_a_content_hash_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": [{"document_id": "doc-1", "payload": {"job_id": "job-1"}}]}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid content_hash"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_with_an_unserializable_payload_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1", "step": float("nan")},
                        "content_hash": "0" * 64,
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid JSON"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_record_for_a_different_job_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": [{"document_id": "doc-1", "payload": {"job_id": "job-2"}}]}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="different job"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_content_hash_mismatch_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1"},
                        "content_hash": "wrong-hash",
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="mismatched content_hash"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_an_empty_content_hash_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1"},
                        "content_hash": "",
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid content_hash"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_malformed_tags_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1"},
                        "tags": "not-a-list",
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid tags"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_a_padded_tag_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"records": [{"document_id": "doc-1", "payload": {"job_id": "job-1"}, "tags": [" broker:a"]}]}
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid tags"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_duplicate_tags_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        payload = {"job_id": "job-1"}
        content_hash = LedgerRecord.build(
            collection="horizon_broker.snapshots",
            document_id="doc-1",
            payload=payload,
        ).content_hash
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": payload,
                        "content_hash": content_hash,
                        "tags": ["broker:horizon-broker", "broker:horizon-broker"],
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid record envelope"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_a_malformed_schema_version_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1"},
                        "schema_version": 1,
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="invalid schema_version"):
            await client.latest_for_job("job-1")
    finally:
        await client.close()
        await server.close()


async def test_a_query_with_an_unsupported_schema_version_is_a_typed_failure() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "records": [
                    {
                        "document_id": "doc-1",
                        "payload": {"job_id": "job-1"},
                        "schema_version": "2.0",
                    }
                ]
            }
        )

    server, url = await start_ledger(handler)
    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url=url, max_retries=0))
    try:
        with pytest.raises(LedgerUnavailableError, match="unsupported schema_version"):
            await client.latest_for_job("job-1")
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


async def test_ledger_forgets_owned_session_when_close_fails() -> None:
    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    client = EdgeRecordClient(LedgerConfig(enabled=True, base_url="http://synthetic"))
    client._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="session close failure"):
        await client.close()
    assert client._session is None


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
