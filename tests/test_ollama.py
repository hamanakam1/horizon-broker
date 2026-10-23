"""The Ollama HTTP client."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horizon_broker.config import BrokerConfig
from horizon_broker.contracts.jobs import ChatMessage, InferencePayload, JobState
from horizon_broker.ollama.client import OllamaClient, OllamaError
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node
from tests.test_engine import start_fake_ollama


async def start_stream_server(body: bytes, endpoint: str = "/api/generate") -> TestServer:
    async def stream(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="application/x-ndjson")

    app = web.Application()
    app.router.add_post(endpoint, stream)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.parametrize("chat", [False, True], ids=["generate", "chat"])
@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b"", "completion marker"),
        (b'{"response":"partial","done":false}\n', "completion marker"),
        (b'{"error":"synthetic engine failure"}\n', "synthetic engine failure"),
        (b'{"response":', "invalid JSON"),
        (b"[]\n", "must be an object"),
        (b"\xff\n", "invalid UTF-8"),
        (b'{"done":"false"}\n', "done must be a boolean"),
        (b'{"eval_count":"1","done":false}\n', "eval_count must be a non-negative integer"),
        (b'{"prompt_eval_count":-1,"done":false}\n', "prompt_eval_count must be a non-negative integer"),
        (b'{"response":1,"done":false}\n', "response must be a string"),
        (b'{"message":{"content":1},"done":false}\n', "message content must be a string"),
        (b'{"error":{"message":"bad"}}\n', "error must be a string"),
        (b'{"done_reason":1,"done":false}\n', "done_reason must be a string"),
        (b'{"model":1,"done":false}\n', "model must be a canonical string"),
        (b'{"model":"","done":false}\n', "model must be a canonical string"),
        (b'{"model":" synthetic-model","done":false}\n', "model must be a canonical string"),
        (b'{"response":"done","done":true,"metadata":NaN}\n', "invalid JSON"),
    ],
    ids=[
        "empty",
        "truncated",
        "engine-error",
        "invalid-json",
        "array",
        "encoding",
        "done-type",
        "eval-count-type",
        "prompt-count-range",
        "response-type",
        "message-content-type",
        "error-type",
        "done-reason-type",
        "model-type",
        "model-empty",
        "model-padding",
        "non-standard-number",
    ],
)
async def test_invalid_streams_do_not_succeed(chat: bool, body: bytes, error: str) -> None:
    """A successful HTTP response alone does not prove generation completed."""
    server = await start_stream_server(body, "/api/chat" if chat else "/api/generate")
    client = OllamaClient(str(server.make_url("")))
    payload = InferencePayload(
        model="synthetic-model",
        prompt="hello",
        messages=[ChatMessage(role="user", content="hello")] if chat else [],
    )
    try:
        with pytest.raises(OllamaError, match=error):
            _ = [chunk async for chunk in client.generate(payload)]
    finally:
        await client.close()
        await server.close()


async def test_completion_marker_ends_the_stream() -> None:
    server = await start_stream_server(
        b'\n{"response":"finished","done":true,"eval_count":1}\ninvalid trailing bytes\n'
    )
    client = OllamaClient(str(server.make_url("")))
    try:
        chunks = [
            chunk async for chunk in client.generate(InferencePayload(model="synthetic-model"))
        ]
        assert len(chunks) == 1
        assert chunks[0].text == "finished"
        assert chunks[0].done is True
        assert chunks[0].eval_count == 1
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize(
    "ending",
    [b"", b'{"error":"synthetic engine failure"}\n', b"invalid JSON\n"],
    ids=["truncated", "engine-error", "malformed"],
)
async def test_invalid_stream_fails_job_and_releases_waiter(
    config: BrokerConfig, ending: bytes
) -> None:
    """Callers must never receive a partial answer marked as a successful job."""
    server = await start_stream_server(b'{"response":"partial","done":false}\n' + ending)
    node = make_node("synthetic-node", ollama_url=str(server.make_url("")))
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    await engine.start()
    try:
        job = make_job()
        await engine.submit(job)
        assert await asyncio.wait_for(engine.wait_for_result(job.job_id), timeout=2.0) is None
        assert job.state is JobState.FAILED
        assert job.result_text is None
        assert job.error
        assert engine.metrics.counter("horizon_jobs_failed_total") == 1
        assert engine.metrics.counter("horizon_jobs_completed_total") == 0
        assert not engine.registry.jobs_on(node.node_id)
    finally:
        await engine.stop(drain=False)
        await server.close()


async def test_the_generation_honours_the_configured_timeout() -> None:
    """A generation that overruns its budget must not hang forever.

    ``ClientTimeout(total=None)`` disables the configured budget, so a
    stalled engine would keep the job alive indefinitely.
    """
    server, url = await start_fake_ollama(tokens=("slow",), delay=5.0)
    client = OllamaClient(url, timeout_seconds=0.2)
    try:
        payload = InferencePayload(model="qwen2.5:7b", prompt="hello")
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            async for _ in client.generate(payload):
                pass
    finally:
        await client.close()
        await server.close()


async def test_a_generation_streams_its_tokens() -> None:
    server, url = await start_fake_ollama(tokens=("Hello", " world"))
    client = OllamaClient(url, timeout_seconds=30.0)
    try:
        payload = InferencePayload(model="qwen2.5:7b", prompt="hello")
        chunks = [chunk.text async for chunk in client.generate(payload)]
        assert "".join(chunks) == "Hello world"
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize("chat", [False, True], ids=["generate", "chat"])
async def test_model_overrides_must_be_canonical(chat: bool) -> None:
    payload = InferencePayload(
        model="synthetic-model",
        messages=[ChatMessage(role="user", content="hello")] if chat else [],
    )
    render = payload.to_chat_request if chat else payload.to_engine_request
    with pytest.raises(ValueError, match="model override"):
        render(model_override=" synthetic-model")


@pytest.mark.parametrize(
    ("path", "method", "body", "error"),
    [
        ("/api/version", "version", "[]", "version returned"),
        ("/api/version", "version", '{"version":1}', "invalid version"),
        ("/api/version", "version", '{"version":""}', "invalid version"),
        ("/api/version", "version", '{"version":" 0.1 "}', "invalid version"),
        ("/api/tags", "list_models", '{"models":["bad"]}', "model entry"),
        ("/api/tags", "list_models", '{"models":[{"name":" "}]}', "model entry"),
        ("/api/tags", "list_models", '{"models":[{"name":" model"}]}', "non-canonical"),
        ("/api/tags", "list_models", '{"models":[{"name":"model"},{"name":"model"}]}', "duplicate"),
        ("/api/ps", "ps", '{"models":["bad"]}', "model entry"),
        ("/api/ps", "ps", '{"models":[{}]}', "model entry"),
        ("/api/ps", "ps", '{"models":[{"name":"model "}]}', "non-canonical"),
        ("/api/ps", "ps", '{"models":[{"name":"model"},{"name":"model"}]}', "duplicate"),
        ("/api/version", "version", "not-json", "invalid JSON"),
        ("/api/tags", "list_models", "not-json", "invalid JSON"),
        ("/api/ps", "ps", "not-json", "invalid JSON"),
    ],
)
async def test_malformed_metadata_responses_are_typed_failures(
    path: str, method: str, body: str, error: str
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text=body, content_type="application/json")

    app = web.Application()
    app.router.add_get(path, handler)
    server = TestServer(app)
    await server.start_server()
    client = OllamaClient(str(server.make_url("")))
    try:
        with pytest.raises(OllamaError, match=error):
            await getattr(client, method)()
    finally:
        await client.close()
        await server.close()


async def test_ollama_forgets_owned_session_when_close_fails() -> None:
    class Session:
        async def close(self) -> None:
            raise RuntimeError("synthetic session close failure")

    client = OllamaClient("http://synthetic")
    client._session = Session()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="session close failure"):
        await client.close()
    assert client._session is None
