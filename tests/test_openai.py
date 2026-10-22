"""The OpenAI-compatible ingress."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from horizon_broker.ingress.app import create_app
from horizon_broker.ingress.openai import OpenAITranslationError, chat_request_to_job
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_node
from tests.test_engine import start_fake_ollama


async def make_client(config: Any, ollama_url: str) -> tuple[TestClient, BrokerEngine]:
    node = make_node("node-a", ollama_url=ollama_url)
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    app = create_app(engine, config)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, engine


def test_a_chat_request_becomes_a_job() -> None:
    job = chat_request_to_job(
        {
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello there"},
            ],
            "max_tokens": 128,
            "temperature": 0.2,
        }
    )
    assert job.priority_tier.value == "interactive"
    assert job.payload.model == "qwen2.5:7b"
    assert job.payload.parameters.num_predict == 128
    assert job.payload.parameters.temperature == 0.2
    assert job.payload.system == "be brief"
    assert len(job.payload.messages) == 2
    assert job.payload.estimated_prompt_tokens > 0


def test_a_slow_request_becomes_flex_work() -> None:
    job = chat_request_to_job(
        {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "scan the repo"}],
            "horizon_max_acceptable_latency_ms": 3_600_000,
        }
    )
    assert job.priority_tier.value == "flex"


def test_an_explicit_tier_wins() -> None:
    job = chat_request_to_job(
        {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "scan the repo"}],
            "horizon_priority_tier": "interactive",
            "horizon_max_acceptable_latency_ms": 3_600_000,
        }
    )
    assert job.priority_tier.value == "interactive"


def test_a_request_without_messages_is_rejected() -> None:
    try:
        chat_request_to_job({"model": "qwen2.5:7b", "messages": []})
    except OpenAITranslationError as exc:
        assert "messages" in str(exc)
    else:
        raise AssertionError("the request should have been rejected")


def test_a_stop_string_becomes_a_tuple() -> None:
    job = chat_request_to_job(
        {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END", "STOP"],
        }
    )
    assert job.payload.parameters.stop == ("END", "STOP")


async def test_a_chat_completion_is_answered(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("Hello", " there"))
    client, engine = await make_client(config, url)
    await engine.start()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status == 200
        body = await response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello there"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["total_tokens"] > 0
    finally:
        await engine.stop()
        await client.close()
        await server.close()


async def test_a_streamed_chat_completion_uses_server_sent_events(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("Hi", " there"))
    client, engine = await make_client(config, url)
    await engine.start()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        text = await response.text()
        assert text.endswith("data: [DONE]\n\n")
        chunks = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: {")
        ]
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        assert content == "Hi there"
    finally:
        await engine.stop()
        await client.close()
        await server.close()


async def test_a_broken_request_is_a_422(config: Any) -> None:
    client, engine = await make_client(config, "http://127.0.0.1:1")
    await engine.start()
    try:
        response = await client.post("/v1/chat/completions", json={"model": "qwen2.5:7b"})
        assert response.status == 422
        assert "message" in (await response.json())["error"]
    finally:
        await engine.stop()
        await client.close()


async def test_a_failed_job_is_reported(config: Any) -> None:
    # No Ollama server behind this URL: the job fails, the caller is told.
    client, engine = await make_client(config, "http://127.0.0.1:1")
    await engine.start()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hi"}],
                "horizon_timeout_seconds": 5,
            },
        )
        assert response.status in (500, 504)
        assert "job_id" in (await response.json())["error"]
    finally:
        await engine.stop()
        await client.close()


async def test_the_model_list_comes_from_the_fleet(config: Any) -> None:
    client, engine = await make_client(config, "http://127.0.0.1:1")
    await engine.start()
    try:
        response = await client.get("/v1/models")
        assert response.status == 200
        body = await response.json()
        assert body["object"] == "list"
        assert [entry["id"] for entry in body["data"]] == ["qwen2.5:7b"]
    finally:
        await engine.stop()
        await client.close()
