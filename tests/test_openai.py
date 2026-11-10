"""The OpenAI-compatible ingress."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from horizon_broker.contracts.jobs import PriorityTier
from horizon_broker.ingress.app import create_app
from horizon_broker.ingress.openai import OpenAITranslationError, chat_request_to_job, sse
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
    # The system prompt travels in ``payload.system`` only; the message
    # list keeps the conversation.
    assert len(job.payload.messages) == 1
    assert job.payload.messages[0].role == "user"
    assert job.payload.estimated_prompt_tokens > 0


def test_sse_rejects_non_standard_json_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        sse({"value": float("nan")})


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


def test_an_invalid_explicit_tier_is_a_translation_error() -> None:
    with pytest.raises(OpenAITranslationError, match="horizon_priority_tier"):
        chat_request_to_job(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "horizon_priority_tier": "urgent",
            }
        )


def test_a_request_without_messages_is_rejected() -> None:
    try:
        chat_request_to_job({"model": "qwen2.5:7b", "messages": []})
    except OpenAITranslationError as exc:
        assert "messages" in str(exc)
    else:
        raise AssertionError("the request should have been rejected")


def test_malformed_sampling_parameters_are_translation_errors() -> None:
    for field in ("temperature", "top_p", "top_k"):
        body = {"messages": [{"role": "user", "content": "hello"}], field: "invalid"}
        try:
            chat_request_to_job(body)
        except OpenAITranslationError as exc:
            assert "sampling parameters" in str(exc)
        else:
            raise AssertionError(f"{field} should have been rejected")


def test_a_padded_model_is_a_translation_error() -> None:
    with pytest.raises(OpenAITranslationError, match="model must not"):
        chat_request_to_job(
            {
                "model": " qwen2.5:7b",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )


def test_boolean_numeric_limits_are_not_treated_as_integers() -> None:
    job = chat_request_to_job(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": True,
            "horizon_max_acceptable_latency_ms": True,
        }
    )

    assert job.payload.parameters.num_predict is None
    assert job.payload.expected_output_tokens is None
    assert job.max_acceptable_latency_ms == 120_000


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_malformed_generation_limits_are_translation_errors(field: str) -> None:
    with pytest.raises(OpenAITranslationError, match=field):
        chat_request_to_job(
            {
                "messages": [{"role": "user", "content": "hello"}],
                field: "128",
            }
        )


def test_malformed_latency_extension_is_a_translation_error() -> None:
    with pytest.raises(OpenAITranslationError, match="horizon_max_acceptable_latency_ms"):
        chat_request_to_job(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "horizon_max_acceptable_latency_ms": "120000",
            }
        )


def test_boolean_sampler_values_are_translation_errors() -> None:
    for field in ("temperature", "top_p", "top_k"):
        body = {"messages": [{"role": "user", "content": "hello"}], field: True}
        try:
            chat_request_to_job(body)
        except OpenAITranslationError as exc:
            assert "sampling parameters" in str(exc)
        else:
            raise AssertionError(f"{field} should have been rejected")


def test_non_wire_sampler_numeric_types_are_translation_errors() -> None:
    for field, value in (("temperature", "0.5"), ("top_p", "0.5"), ("top_k", 10.5), ("seed", 10.5)):
        body = {"messages": [{"role": "user", "content": "hello"}], field: value}
        try:
            chat_request_to_job(body)
        except OpenAITranslationError as exc:
            assert "sampling parameters" in str(exc)
        else:
            raise AssertionError(f"{field} should have been rejected")


def test_non_string_caller_uses_the_safe_default() -> None:
    job = chat_request_to_job(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "horizon_caller": {"unexpected": "object"},
        }
    )

    assert job.caller == "openai"


def test_a_padded_caller_is_a_translation_error() -> None:
    with pytest.raises(OpenAITranslationError, match="caller must not"):
        chat_request_to_job(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "horizon_caller": " agent ",
            }
        )


def test_a_stop_string_becomes_a_tuple() -> None:
    job = chat_request_to_job(
        {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END", "STOP"],
        }
    )
    assert job.payload.parameters.stop == ("END", "STOP")


def test_non_string_stop_values_are_translation_errors() -> None:
    try:
        chat_request_to_job(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["END", 7],
            }
        )
    except OpenAITranslationError as exc:
        assert "stop values" in str(exc)
    else:
        raise AssertionError("non-string stop values should have been rejected")


def test_the_tier_follows_the_configured_latency_threshold() -> None:
    """The OpenAI ingress and the native ingress must agree on the tier.

    The native path uses ``scheduling.interactive_latency_threshold_ms``;
    the OpenAI path used a hard-coded 120 s, so the same work was
    categorised differently depending on the door it came through.
    """
    body = {
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "scan the repo"}],
        "horizon_max_acceptable_latency_ms": 60_000,
    }
    strict = chat_request_to_job(body, interactive_threshold_ms=2_000)
    assert strict.priority_tier is PriorityTier.FLEX

    lenient = chat_request_to_job(body, interactive_threshold_ms=120_000)
    assert lenient.priority_tier is PriorityTier.INTERACTIVE


def test_the_system_prompt_is_sent_once() -> None:
    """The system message must not reach the engine twice.

    It is carried in ``payload.system`` and inserted at the front of the
    chat request, so leaving it in the message list sends it twice.
    """
    body = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "scan the repo"},
        ],
    }
    job = chat_request_to_job(body)
    request = job.payload.to_chat_request()
    system_messages = [m for m in request["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "be brief"
    assert [m["role"] for m in request["messages"]] == ["system", "user"]


def test_payload_rendering_revalidates_mutated_messages() -> None:
    job = chat_request_to_job(
        {"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "hello"}]}
    )
    job.payload.messages.append({"role": "user", "content": "corrupt"})

    request = job.payload.to_chat_request()
    assert request["messages"][-1] == {"role": "user", "content": "corrupt"}


def test_a_request_without_a_system_message_is_untouched() -> None:
    body = {
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "scan the repo"}],
    }
    job = chat_request_to_job(body)
    request = job.payload.to_chat_request()
    assert [m["role"] for m in request["messages"]] == ["user"]
    assert job.payload.system is None


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


async def test_a_non_boolean_stream_flag_is_rejected(config: Any) -> None:
    server, url = await start_fake_ollama()
    client, _ = await make_client(config, url)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": "false",
            },
        )
        assert response.status == 422
        assert "stream must be a boolean" in (await response.json())["error"]["message"]
    finally:
        await client.close()
        await server.close()


async def test_a_non_object_chat_body_is_a_bad_request(config: Any) -> None:
    server, url = await start_fake_ollama()
    client, _ = await make_client(config, url)
    try:
        response = await client.post("/v1/chat/completions", json=[])
        assert response.status == 400
        assert (await response.json())["error"]["message"] == "body must be a JSON object"
    finally:
        await client.close()
        await server.close()


async def test_an_infinite_timeout_is_rejected_as_non_standard_json(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("ok",))
    client, engine = await make_client(config, url)
    await engine.start()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hello"}],
                "horizon_timeout_seconds": float("inf"),
            },
        )
        assert response.status == 400
        assert "valid JSON" in (await response.text())
    finally:
        await engine.stop()
        await client.close()
        await server.close()


async def test_an_overflowing_timeout_uses_the_bounded_default(config: Any) -> None:
    server, url = await start_fake_ollama(tokens=("ok",))
    client, engine = await make_client(config, url)
    await engine.start()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "hello"}],
                "horizon_timeout_seconds": 10**1000,
            },
        )
        assert response.status == 200
    finally:
        await engine.stop()
        await client.close()
        await server.close()


async def test_a_streamed_chat_completion_sends_the_tokens_as_they_arrive(config: Any) -> None:
    """``stream: true`` must stream.

    The whole answer used to be buffered and emitted as one content delta;
    a caller that asked for a stream got the same bytes at the end.
    """
    server, url = await start_fake_ollama(tokens=("Hi", " there", " friend"))
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
        text = await response.text()
        chunks = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: {")
        ]
        contents = [
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk.get("choices")
        ]
        # One delta per token, not one delta with the whole answer. The
        # opening role delta and the closing delta carry no content.
        assert [piece for piece in contents if piece] == ["Hi", " there", " friend"]
    finally:
        await engine.stop()
        await client.close()
        await server.close()


async def test_the_usage_block_reports_the_measured_tokens(config: Any) -> None:
    """The engine counted the tokens; an estimate would misreport the cost."""
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
        usage = (await response.json())["usage"]
        # The fake engine reports prompt_eval_count=8 and eval_count=2.
        assert usage["prompt_tokens"] == 8
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 10
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
