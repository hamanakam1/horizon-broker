"""The Ollama HTTP client."""

from __future__ import annotations

import asyncio

import pytest

from horizon_broker.contracts.jobs import InferencePayload
from horizon_broker.ollama.client import OllamaClient
from tests.test_engine import start_fake_ollama


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
