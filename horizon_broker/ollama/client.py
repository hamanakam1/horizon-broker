"""Ollama client.

The broker speaks Ollama's HTTP API for nodes that execute directly against a
running Ollama server (the M3 Max in the home rack, a Pi node on the UniFi
mesh). Responses are streamed as NDJSON so that the supervisor can emit
checkpoints while the generation is still running.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from horizon_broker.contracts.jobs import InferencePayload


class OllamaError(RuntimeError):
    """The Ollama endpoint returned an error."""


@dataclass(frozen=True)
class GenerationChunk:
    """One streamed chunk of a generation."""

    text: str = ""
    done: bool = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Minimal asynchronous Ollama HTTP client."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 600.0,
        keep_alive: str = "30m",
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying session if this client owns it."""
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None

    async def version(self) -> str:
        """Return the server version string."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/version") as response:
            if response.status != 200:
                raise OllamaError(f"version failed: HTTP {response.status}")
            body = await response.json()
            return str(body.get("version", "unknown"))

    async def list_models(self) -> list[str]:
        """Return the identifiers of the models installed on the server."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/tags") as response:
            if response.status != 200:
                raise OllamaError(f"tags failed: HTTP {response.status}")
            body = await response.json()
            return [str(entry.get("name", "")) for entry in body.get("models", [])]

    async def ps(self) -> list[dict[str, Any]]:
        """Return the models currently resident in memory."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/ps") as response:
            if response.status != 200:
                raise OllamaError(f"ps failed: HTTP {response.status}")
            body = await response.json()
            return list(body.get("models", []))

    async def generate(
        self,
        payload: InferencePayload,
        *,
        model: str | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream a completion for ``payload``.

        A payload with chat messages is sent to ``/api/chat`` so the model's
        own chat template is applied; a flat prompt goes to ``/api/generate``.
        Flattening a conversation by hand would silently change the prompt.
        """
        session = await self._get_session()
        if payload.messages:
            body = payload.to_chat_request(model_override=model)
            endpoint = "/api/chat"
        else:
            body = payload.to_engine_request(model_override=model)
            endpoint = "/api/generate"
        body["keep_alive"] = self.keep_alive
        timeout = aiohttp.ClientTimeout(total=None)
        async with session.post(
            f"{self.base_url}{endpoint}", json=body, timeout=timeout
        ) as response:
            if response.status != 200:
                detail = await response.text()
                raise OllamaError(f"generate failed: HTTP {response.status} {detail[:400]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message")
                text = str(chunk.get("response", ""))
                if not text and isinstance(message, dict):
                    text = str(message.get("content", ""))
                yield GenerationChunk(
                    text=text,
                    done=bool(chunk.get("done", False)),
                    done_reason=chunk.get("done_reason"),
                    prompt_eval_count=chunk.get("prompt_eval_count"),
                    eval_count=chunk.get("eval_count"),
                    model=chunk.get("model"),
                    raw=chunk,
                )
