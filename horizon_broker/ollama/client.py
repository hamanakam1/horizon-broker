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
from typing import Any, NoReturn

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
            session = self._session
            try:
                await session.close()
            finally:
                self._session = None

    async def version(self) -> str:
        """Return the server version string."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/api/version", timeout=self._request_timeout()
        ) as response:
            if response.status != 200:
                raise OllamaError(f"version failed: HTTP {response.status}")
            body = await self._read_json(response, "version")
            if not isinstance(body, dict):
                raise OllamaError("version returned a non-object response")
            version = body.get("version")
            if (
                not isinstance(version, str)
                or not version.strip()
                or version != version.strip()
            ):
                raise OllamaError("version returned an invalid version")
            return version

    async def list_models(self) -> list[str]:
        """Return the identifiers of the models installed on the server."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/api/tags", timeout=self._request_timeout()
        ) as response:
            if response.status != 200:
                raise OllamaError(f"tags failed: HTTP {response.status}")
            body = await self._read_json(response, "tags")
            if not isinstance(body, dict) or not isinstance(body.get("models"), list):
                raise OllamaError("tags returned an invalid models response")
            names: list[str] = []
            for entry in body["models"]:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("name"), str)
                    or not entry["name"].strip()
                    or entry["name"] != entry["name"].strip()
                ):
                    raise OllamaError("tags returned a non-canonical model entry")
                if entry["name"] in names:
                    raise OllamaError("tags returned duplicate model entries")
                names.append(entry["name"])
            return names

    async def ps(self) -> list[dict[str, Any]]:
        """Return the models currently resident in memory."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/api/ps", timeout=self._request_timeout()
        ) as response:
            if response.status != 200:
                raise OllamaError(f"ps failed: HTTP {response.status}")
            body = await self._read_json(response, "ps")
            if not isinstance(body, dict) or not isinstance(body.get("models"), list):
                raise OllamaError("ps returned an invalid models response")
            names: set[str] = set()
            for entry in body["models"]:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("name"), str)
                    or not entry["name"].strip()
                    or entry["name"] != entry["name"].strip()
                ):
                    raise OllamaError("ps returned a non-canonical model entry")
                if entry["name"] in names:
                    raise OllamaError("ps returned duplicate model entries")
                names.add(entry["name"])
            return list(body["models"])

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
        # The configured budget bounds the whole generation; without it a
        # stalled engine would keep the job alive for ever.
        timeout = self._request_timeout()
        async with session.post(
            f"{self.base_url}{endpoint}", json=body, timeout=timeout
        ) as response:
            if response.status != 200:
                detail = await response.text()
                raise OllamaError(f"generate failed: HTTP {response.status} {detail[:400]}")
            async for raw_line in response.content:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise OllamaError("generation stream contains invalid UTF-8") from exc
                if not line:
                    continue
                try:
                    chunk = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise OllamaError("generation stream contains invalid JSON") from exc
                if not isinstance(chunk, dict):
                    raise OllamaError("generation stream record must be an object")
                if "error" in chunk:
                    error = chunk["error"]
                    if not isinstance(error, str):
                        raise OllamaError("generation stream error must be a string")
                    raise OllamaError(f"generation failed: {error[:400]}")
                done = chunk.get("done", False)
                if not isinstance(done, bool):
                    raise OllamaError("generation stream done must be a boolean")
                for counter_name in ("prompt_eval_count", "eval_count"):
                    counter = chunk.get(counter_name)
                    if counter is not None and (
                        isinstance(counter, bool)
                        or not isinstance(counter, int)
                        or counter < 0
                    ):
                        raise OllamaError(
                            f"generation stream {counter_name} must be a non-negative integer"
                        )
                message = chunk.get("message")
                done_reason = chunk.get("done_reason")
                if done_reason is not None and not isinstance(done_reason, str):
                    raise OllamaError("generation stream done_reason must be a string")
                model_name = chunk.get("model")
                if model_name is not None and (
                    not isinstance(model_name, str)
                    or not model_name.strip()
                    or model_name != model_name.strip()
                ):
                    raise OllamaError("generation stream model must be a canonical string")
                response_text = chunk.get("response", "")
                if not isinstance(response_text, str):
                    raise OllamaError("generation stream response must be a string")
                text = response_text
                if not text and isinstance(message, dict):
                    message_text = message.get("content", "")
                    if not isinstance(message_text, str):
                        raise OllamaError("generation stream message content must be a string")
                    text = message_text
                yield GenerationChunk(
                    text=text,
                    done=done,
                    done_reason=done_reason,
                    prompt_eval_count=chunk.get("prompt_eval_count"),
                    eval_count=chunk.get("eval_count"),
                    model=model_name,
                    raw=chunk,
                )
                if done:
                    return
            # HTTP success and a clean EOF can still carry a truncated answer.
            # Only the engine's terminal record establishes completion.
            raise OllamaError("generation stream ended without a completion marker")

    def _request_timeout(self) -> aiohttp.ClientTimeout:
        """Apply the configured budget even when a shared session is injected."""
        return aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            sock_connect=self.timeout_seconds,
            sock_read=self.timeout_seconds,
        )

    @staticmethod
    async def _read_json(response: aiohttp.ClientResponse, operation: str) -> Any:
        """Turn remote decoder/content-type failures into the client error type."""
        try:
            text = await response.text()
            return json.loads(text, parse_constant=_reject_json_constant)
        except (aiohttp.ClientError, ValueError) as exc:
            raise OllamaError(f"{operation} returned invalid JSON") from exc


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")
