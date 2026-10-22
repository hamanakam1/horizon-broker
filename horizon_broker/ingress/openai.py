"""OpenAI-compatible translation.

Agents, SDKs and CLIs speak the OpenAI chat-completions API. The broker
accepts that shape so an agent can be pointed at the broker instead of at
Ollama, without rewriting anything: the request is translated into an
:class:`InferenceJob`, queued like any other work, and the answer is
translated back.

Two extension fields are accepted and ignored by other clients:

``horizon_priority_tier``
    ``interactive`` or ``flex``. Without it the tier follows the latency
    budget in ``horizon_max_acceptable_latency_ms``.

``horizon_max_acceptable_latency_ms``
    The latency budget of the job. Without it the request is treated as
    interactive.
"""

from __future__ import annotations

import json
import time
from typing import Any

from horizon_broker.contracts.jobs import (
    ChatMessage,
    InferenceJob,
    InferencePayload,
    PriorityTier,
    SamplingParams,
)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_LATENCY_MS = 120_000
"""Default budget for an OpenAI-shaped request without an explicit one."""


class OpenAITranslationError(ValueError):
    """The request body is not a usable chat-completions request."""


def _messages_from(body: dict[str, Any]) -> list[ChatMessage]:
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise OpenAITranslationError("messages must be a non-empty list")
    messages: list[ChatMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise OpenAITranslationError("each message must be an object")
        role = item.get("role", "user")
        content = item.get("content")
        if not isinstance(content, str):
            raise OpenAITranslationError("message content must be a string")
        if role not in ("system", "user", "assistant", "tool"):
            raise OpenAITranslationError(f"unsupported message role: {role}")
        messages.append(ChatMessage(role=role, content=content))
    return messages


def _parameters_from(body: dict[str, Any]) -> SamplingParams:
    stop = body.get("stop")
    stop_values: tuple[str, ...]
    if isinstance(stop, str):
        stop_values = (stop,)
    elif isinstance(stop, list):
        stop_values = tuple(str(item) for item in stop)
    else:
        stop_values = ()
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    num_predict = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else None
    seed = body.get("seed")
    return SamplingParams(
        temperature=float(body.get("temperature", 0.8)),
        top_p=float(body.get("top_p", 0.95)),
        top_k=int(body.get("top_k", 40)),
        num_predict=num_predict,
        seed=int(seed) if isinstance(seed, int) else None,
        stop=stop_values,
    )


def chat_request_to_job(body: dict[str, Any]) -> InferenceJob:
    """Translate an OpenAI chat-completions request into a broker job."""
    if not isinstance(body, dict):
        raise OpenAITranslationError("the request body must be an object")
    messages = _messages_from(body)
    model = body.get("model")
    if not isinstance(model, str) or not model:
        model = DEFAULT_MODEL

    system = next((m.content for m in messages if m.role == "system"), None)
    payload = InferencePayload(
        model=model,
        prompt="",
        messages=messages,
        system=system,
        parameters=_parameters_from(body),
        expected_output_tokens=body.get("max_tokens")
        if isinstance(body.get("max_tokens"), int)
        else None,
    )

    tier = body.get("horizon_priority_tier")
    latency = body.get("horizon_max_acceptable_latency_ms")
    if tier not in (PriorityTier.INTERACTIVE.value, PriorityTier.FLEX.value):
        tier = PriorityTier.FLEX.value if isinstance(latency, int) and latency > 120_000 else (
            PriorityTier.INTERACTIVE.value
        )
    if not isinstance(latency, int) or latency <= 0:
        latency = DEFAULT_LATENCY_MS

    return InferenceJob(
        priority_tier=PriorityTier(tier),
        payload=payload,
        max_acceptable_latency_ms=latency,
        caller=body.get("horizon_caller", "openai"),
    )


def completion_object(
    job: InferenceJob,
    text: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Render the non-streaming chat-completion response."""
    return {
        "id": f"chatcmpl-{job.job_id}",
        "object": "chat.completion",
        "created": int(job.submitted_at.timestamp()),
        "model": job.payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chunk_object(
    job: InferenceJob,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Render one streamed chat-completion chunk."""
    return {
        "id": f"chatcmpl-{job.job_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": job.payload.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse(payload: dict[str, Any]) -> bytes:
    """Render one server-sent event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


DONE = b"data: [DONE]\n\n"
