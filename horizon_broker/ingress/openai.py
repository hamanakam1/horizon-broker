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

DEFAULT_INTERACTIVE_THRESHOLD_MS = 2_000
"""Latency budget at or below which a request is interactive work.

The native ingress uses ``scheduling.interactive_latency_threshold_ms``;
the OpenAI ingress uses the same number, so the same work is categorised
the same way whichever door it came through.
"""


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
        if not all(isinstance(item, str) for item in stop):
            raise OpenAITranslationError("stop values must be strings")
        stop_values = tuple(stop)
    else:
        stop_values = ()
    for field_name in ("max_tokens", "max_completion_tokens"):
        value = body.get(field_name)
        if (
            value is not None
            and not isinstance(value, bool)
            and (not isinstance(value, int) or value <= 0)
        ):
            raise OpenAITranslationError(f"{field_name} must be a positive integer")
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    num_predict = (
        int(max_tokens)
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0
        else None
    )
    try:
        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 0.95)
        top_k = body.get("top_k", 40)
        seed = body.get("seed")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be numeric")
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
            raise ValueError("top_p must be numeric")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer")
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_predict=num_predict,
            seed=seed,
            stop=stop_values,
        )
    except (TypeError, ValueError) as exc:
        raise OpenAITranslationError("sampling parameters are invalid") from exc


def chat_request_to_job(
    body: dict[str, Any],
    *,
    interactive_threshold_ms: int = DEFAULT_INTERACTIVE_THRESHOLD_MS,
) -> InferenceJob:
    """Translate an OpenAI chat-completions request into a broker job."""
    if not isinstance(body, dict):
        raise OpenAITranslationError("the request body must be an object")
    messages = _messages_from(body)
    model = body.get("model")
    if not isinstance(model, str) or not model:
        model = DEFAULT_MODEL
    elif model != model.strip():
        raise OpenAITranslationError("model must not have surrounding whitespace")

    # The system prompt travels in ``payload.system``, which the chat
    # request inserts at the front. Leaving it in the message list as well
    # would send it to the engine twice.
    system_message = next((m for m in messages if m.role == "system"), None)
    system = system_message.content if system_message is not None else None
    remaining = [m for m in messages if m is not system_message]
    payload = InferencePayload(
        model=model,
        prompt="",
        messages=remaining,
        system=system,
        parameters=_parameters_from(body),
        expected_output_tokens=body.get("max_tokens")
        if isinstance(body.get("max_tokens"), int)
        and not isinstance(body.get("max_tokens"), bool)
        else None,
    )

    tier = body.get("horizon_priority_tier")
    latency = body.get("horizon_max_acceptable_latency_ms")
    if tier is not None and tier not in (PriorityTier.INTERACTIVE.value, PriorityTier.FLEX.value):
        raise OpenAITranslationError("horizon_priority_tier must be interactive or flex")
    if (
        latency is not None
        and not isinstance(latency, bool)
        and (not isinstance(latency, int) or latency <= 0)
    ):
        raise OpenAITranslationError(
            "horizon_max_acceptable_latency_ms must be a positive integer"
        )
    if tier is None:
        tier = (
            PriorityTier.FLEX.value
            if (
                isinstance(latency, int)
                and not isinstance(latency, bool)
                and latency > interactive_threshold_ms
            )
            else PriorityTier.INTERACTIVE.value
        )
    if not isinstance(latency, int) or isinstance(latency, bool) or latency <= 0:
        latency = DEFAULT_LATENCY_MS
    caller = body.get("horizon_caller", "openai")
    if not isinstance(caller, str) or not caller.strip():
        caller = "openai"
    elif caller != caller.strip():
        raise OpenAITranslationError("caller must not have surrounding whitespace")

    return InferenceJob(
        priority_tier=PriorityTier(tier),
        payload=payload,
        max_acceptable_latency_ms=latency,
        caller=caller,
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
    return f"data: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n".encode()


DONE = b"data: [DONE]\n\n"
