"""Anthropic Adapter — translate OpenAI wire format ↔ Anthropic Messages API.

All agenthatch API calls use the OpenAI-compatible wire protocol
(openai.OpenAI client → /v1/chat/completions). This adapter translates
those calls to the Anthropic Messages API format and translates responses
back to OpenAI-compatible structures.

Supported features:
  - Non-streaming chat completions
  - Streaming chat completions (SSE delta translation)
  - Tool use (tool_use content blocks ↔ tool_calls)
  - System messages (OpenAI role=system → Anthropic top-level system param)
  - Thinking (extended thinking with budget_tokens)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agenthatch")

# ── OpenAI-compatible response dataclasses ────────────────────────────────


@dataclass
class _ToolCall:
    """OpenAI-compatible tool call."""
    id: str
    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Message:
    """OpenAI-compatible message."""
    role: str
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    """OpenAI-compatible choice."""
    index: int
    message: _Message
    finish_reason: str = "stop"
    delta: _Message | None = None


@dataclass
class _Usage:
    """OpenAI-compatible usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _StreamChoice:
    """OpenAI-compatible streaming choice."""
    index: int
    delta: _Message
    finish_reason: str | None = None


@dataclass
class _StreamChunk:
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[_StreamChoice] = field(default_factory=list)
    usage: _Usage | None = None


@dataclass
class _Response:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage | None = None


# ── Anthropic → OpenAI message translation ─────────────────────────────────


def _anthropic_content_to_openai(content_block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single Anthropic content block to OpenAI tool_call or text."""
    ctype = content_block.get("type", "")
    if ctype == "text":
        return {"type": "text", "text": content_block.get("text", "")}
    elif ctype == "tool_use":
        return {
            "type": "tool_use",
            "id": content_block.get("id", ""),
            "name": content_block.get("name", ""),
            "input": content_block.get("input", {}),
        }
    elif ctype == "thinking":
        return {"type": "thinking", "thinking": content_block.get("thinking", "")}
    return None


def _anthropic_message_to_openai_choice(
    message: Any, index: int, stop_reason: str
) -> _Choice:
    """Convert Anthropic message to OpenAI-compatible Choice."""
    content = ""
    tool_calls: list[_ToolCall] = []
    thinking_content = ""

    raw_content = message.content if hasattr(message, "content") else message.get("content", [])
    for block in raw_content:
        if isinstance(block, dict):
            ctype = block.get("type", "")
        else:
            ctype = getattr(block, "type", "")

        if ctype == "text":
            text = block.get("text", "") if isinstance(block, dict) else block.text
            content += text
        elif ctype == "tool_use":
            tc_id = block.get("id", "") if isinstance(block, dict) else block.id
            tc_name = block.get("name", "") if isinstance(block, dict) else block.name
            tc_input = block.get("input", {}) if isinstance(block, dict) else block.input
            tool_calls.append(_ToolCall(
                id=tc_id,
                function={"name": tc_name, "arguments": json.dumps(tc_input)},
            ))
        elif ctype == "thinking":
            thinking = block.get("thinking", "") if isinstance(block, dict) else getattr(block, "thinking", "")
            thinking_content += thinking

    # Fallback: if no text content but thinking blocks exist, use thinking as content
    if not content and thinking_content:
        content = thinking_content

    finish_reason = "stop"
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "end_turn":
        finish_reason = "stop"

    return _Choice(
        index=index,
        message=_Message(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        ),
        finish_reason=finish_reason,
    )


def _anthropic_stream_event_to_openai_delta(
    event: Any, model: str, chunk_id: str
) -> _StreamChunk:
    """Convert Anthropic SSE event to OpenAI-compatible streaming chunk."""
    delta_type = getattr(event, "type", "")

    if delta_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if block is None:
            return _StreamChunk(id=chunk_id, model=model)
        ctype = getattr(block, "type", "")
        if ctype == "tool_use":
            tc_id = getattr(block, "id", "")
            tc_name = getattr(block, "name", "")
            delta = _Message(
                role="assistant",
                tool_calls=[_ToolCall(
                    id=tc_id,
                    function={"name": tc_name, "arguments": ""},
                )],
            )
            return _StreamChunk(
                id=chunk_id, model=model,
                choices=[_StreamChoice(index=0, delta=delta)],
            )

    elif delta_type == "content_block_delta":
        delta_info = getattr(event, "delta", None)
        if delta_info is None:
            return _StreamChunk(id=chunk_id, model=model)
        ctype = getattr(delta_info, "type", "")

        if ctype == "text_delta":
            text = getattr(delta_info, "text", "")
            delta = _Message(role="assistant", content=text)
            return _StreamChunk(
                id=chunk_id, model=model,
                choices=[_StreamChoice(index=0, delta=delta)],
            )
        elif ctype == "input_json_delta":
            partial = getattr(delta_info, "partial_json", "")
            delta = _Message(
                role="assistant",
                tool_calls=[_ToolCall(
                    id="",
                    function={"name": "", "arguments": partial},
                )],
            )
            return _StreamChunk(
                id=chunk_id, model=model,
                choices=[_StreamChoice(index=0, delta=delta)],
            )

    elif delta_type == "message_delta":
        usage_info = getattr(event, "usage", None)
        stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
        finish = "stop"
        if stop_reason == "tool_use":
            finish = "tool_calls"
        elif stop_reason == "end_turn":
            finish = "stop"
        usage = None
        if usage_info:
            usage = _Usage(
                prompt_tokens=getattr(usage_info, "input_tokens", 0),
                completion_tokens=getattr(usage_info, "output_tokens", 0),
                total_tokens=getattr(usage_info, "input_tokens", 0) + getattr(usage_info, "output_tokens", 0),
            )
        return _StreamChunk(
            id=chunk_id, model=model,
            choices=[_StreamChoice(index=0, delta=_Message(role="assistant"), finish_reason=finish)],
            usage=usage,
        )

    elif delta_type == "message_stop":
        return _StreamChunk(
            id=chunk_id, model=model,
            choices=[_StreamChoice(index=0, delta=_Message(role="assistant"), finish_reason="stop")],
        )

    return _StreamChunk(id=chunk_id, model=model)


# ── OpenAI → Anthropic request translation ──────────────────────────────────


def _openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns:
        (system_prompt_or_blocks, messages_list)
        system_prompt_or_blocks: string if simple text, list if content blocks
    """
    system_parts: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []
    current_tool_calls: dict[int, list[dict[str, Any]]] = {}
    merge_tool_results: dict[int, dict[str, Any]] = {}

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                system_parts.extend(content)
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Store tool_use blocks for this assistant turn
                anthropic_content: list[dict[str, Any]] = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": json.loads(fn.get("arguments", "{}"))
                        if isinstance(fn.get("arguments"), str)
                        else fn.get("arguments", {}),
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": anthropic_content if anthropic_content else content,
                })
            else:
                anthropic_content = []
                if isinstance(content, str) and content:
                    anthropic_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    anthropic_content = content
                anthropic_messages.append({
                    "role": "assistant",
                    "content": anthropic_content or [{"type": "text", "text": ""}],
                })

        elif role == "tool":
            # Tool result message in Anthropic format
            tool_call_id = msg.get("tool_call_id", "")
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content if isinstance(content, str) else json.dumps(content),
                }],
            })

        elif role == "user":
            anthropic_content = []
            if isinstance(content, str):
                anthropic_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Base64 image support
                        img_url = item.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:"):
                            media_type, b64 = img_url.split(",", 1) if "," in img_url else ("image/jpeg", img_url)
                            media_type = media_type.replace("data:", "").split(";")[0]
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            })
                    else:
                        anthropic_content.append(item)
            anthropic_messages.append({
                "role": "user",
                "content": anthropic_content or [{"type": "text", "text": ""}],
            })

    # Convert system_parts to string or list
    if not system_parts:
        system = ""
    elif len(system_parts) == 1 and system_parts[0].get("type") == "text":
        system = system_parts[0]["text"]
    else:
        system = system_parts

    return system, anthropic_messages


def _openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI-format tools to Anthropic format."""
    if not tools:
        return None
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return anthropic_tools if anthropic_tools else None


# ── Adapter class ───────────────────────────────────────────────────────────


class AnthropicChatCompletions:
    """Anthropic adapter callable that mimics openai.chat.completions.create."""

    def __init__(self, client: Any):
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _Response | Any:
        """Create a chat completion via Anthropic API.

        Translates OpenAI-format parameters to Anthropic and back.
        """
        system, anthropic_messages = _openai_messages_to_anthropic(messages)
        anthropic_tools = _openai_tools_to_anthropic(tools)

        # Build extra_headers for thinking support
        if extra_body:
            thinking = extra_body.get("thinking")
            if thinking:
                anthropic_tools = [{
                    "type": "custom",
                    "name": "thinking",
                    "thinking": {
                        "type": thinking.get("type", "enabled"),
                        "budget_tokens": thinking.get("budget_tokens", 4000),
                    },
                }]
                if anthropic_tools:
                    anthropic_tools = [{
                        "type": "custom",
                        "name": "thinking",
                        "thinking": {
                            "type": thinking.get("type", "enabled"),
                            "budget_tokens": thinking.get("budget_tokens", 4000),
                        },
                    }]

        kwargs_for_api: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        if system:
            if isinstance(system, str):
                kwargs_for_api["system"] = system
            else:
                kwargs_for_api["system"] = system

        if anthropic_tools:
            kwargs_for_api["tools"] = anthropic_tools

        if temperature is not None and temperature > 0:
            kwargs_for_api["temperature"] = temperature

        # Handle tool_choice
        if tool_choice and tool_choice != "auto":
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                fn_name = tool_choice.get("function", {}).get("name", "")
                kwargs_for_api["tool_choice"] = {"type": "tool", "name": fn_name}
            elif tool_choice == "none":
                pass  # Anthropic: omit tools to disable
            elif tool_choice == "required":
                kwargs_for_api["tool_choice"] = {"type": "any"}

        if stream:
            return self._stream_create(model, kwargs_for_api)
        else:
            return self._sync_create(model, kwargs_for_api)

    def _sync_create(self, model: str, kwargs: dict[str, Any]) -> _Response:
        """Non-streaming completion via Anthropic API."""
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise

        choice = _anthropic_message_to_openai_choice(
            response, 0, getattr(response, "stop_reason", "end_turn")
        )
        usage = _Usage(
            prompt_tokens=getattr(response, "usage", None).input_tokens if getattr(response, "usage", None) else 0,
            completion_tokens=getattr(response, "usage", None).output_tokens if getattr(response, "usage", None) else 0,
            total_tokens=(
                getattr(response, "usage", None).input_tokens +
                getattr(response, "usage", None).output_tokens
            ) if getattr(response, "usage", None) else 0,
        )

        return _Response(
            id=getattr(response, "id", ""),
            model=getattr(response, "model", model),
            created=int(time.time()),
            choices=[choice],
            usage=usage,
        )

    def _stream_create(self, model: str, kwargs: dict[str, Any]) -> Any:
        """Streaming completion via Anthropic API, yielding OpenAI-format chunks."""
        import uuid

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    chunk = _anthropic_stream_event_to_openai_delta(event, model, chunk_id)
                    if chunk.choices:
                        yield chunk
        except Exception as e:
            logger.error("Anthropic streaming error: %s", e)
            raise


class AnthropicChat:
    """Mimics openai.chat namespace."""
    def __init__(self, client: Any):
        self.completions = AnthropicChatCompletions(client)


class AnthropicAdapter:
    """Adapter that wraps Anthropic Python SDK to look like openai.OpenAI.

    Usage:
        adapter = AnthropicAdapter(api_key="...", base_url="https://api.anthropic.com")
        # adapter.chat.completions.create(...) works like OpenAI
    """

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", **kwargs: Any):
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self.chat = AnthropicChat(self._client)