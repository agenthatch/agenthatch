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
  - Prompt caching (cache_control breakpoints on system/tools/last message)
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
    """OpenAI-compatible usage.

    Prompt caching: Anthropic reports cached tokens separately
    (cache_read_input_tokens / cache_creation_input_tokens) and excludes
    them from input_tokens. prompt_tokens/total_tokens fold them in so
    OpenAI-format consumers see the full input size.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


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


def _is_opus_47_or_later(model: str) -> bool:
    """Check if model is Opus 4.7+ (temperature/top_p/top_k removed)."""
    model_lower = model.lower()
    return any(
        m in model_lower
        for m in ("claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-7", "claude-sonnet-4-8")
    )


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


_CACHE_CONTROL = {"type": "ephemeral"}


def _apply_cache_control(
    system: str | list[dict[str, Any]],
    anthropic_messages: list[dict[str, Any]],
    anthropic_tools: list[dict[str, Any]] | None,
) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Add cache_control breakpoints for Anthropic prompt caching.

    Marks up to 3 of the 4 breakpoints Anthropic allows:
      1. Last system block — caches the system prompt (stable across turns)
      2. Last tool definition — caches the tool schemas
      3. Last content block of the final message — caches the conversation
         prefix as it grows turn over turn

    Blocks are copied before marking so the caller's message objects are
    never mutated (they are reused across agent-loop turns).

    Cached input is billed at 10% (read) / 125% (write, 5-min TTL) of the
    base input price — a clear net win for agent loops that resend the
    same system prompt + tools + history every turn.
    """
    if system:
        if isinstance(system, str):
            system = [{
                "type": "text",
                "text": system,
                "cache_control": dict(_CACHE_CONTROL),
            }]
        else:
            system = [dict(b) if isinstance(b, dict) else b for b in system]
            # Only dict blocks can carry cache_control; the translation layer
            # passes through non-dict items verbatim, so guard before marking.
            if isinstance(system[-1], dict):
                system[-1]["cache_control"] = dict(_CACHE_CONTROL)

    if anthropic_tools:
        anthropic_tools = [dict(t) for t in anthropic_tools]
        anthropic_tools[-1]["cache_control"] = dict(_CACHE_CONTROL)

    if anthropic_messages:
        last = anthropic_messages[-1]
        content = last.get("content")
        if isinstance(content, list) and content:
            content = [dict(b) if isinstance(b, dict) else b for b in content]
            if isinstance(content[-1], dict):
                content[-1]["cache_control"] = dict(_CACHE_CONTROL)
                anthropic_messages = anthropic_messages[:-1] + [
                    {**last, "content": content}
                ]

    return system, anthropic_messages, anthropic_tools


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

        # Prompt caching: mark system/tools/last-message breakpoints
        system, anthropic_messages, anthropic_tools = _apply_cache_control(
            system, anthropic_messages, anthropic_tools
        )

        # Build thinking parameter for extended thinking support
        # Opus 4.6+/Sonnet 4.6+: adaptive thinking (no budget_tokens)
        # Opus 4.7/4.8: adaptive only — budget_tokens returns 400
        # Supports optional thinking.display for 4.7/4.8: "omitted" (default) or "summarized"
        thinking_config: dict[str, Any] | None = None
        if extra_body:
            thinking = extra_body.get("thinking")
            if thinking:
                thinking_config = {
                    "type": thinking.get("type", "adaptive"),
                }
                # Only include display if explicitly set (default is "omitted" on 4.7/4.8)
                if "display" in thinking:
                    thinking_config["display"] = thinking["display"]
                # budget_tokens is deprecated on 4.6+ and removed on 4.7/4.8
                # Only include if explicitly provided for backward compat with older models
                if "budget_tokens" in thinking and thinking.get("type") == "enabled":
                    thinking_config["budget_tokens"] = thinking["budget_tokens"]

        kwargs_for_api: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        if thinking_config:
            kwargs_for_api["thinking"] = thinking_config

        # Effort parameter (GA, no beta header)
        # Controls thinking depth: "low"|"medium"|"high"|"max"|"xhigh" (Opus 4.7+)
        # Best practice: "xhigh" for coding/agentic use cases on Opus 4.7/4.8,
        # "high" as minimum for intelligence-sensitive work
        if extra_body:
            effort = extra_body.get("effort")
            if effort:
                kwargs_for_api["output_config"] = {"effort": effort}

        if system:
            if isinstance(system, str):
                kwargs_for_api["system"] = system
            else:
                kwargs_for_api["system"] = system

        if anthropic_tools:
            kwargs_for_api["tools"] = anthropic_tools

        # Opus 4.7/4.8: temperature, top_p, top_k are removed (400 if passed)
        # Only include temperature for pre-4.7 models
        if temperature is not None and temperature > 0:
            if not _is_opus_47_or_later(model):
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

        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
            output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
            cache_read = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_creation:
                logger.debug(
                    "Anthropic prompt cache: read=%d created=%d",
                    cache_read, cache_creation,
                )
        else:
            input_tokens = output_tokens = cache_read = cache_creation = 0
        usage = _Usage(
            prompt_tokens=input_tokens + cache_read + cache_creation,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + cache_read + cache_creation + output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
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

        # Input-side tokens (incl. prompt-cache reads/writes) arrive in the
        # message_start event; output tokens arrive in message_delta. Capture
        # the former and merge into the final usage chunk so OpenAI-format
        # consumers see the full picture in one place.
        input_prompt_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0

        try:
            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "message_start":
                        message = getattr(event, "message", None)
                        usage_info = getattr(message, "usage", None) if message else None
                        if usage_info:
                            cache_read_tokens = (
                                getattr(usage_info, "cache_read_input_tokens", 0) or 0
                            )
                            cache_creation_tokens = (
                                getattr(usage_info, "cache_creation_input_tokens", 0) or 0
                            )
                            input_prompt_tokens = (
                                (getattr(usage_info, "input_tokens", 0) or 0)
                                + cache_read_tokens
                                + cache_creation_tokens
                            )
                            if cache_read_tokens or cache_creation_tokens:
                                logger.debug(
                                    "Anthropic prompt cache: read=%d created=%d",
                                    cache_read_tokens,
                                    cache_creation_tokens,
                                )
                        continue
                    chunk = _anthropic_stream_event_to_openai_delta(event, model, chunk_id)
                    if chunk.choices:
                        if etype == "message_delta" and chunk.usage:
                            # Anthropic's message_delta.usage only carries
                            # output_tokens; input-side tokens live in
                            # message_start. Merge only when the delta didn't
                            # already report input tokens (some OpenAI-compatible
                            # gateways replay full usage there — avoid double
                            # counting).
                            if input_prompt_tokens and not chunk.usage.prompt_tokens:
                                chunk.usage.prompt_tokens += input_prompt_tokens
                                chunk.usage.total_tokens += input_prompt_tokens
                            # Propagate cache fields so streaming consumers
                            # (TokenCounter) see the same data as non-streaming.
                            chunk.usage.cache_read_input_tokens = cache_read_tokens
                            chunk.usage.cache_creation_input_tokens = cache_creation_tokens
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