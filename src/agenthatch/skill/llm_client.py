"""LLMClient — Thin wrapper over v0.2 providers + openai SDK.

All AgentHarnesses use this single client interface.
Supports OpenAI-compatible APIs (including Anthropic proxy via base_url).

Usage:
    client = LLMClient(provider_name="openai")
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    result = client.chat_structured(messages=msgs, response_model=MyPydanticModel)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from agenthatch.exceptions import ApiKeyError
from agenthatch.providers import ProviderFeatures, get_default_provider, get_provider, resolve_api_key

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM call interface wrapping v0.2 provider management.

    All AgentHarnesses use this client for both simple chat and
    structured (Instructor) output calls.
    """

    def __init__(self, provider_name: str | None = None, model: str | None = None):
        """Initialize LLM client for a provider.

        Args:
            provider_name: Provider name (openai/anthropic/deepseek/ollama/custom.<name>).
                           If None, uses the default provider from config.
            model: Model name. If None, uses provider's default_model.
        """
        name = provider_name or get_default_provider()
        self._info = get_provider(name)
        api_key = resolve_api_key(name)
        if not api_key:
            raise ApiKeyError(f"No API key resolved for provider '{name}'")

        self._provider_name = name
        self._model = model or self._info.default_model
        self._features = self._info.features

        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self._info.base_url,
            timeout=120.0,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def features(self) -> ProviderFeatures:
        return self._features

    # ── Simple chat ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple chat completion. Returns response text."""
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    # ── Structured output (Instructor pattern) ───────────────────────

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type,
        model: str | None = None,
        max_retries: int = 2,
    ) -> Any:
        """Structured output via Instructor (LLM → Pydantic).

        Wraps instructor.from_openai() with retry loop.
        Returns validated Pydantic model instance.
        """
        import instructor

        # Mode.JSON required: glm-5-external in TOOLS mode serializes
        # nested Pydantic objects as JSON strings instead of native dicts,
        # causing validation errors like "Input should be an object".
        client = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
        return client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            response_model=response_model,
            max_retries=max_retries,
        )

    # ── v0.4 Tool Calling ──────────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> ToolCallResponse:
        """带 tool calling 的 chat completion.

        Degradation: if provider declares supports_tools=False,
        call chat() and return a text-only ToolCallResponse.
        """
        if not self._features.supports_tools:
            text = self.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
            return ToolCallResponse(text=text, tool_calls=[])

        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ToolCallResponse.from_openai(response)

    _SYNTHETIC_CHUNK_SIZE = 4

    def stream_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """流式 chat completion with tool calling.

        Degradation strategy:
          Phase 1: If provider declares supports_stream_tools=False,
                   skip streaming entirely, call chat_with_tools() synchronously,
                   then yield the result as synthetic deltas.
          Phase 2: If provider declares supports_stream_tools=True but the API
                   returns an error at runtime, catch the exception and fall back
                   to Phase 1 (synchronous with synthetic streaming).
        """
        if not self._features.supports_stream_tools:
            return self._stream_synthetic_fallback(messages, tools, model, temperature, max_tokens)

        try:
            return self._stream_native(messages, tools, model, temperature, max_tokens)
        except Exception as e:
            if self._is_stream_tools_error(e):
                logger.warning(
                    "Provider '%s' failed on stream+tools, falling back to synthetic streaming: %s",
                    self._provider_name, e,
                )
                self._features = ProviderFeatures(
                    supports_tools=self._features.supports_tools,
                    supports_stream_tools=False,
                    supports_json_mode=self._features.supports_json_mode,
                    supports_parallel_tool_calls=self._features.supports_parallel_tool_calls,
                )
                return self._stream_synthetic_fallback(messages, tools, model, temperature, max_tokens)
            raise

    def _stream_native(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """Native streaming with tool calling (OpenAI-compatible)."""
        import json

        stream = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        text_parts: list[str] = []
        tc_accum: dict[int, dict[str, str]] = {}

        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta is None:
                continue

            if delta.content:
                text_parts.append(delta.content)
                yield StreamDelta(type="text", content=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_accum:
                        tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
                    acc = tc_accum[idx]

                    if tc_delta.id:
                        acc["id"] = tc_delta.id

                    name_changed = False
                    if tc_delta.function and tc_delta.function.name:
                        acc["name"] += tc_delta.function.name
                        name_changed = True

                    if tc_delta.function and tc_delta.function.arguments:
                        acc["arguments"] += tc_delta.function.arguments
                        yield StreamDelta(
                            type="tool_call_args",
                            content=tc_delta.function.arguments,
                            tool_index=idx,
                        )

                    if name_changed:
                        yield StreamDelta(
                            type="tool_call_start",
                            content="",
                            tool_index=idx,
                            tool_name=acc["name"],
                            tool_id=acc["id"],
                        )

        final_tool_calls: list[ToolCall] = []
        for idx in sorted(tc_accum.keys()):
            acc = tc_accum[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            final_tool_calls.append(ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments=args,
            ))

        return ToolCallResponse(
            text="".join(text_parts) or None,
            tool_calls=final_tool_calls,
        )

    def _stream_synthetic_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """Synthetic streaming: call chat_with_tools() synchronously,
        then yield result as StreamDelta chunks.

        This provides the same Generator[StreamDelta, None, ToolCallResponse]
        interface as _stream_native, so ConversationLoop.stream() works
        identically regardless of provider capability.
        """
        response = self.chat_with_tools(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.text:
            for i in range(0, len(response.text), self._SYNTHETIC_CHUNK_SIZE):
                yield StreamDelta(type="text", content=response.text[i:i + self._SYNTHETIC_CHUNK_SIZE])

        for i, tc in enumerate(response.tool_calls):
            yield StreamDelta(
                type="tool_call_start",
                content="",
                tool_index=i,
                tool_name=tc.name,
                tool_id=tc.id,
            )

        return response

    @staticmethod
    def _is_stream_tools_error(exc: Exception) -> bool:
        """Detect whether an exception is caused by stream+tools incompatibility.

        Checks for common error patterns from OpenAI-compatible APIs:
          - "stream" and "tool" both mentioned in error message
          - HTTP 400 with tool/stream related error code
          - Zhipu AI specific error patterns
        """
        msg = str(exc).lower()
        has_stream = "stream" in msg
        has_tool = "tool" in msg or "function" in msg
        has_incompatible = "not support" in msg or "incompatible" in msg or "invalid" in msg

        if has_stream and (has_tool or has_incompatible):
            return True

        if hasattr(exc, "status_code") and exc.status_code == 400:
            if has_stream or has_tool:
                return True

        return False


# ── v0.4 Tool Calling Data Classes ──────────────────────────────────


@dataclass
class ToolCall:
    """LLM 返回的单个 tool call."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResponse:
    """LLM 响应：可能包含文本、tool calls、或两者皆有."""
    text: str | None = None
    tool_calls: list[ToolCall] = dc_field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @classmethod
    def from_openai(cls, response: Any) -> ToolCallResponse:
        """从 OpenAI ChatCompletion 对象构造."""
        import json

        msg = response.choices[0].message
        text = msg.content or None
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        return cls(text=text, tool_calls=tool_calls)


@dataclass
class StreamDelta:
    """流式输出的单个 delta 片段."""
    type: str
    content: str = ""
    tool_index: int | None = None
    tool_name: str | None = None
    tool_id: str | None = None