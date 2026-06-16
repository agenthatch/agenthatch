"""LLMClient — Unified LLM call interface (agenthatch-core).

Supports OpenAI-compatible APIs. Core version accepts provider details directly
rather than resolving from agenthatch config.

Usage:
    client = LLMClient(provider="openai", model="gpt-4o", api_key="sk-...")
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    result = client.chat_structured(messages=msgs, response_model=MyPydanticModel)
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field as dc_field
from typing import Any

from agenthatch_core.exceptions import ApiKeyError, ProviderCapabilityError
from agenthatch_core.llm.types import StreamDelta, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ProviderFeatures:
    """Capability flags for a provider's API surface."""
    supports_tools: bool = True
    supports_stream_tools: bool = True
    supports_json_mode: bool = True
    supports_parallel_tool_calls: bool = True
    supports_reasoning_content: bool = False
    requires_anthropic_adapter: bool = False
    available_models: list[str] = dc_field(default_factory=list)


@dataclass
class ToolCallResponse:
    """Response from chat_with_tools."""
    text: str | None
    tool_calls: list[ToolCall] = dc_field(default_factory=list)
    finish_reason: str | None = None
    reasoning_content: str | None = None  # v0.9.8: DeepSeek thinking mode

    @classmethod
    def from_openai(cls, response: Any, llm_client: Any | None = None) -> ToolCallResponse:
        """Construct ToolCallResponse from OpenAI ChatCompletion."""
        msg = response.choices[0].message

        if llm_client and hasattr(llm_client, '_extract_content'):
            text = llm_client._extract_content(msg) or None
        else:
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
        return cls(
            text=text,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            reasoning_content=getattr(msg, "reasoning_content", None),
        )


class LLMClient:
    """Unified LLM call interface.

    Core version: accepts provider details directly.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        features: ProviderFeatures | None = None,
        context_window: int | None = None,
        thinking: bool = True,              # v0.8: deep thinking ON by default
        reasoning_effort: str = "medium",    # v0.8: OpenAI o-series
        effort: str | None = None,          # v0.10: Anthropic effort (low/medium/high/max/xhigh)
        **kwargs: Any,
    ):
        if not api_key:
            raise ApiKeyError(f"No API key provided for provider '{provider}'")

        self._provider_name = provider
        self._model = model
        self._features = features or ProviderFeatures()
        self._max_tokens = max_tokens or 4096
        self._context_window = context_window or 128000
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort
        self._effort = effort
        self.last_usage: Any = None

        if self._features.requires_anthropic_adapter:
            from agenthatch_core.llm.anthropic_adapter import AnthropicAdapter

            self._client = AnthropicAdapter(
                api_key=api_key,
                base_url=base_url or "https://api.anthropic.com",
            )
        else:
            import openai

            if not base_url and provider != "openai":
                logger.warning(
                    "No base_url configured for provider '%s'; "
                    "falling back to OpenAI endpoint. This is almost "
                    "certainly wrong — set base_url in your config or "
                    "pass it explicitly to LLMClient.",
                    provider,
                )

            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
                timeout=kwargs.get("timeout", 120.0),
            )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def features(self) -> ProviderFeatures:
        return self._features

    @property
    def context_window(self) -> int:
        return self._context_window

    # ── v0.8: Deep thinking configuration ─────────────────────────────

    def _build_thinking_body(self) -> dict | None:
        """Build extra_body for deep thinking.

        Provider-specific thinking configuration:
          DeepSeek:  {"thinking": {"type": "enabled"}}
          OpenAI:    {"reasoning_effort": "medium"}   (o-series / GPT-5.5)
          Anthropic: {"thinking": {"type": "adaptive"}} (Opus 4.6+, Sonnet 4.6+)
                     budget_tokens is DEPRECATED on 4.6+ and REMOVED on 4.7/4.8.
                     Use output_config.effort to control thinking depth.
          Others:    None (passthrough)
        """
        if not self._thinking:
            return None
        provider = self._provider_name.lower()
        if provider == "deepseek":
            return {"thinking": {"type": "enabled"}}
        elif provider == "openai":
            return {"reasoning_effort": self._reasoning_effort}
        elif provider == "anthropic":
            body: dict[str, Any] = {"thinking": {"type": "adaptive"}}
            if self._effort:
                body["effort"] = self._effort
            return body
        return None

    def _effective_tool_choice(
        self,
        tool_choice: str = "auto",
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any] | None:
        """Resolve effective tool_choice value respecting provider capabilities.

        Returns:
            - None if tools is empty or None (skip tool_choice param)
            - "auto" for standard tool calling
            - "required" to force a tool call (when explicitly requested)
            - dict like {"type": "function", "function": {"name": "x"}} for
              forced specific tool (when tool_choice names a specific function)

        Some providers (e.g. Ollama) don't support tool_choice at all.
        Some models need "required" to reliably call tools.
        """
        if not tools:
            return None

        if not self._features.supports_tools:
            return None

        if tool_choice == "auto":
            return "auto"

        if tool_choice == "required":
            return "required"

        if tool_choice == "none":
            return "none"

        # Named tool — check it exists in the tools list
        tool_names = {
            t.get("function", {}).get("name", "")
            for t in tools
            if isinstance(t, dict)
        }
        if tool_choice in tool_names:
            return {"type": "function", "function": {"name": tool_choice}}

        return "auto"

    # ── Retry ──────────────────────────────────────────────────────────

    def _retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        retryable_statuses: set[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call fn with exponential backoff on transient errors.

        Retries on HTTP errors (429, 500, 502, 503, 504) AND on
        timeout/connection errors (httpx.ReadTimeout, ConnectError,
        openai.APITimeoutError, etc.) which have no status_code.
        """
        if retryable_statuses is None:
            retryable_statuses = {429, 500, 502, 503, 504}

        # Exception types that indicate transient network issues
        _TIMEOUT_TYPES: tuple[type[Exception], ...] = ()
        try:
            import httpx
            _TIMEOUT_TYPES = (
                httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            )
        except ImportError:
            pass

        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                status = (
                    getattr(e, "status_code", None)
                    or getattr(getattr(e, "response", None), "status_code", None)
                    or getattr(e, "code", None)
                )

                # C4 fix: also retry on timeout/connection errors
                # which have no HTTP status code
                is_retryable = (
                    status in retryable_statuses
                    or isinstance(e, _TIMEOUT_TYPES)
                    or "timeout" in str(type(e).__name__).lower()
                )

                if not is_retryable:
                    raise
                if attempt == max_retries:
                    raise
                delay = min(1.0 * (2 ** attempt), 30.0)
                delay *= random.uniform(0.75, 1.25)
                logger.warning(
                    "LLM retry %d/%d after %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
                time.sleep(delay)

    # ── Simple chat ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple chat completion. Returns response text."""
        response = self._retry(
            self._client.chat.completions.create,
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.last_usage = getattr(response, "usage", None)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "Response truncated by max_tokens limit."
            )
        return self._extract_content(choice.message)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, str]:
        """Streaming chat completion without tools.

        Returns a generator yielding text deltas, with final response as
        the StopIteration value.  Delegates to _stream_native with no tools
        rather than using empty tool lists (which can trigger tool_choice
        anomalies in some models).
        """
        stream = self._retry(
            self._client.chat.completions.create,
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        for event in stream:
            delta = event.choices[0].delta if event.choices else None  # type: ignore[union-attr]
            if delta is None:
                if hasattr(event, "usage") and event.usage:
                    self.last_usage = event.usage
                continue

            if delta.content:
                text_parts.append(delta.content)
                yield delta.content

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
                # v0.7.11: Emit as ThinkingDelta event, NOT as visible content
                yield ThinkingDelta(content=delta.reasoning_content)

        text = "".join(text_parts)
        if not text and reasoning_parts:
            text = "".join(reasoning_parts)

        return text

    # ── Structured output (Instructor pattern) ───────────────────────

    def _chat_structured_raw(
        self,
        messages: list[dict[str, Any]],
        response_model: type,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Any:
        """Raw JSON parsing fallback for structured output.

        Used when instructor.from_openai() is unavailable (e.g. with
        AnthropicAdapter for custom providers).  Calls the chat API
        directly and parses the JSON response.
        """
        response = self._retry(
            self._client.chat.completions.create,
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.last_usage = getattr(response, "usage", None)
        msg = response.choices[0].message
        content = self._extract_content(msg)
        if not content:
            raise ValueError(
                "chat_structured: model returned empty content"
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            from agenthatch_core.context.manager import _extract_balanced_json
            json_strs = _extract_balanced_json(content)
            if json_strs:
                parsed = json.loads(json_strs[0])
            else:
                raise ValueError(
                    "chat_structured: no valid JSON found in response"
                )
        return response_model.model_validate(parsed)

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 2,
        thinking: bool | None = None,  # v0.8: per-call thinking override
    ) -> Any:
        """v0.8: Structured output via Instructor with optional deep thinking.

        Passes extra_body for thinking when enabled. Falls back gracefully
        if Instructor + thinking are incompatible.

        v0.8.19: When requires_anthropic_adapter is True, skip instructor
        entirely — AnthropicAdapter is not an openai.OpenAI instance and
        instructor.from_openai() will fail with a warning + no valid JSON.
        """
        extra = self._build_thinking_body() if (
            thinking if thinking is not None else self._thinking
        ) else None

        # v0.8.19: Skip instructor for AnthropicAdapter — it is not an
        # OpenAI client and instructor.from_openai() will fail.
        if self._features.requires_anthropic_adapter:
            return self._chat_structured_raw(
                messages=messages,
                response_model=response_model,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        import instructor

        client = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
        call_kwargs: dict[str, Any] = dict(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            response_model=response_model,
            max_retries=max_retries,
        )
        if extra:
            call_kwargs["extra_body"] = extra

        try:
            return client.chat.completions.create(**call_kwargs)
        except Exception as e:
            logger.debug("chat_structured Instructor call failed: %s", e)
            # v0.8: Try without thinking
            if extra:
                call_kwargs.pop("extra_body", None)
                try:
                    return client.chat.completions.create(**call_kwargs)
                except Exception:
                    pass
            # Fallback: raw JSON parsing (pre-v0.8 behavior)
            return self._chat_structured_raw(
                messages=messages,
                response_model=response_model,
                model=model,
                temperature=0.0,
                max_tokens=4096,
            )

    # ── Tool Calling ─────────────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> ToolCallResponse:
        """Chat completion with tool calling."""
        if not self._features.supports_tools:
            text = self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return ToolCallResponse(text=text, tool_calls=[])

        response = self._retry(
            self._client.chat.completions.create,  # type: ignore[call-overload]
            model=model or self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.last_usage = getattr(response, "usage", None)
        result = ToolCallResponse.from_openai(response, llm_client=self)
        msg = response.choices[0].message
        extracted = self._extract_content(msg)
        if extracted:
            result.text = extracted
        return result

    _SYNTHETIC_CHUNK_SIZE = 4

    def stream_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """Streaming chat completion with tool calling."""
        if not self._features.supports_stream_tools:
            return self._stream_synthetic_fallback(messages, tools, model, temperature, max_tokens, tool_choice)

        try:
            return self._stream_native(messages, tools, model, temperature, max_tokens, tool_choice)
        except Exception as e:
            if self._is_stream_tools_error(e):
                logger.warning(
                    "Provider '%s' failed on stream+tools, falling back to synthetic: %s",
                    self._provider_name, e,
                )
                self._features.supports_stream_tools = False
                return self._stream_synthetic_fallback(
                    messages, tools, model, temperature, max_tokens, tool_choice
                )
            raise

    def _stream_native(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int,
        tool_choice: str = "auto",
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """Native streaming with tool calling."""
        stream = self._retry(
            self._client.chat.completions.create,
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_accum: dict[int, dict[str, str]] = {}

        for event in stream:
            delta = event.choices[0].delta if event.choices else None  # type: ignore[union-attr]
            if delta is None:
                if hasattr(event, "usage") and event.usage:
                    self.last_usage = event.usage
                continue

            if delta.content:
                text_parts.append(delta.content)
                yield StreamDelta(type="text", content=delta.content)

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
                yield StreamDelta(
                    type="reasoning", content=delta.reasoning_content
                )

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
                        if not acc["name"]:
                            acc["name"] = tc_delta.function.name
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

        text = "".join(text_parts)
        if not text and reasoning_parts:
            text = "".join(reasoning_parts)
        reasoning = "".join(reasoning_parts) or None

        return ToolCallResponse(
            text=text or None,
            tool_calls=final_tool_calls,
            reasoning_content=reasoning,
        )

    def _stream_synthetic_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int,
        tool_choice: str = "auto",
    ) -> Generator[StreamDelta, None, ToolCallResponse]:
        """Synthetic streaming fallback."""
        response = self.chat_with_tools(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )

        if response.text:
            for i in range(0, len(response.text), self._SYNTHETIC_CHUNK_SIZE):
                chunk = response.text[i: i + self._SYNTHETIC_CHUNK_SIZE]
                yield StreamDelta(type="text", content=chunk)

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
        """Detect stream+tools incompatibility errors."""
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

    def _extract_content(self, message: Any) -> str:
        """Extract text content from LLM response, handling reasoning models."""
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)

        if self._features.supports_reasoning_content:
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                return reasoning

        # Always fallback to reasoning_content for reasoning models
        # even when the feature flag isn't explicitly set
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        return ""