"""TokenCounter — token usage tracking for streaming and non-streaming.

Level 0 — tracks prompt_tokens, completion_tokens, total_tokens,
cache_read_tokens, cache_write_tokens, and reasoning_tokens.
Provides ThinkingDelta event for streaming reasoning content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThinkingDelta:
    """A chunk of reasoning/thinking content during streaming."""
    content: str
    index: int = 0


@dataclass
class TokenCounter:
    """Tracks token usage across a conversation.

    Six standard fields covering all major provider usage dimensions.
    Incremented as stream deltas arrive or from final usage response.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    call_count: int = 0
    elapsed_ms: int = 0

    def add_usage(self, usage: dict[str, int] | Any) -> None:
        """Merge usage info from LLM response."""
        self.call_count += 1
        if isinstance(usage, dict):
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
            self.cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
            self.reasoning_tokens += _safe_get_reasoning_tokens(usage)
            self.cached_tokens += usage.get("cached_tokens", 0)
        elif hasattr(usage, "prompt_tokens"):
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                self.cached_tokens += getattr(prompt_details, "cached_tokens", 0) or 0

    def snapshot(self) -> dict[str, int]:
        """Return current counters as a dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "call_count": self.call_count,
            "elapsed_ms": self.elapsed_ms,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.call_count = 0
        self.elapsed_ms = 0


def _safe_get_reasoning_tokens(usage: dict[str, Any]) -> int:
    """Extract reasoning_tokens from usage dict, handling CompletionTokensDetails object.

    v0.7.10: OpenAI returns CompletionTokensDetails as a Pydantic model,
    not a plain dict. This helper handles both cases.
    """
    # Direct reasoning_tokens field (some providers)
    direct = usage.get("reasoning_tokens", 0)
    if isinstance(direct, int) and direct > 0:
        return direct

    # Nested under completion_tokens_details (OpenAI)
    details = usage.get("completion_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        return details.get("reasoning_tokens", 0)
    # Object attribute access for Pydantic/OpenAI models
    return getattr(details, "reasoning_tokens", 0) or 0
