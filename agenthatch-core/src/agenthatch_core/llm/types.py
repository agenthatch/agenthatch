"""agenthatch-core LLM types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StreamDelta:
    """Streaming response delta."""
    type: Literal["text", "reasoning", "tool_call_start", "tool_call_args"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_index: int | None = None
    elapsed: float | None = None
    result_preview: str | None = None


@dataclass
class ToolCall:
    """A single tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]