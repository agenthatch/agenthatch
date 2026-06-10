"""DirectLoop — single-turn loop engine for prompt-only skills.

Level 0 — simplest loop engine.  No tool calling, no multi-turn.
Just sends user input to the LLM and returns the response.
"""

from __future__ import annotations

from typing import Any


class DirectLoop:
    """Single-turn loop engine for prompt-only skills.

    No tool calling, no state management, no sandbox.
    The simplest possible agent loop.

    Usage:
        loop = DirectLoop(llm_client, context_manager)
        response = loop.run(user_input)
    """

    def __init__(
        self,
        llm: Any,
        ctx: Any,
        token_counter: Any = None,
    ):
        self._llm = llm
        self._ctx = ctx
        self._token_counter = token_counter

    def run(self, user_input: str) -> str:
        """Execute a single turn: build messages, call LLM, return result."""
        messages = self._ctx.build_messages(user_input)
        result = self._llm.chat(messages)
        self._ctx.add_to_history("assistant", result)
        self._record_usage()
        return result

    def stream(self, user_input: str):  # Generator[str, None, str]
        """Streaming single-turn execution."""
        messages = self._ctx.build_messages(user_input)
        text_parts: list[str] = []

        for chunk in self._llm.chat_stream(messages):
            text_parts.append(chunk)
            yield chunk

        full_text = "".join(text_parts)
        self._ctx.add_to_history("assistant", full_text)
        self._record_usage()
        return full_text

    def _record_usage(self) -> None:
        """Record token usage from last LLM call."""
        if self._token_counter is None:
            return
        usage = getattr(self._llm, "last_usage", None)
        if usage is None:
            return
        self._token_counter.add_usage({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        })