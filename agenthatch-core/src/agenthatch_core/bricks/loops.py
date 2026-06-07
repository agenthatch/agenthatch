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
    ):
        self._llm = llm
        self._ctx = ctx

    def run(self, user_input: str) -> str:
        """Execute a single turn: build messages, call LLM, return result."""
        messages = self._ctx.build_messages(user_input)
        result = self._llm.chat(messages)
        self._ctx.add_assistant_message(result)
        return result

    def stream(self, user_input: str):  # Generator[str, None, str]
        """Streaming single-turn execution."""
        messages = self._ctx.build_messages(user_input)
        text_parts: list[str] = []

        for chunk in self._llm.chat_stream(messages):
            text_parts.append(chunk)
            yield chunk

        full_text = "".join(text_parts)
        self._ctx.add_assistant_message(full_text)
        return full_text