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
        memory_brick: Any = None,  # v0.7.12
        hooks: Any = None,          # v0.7.12
    ):
        self._llm = llm
        self._ctx = ctx
        self._token_counter = token_counter
        self._memory_brick = memory_brick
        self._hooks = hooks

    def run(self, user_input: str) -> str:
        """Execute a single turn: build messages, call LLM, return result."""
        messages = self._ctx.build_messages(user_input)
        result = self._llm.chat(messages)
        # v1.0.26: persist reasoning_content so a thinking-mode provider
        # (DeepSeek) does not 400 on the second turn.  chat() stores the
        # reasoning on the client; ConversationLoop already does this, but
        # DirectLoop (PROMPT_ONLY archetype) was missed in v1.0.19.
        reasoning = getattr(self._llm, "last_reasoning_content", None)
        self._ctx.add_to_history(
            "assistant", result, reasoning_content=reasoning
        )
        self._record_usage()
        return result

    def stream(self, user_input: str):  # Generator[str, None, str]
        """Streaming single-turn execution."""
        messages = self._ctx.build_messages(user_input)
        text_parts: list[str] = []

        for chunk in self._llm.chat_stream(messages):
            # chat_stream yields ThinkingDelta objects alongside text for
            # reasoning models; only accumulate the text for the history,
            # but still forward every chunk to the caller (TUI).
            if isinstance(chunk, str):
                text_parts.append(chunk)
            yield chunk

        full_text = "".join(text_parts)
        reasoning = getattr(self._llm, "last_reasoning_content", None)
        self._ctx.add_to_history(
            "assistant", full_text, reasoning_content=reasoning
        )
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
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
        })