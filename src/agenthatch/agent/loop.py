"""ConversationLoop — LLM <-> Tool calling cycle (v0.4)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Any

from agenthatch.base.sandbox import Sandbox
from agenthatch.cap.bus import CapBus
from agenthatch.skill.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RichToolCallEvent:
    """Rich-renderable tool call event for TUI consumption."""

    def __init__(
        self,
        phase: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        elapsed: float | None = None,
        result_preview: str | None = None,
    ):
        self.phase = phase
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.elapsed = elapsed
        self.result_preview = result_preview


class ConversationLoop:
    """Drives the User -> LLM -> Tool -> LLM -> Response cycle."""

    MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        llm: LLMClient,
        capbus: CapBus,
        sandbox: Sandbox,
        ctx: Any,
    ):
        self.llm = llm
        self.capbus = capbus
        self.sandbox = sandbox
        self.ctx = ctx

    def run(self, user_input: str) -> str:
        """Execute one conversation turn synchronously."""
        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        response = self.llm.chat_with_tools(
            messages=messages,
            tools=tools,
        )

        for _ in range(self.MAX_TOOL_ROUNDS):
            if not response.has_tool_calls:
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                result = self.capbus.route(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            response = self.llm.chat_with_tools(messages, tools)

        self.ctx.add_to_history("user", user_input)
        if response.text:
            self.ctx.add_to_history("assistant", response.text)

        return response.text or "(no response)"

    def stream(
        self, user_input: str
    ) -> Generator[RichToolCallEvent | str, None, str]:
        """Streaming conversation for TUI Live rendering."""
        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        full_response_text: str = ""

        for _ in range(self.MAX_TOOL_ROUNDS):
            accumulated_text = ""
            has_yielded_tool_header = False

            gen = self.llm.stream_chat_with_tools(
                messages=messages,
                tools=tools,
            )
            response = None
            while True:
                try:
                    delta = next(gen)
                except StopIteration as e:
                    response = e.value
                    break

                if delta.type == "text":
                    accumulated_text += delta.content
                    yield delta.content

                elif delta.type == "tool_call_start" and not has_yielded_tool_header:
                    has_yielded_tool_header = True
                    yield RichToolCallEvent(
                        phase="start",
                        tool_name=delta.tool_name or "unknown",
                    )

            if response is None:
                break

            if not response.has_tool_calls:
                full_response_text = response.text or accumulated_text
                break

            for tc in response.tool_calls:
                t0 = time.time()
                result = self.capbus.route(tc.name, tc.arguments)
                elapsed = time.time() - t0

                yield RichToolCallEvent(
                    phase="done",
                    tool_name=tc.name,
                    elapsed=elapsed,
                    result_preview=result[:200],
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })

        self.ctx.add_to_history("user", user_input)
        final_text = full_response_text or accumulated_text
        if final_text:
            self.ctx.add_to_history("assistant", final_text)

        return final_text or "(no response)"
