"""Regression: thinking-mode reasoning must survive into history (v1.0.19).

DeepSeek in thinking mode answers 400 — "The `reasoning_content` in the
thinking mode must be passed back to the API" — when a follow-up request
replays an earlier assistant turn without the reasoning it was returned
with.  The v0.9.8 fix preserved it only inside the live ``messages`` list,
so every history rebuild dropped it and the tool loop started failing
from the second user turn onward.

These tests pin the replay path: recorded reasoning, rebuilt messages.
"""

from __future__ import annotations

import pytest
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.llm.client import ToolCallResponse
from agenthatch_core.loop.agent_loop import ConversationLoop

from agenthatch.skill.spec import (
    AHSSpec,
    BaseSpec,
    Identity,
    Instructions,
    Intent,
    Interface,
)


@pytest.fixture
def spec() -> AHSSpec:
    return AHSSpec(
        identity=Identity(id="test-skill", display_name="Test Skill", version="1.0.0"),
        intent=Intent(
            triggers=["test"],
            satisfies=["do {thing}"],
            summary="A test skill.",
        ),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def ctx(spec: AHSSpec) -> ContextManager:
    return ContextManager(spec)


class _LoopStub:
    """Stand-in exposing only what ConversationLoop._record_assistant uses."""

    def __init__(self, ctx: ContextManager) -> None:
        self.ctx = ctx


def _response(reasoning: str | None) -> ToolCallResponse:
    return ToolCallResponse(text="answer", tool_calls=[], reasoning_content=reasoning)


class TestHistoryStoresReasoning:
    def test_reasoning_content_reaches_history(self, ctx: ContextManager) -> None:
        ctx.add_to_history(
            "assistant", "the answer", reasoning_content="step by step"
        )

        assert ctx.history[-1]["reasoning_content"] == "step by step"

    def test_history_is_replayed_by_build_messages(self, ctx: ContextManager) -> None:
        """The rebuilt request must carry the reasoning the API asked for."""
        ctx.add_to_history(
            "assistant", None,
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "noop", "arguments": "{}"}}],
            reasoning_content="thought about it",
        )
        ctx.add_to_history("tool", "ok", tool_call_id="c1")

        messages = ctx.build_messages("next question")

        assistant = [m for m in messages if m.get("role") == "assistant"]
        assert assistant, "assistant turn should be replayed"
        assert assistant[-1]["reasoning_content"] == "thought about it"

    def test_absent_reasoning_adds_no_key(self, ctx: ContextManager) -> None:
        """Nothing to replay → no `reasoning_content: null` on the wire."""
        ctx.add_to_history("assistant", "plain answer")

        assert "reasoning_content" not in ctx.history[-1]


class TestRecordAssistantFunnel:
    """Every assistant recording site goes through this helper."""

    def test_text_turn_records_reasoning(
        self, ctx: ContextManager
    ) -> None:
        stub = _LoopStub(ctx)

        msg = ConversationLoop._record_assistant(stub, _response("why"), "answer")

        assert msg["reasoning_content"] == "why"
        assert ctx.history[-1]["reasoning_content"] == "why"

    def test_tool_call_turn_records_reasoning(
        self, ctx: ContextManager
    ) -> None:
        stub = _LoopStub(ctx)
        tool_calls = [{
            "id": "c1", "type": "function",
            "function": {"name": "noop", "arguments": "{}"},
        }]

        msg = ConversationLoop._record_assistant(
            stub, _response("deliberation"), None, tool_calls=tool_calls
        )

        assert msg["tool_calls"] == tool_calls
        assert msg["reasoning_content"] == "deliberation"
        assert ctx.history[-1]["reasoning_content"] == "deliberation"
        assert ctx.history[-1]["tool_calls"] == tool_calls

    def test_missing_reasoning_is_omitted(
        self, ctx: ContextManager
    ) -> None:
        """A non-thinking provider must not gain the field."""
        stub = _LoopStub(ctx)

        msg = ConversationLoop._record_assistant(stub, _response(None), "answer")

        assert "reasoning_content" not in msg
        assert "reasoning_content" not in ctx.history[-1]
