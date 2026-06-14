"""Tests for ConversationLoop — LLM/Tool calling cycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.llm.client import LLMClient, ToolCallResponse
from agenthatch_core.llm.types import StreamDelta, ToolCall
from agenthatch_core.loop.agent_loop import ConversationLoop, RichToolCallEvent

from agenthatch.base.sandbox import Sandbox
from agenthatch.cap.bus import CapBus
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
        identity=Identity(id="loop-test", display_name="Loop Test", version="1.0.0"),
        intent=Intent(triggers=["test"], satisfies=["test"], summary="Test"),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.provider_name = "mock"
    llm.model = "mock-model"
    return llm


@pytest.fixture
def capbus():
    return CapBus()


@pytest.fixture
def sandbox():
    return Sandbox()


@pytest.fixture
def ctx(spec):
    return ContextManager(spec)


@pytest.fixture
def loop(mock_llm, capbus, sandbox, ctx):
    return ConversationLoop(llm=mock_llm, capbus=capbus, sandbox=sandbox, ctx=ctx)


class TestConversationLoopInit:
    def test_init_stores_dependencies(self, mock_llm, capbus, sandbox, ctx):
        loop = ConversationLoop(llm=mock_llm, capbus=capbus, sandbox=sandbox, ctx=ctx)
        assert loop.llm is mock_llm
        assert loop.capbus is capbus
        assert loop.sandbox is sandbox
        assert loop.ctx is ctx


class TestRunSync:
    def test_simple_text_response(self, loop, mock_llm):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text="Hello, world!",
            tool_calls=[],
        )
        result = loop.run("hi")
        assert result == "Hello, world!"

    def test_adds_to_history_after_run(self, loop, mock_llm, ctx):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text="Response",
            tool_calls=[],
        )
        loop.run("question")
        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "question"}
        assert ctx.history[1] == {"role": "assistant", "content": "Response"}

    def test_tool_call_route_and_response(self, loop, mock_llm, capbus):
        capbus.register(
            name="echo",
            cap_type="test",
            schema={"type": "object", "properties": {}},
            source_skill="test",
            executor=MagicMock(execute=lambda **kw: "echoed: " + kw.get("msg", "")),
        )

        # v0.6 auto-continuation: after tool + text, loop calls LLM once more per round.
        # v0.8.14: while True loop needs unlimited fallback responses for budget.
        _call_count = [0]
        def _mock_responses(*args, **kwargs):
            _call_count[0] += 1
            if _call_count[0] == 1:
                return ToolCallResponse(
                    text=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hello"})],
                )
            return ToolCallResponse(text="I echoed your message.", tool_calls=[])
        mock_llm.chat_with_tools.side_effect = _mock_responses

        result = loop.run("echo hello")
        assert "I echoed your message" in result

    def test_empty_response_returns_placeholder(self, loop, mock_llm):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(text=None, tool_calls=[])
        result = loop.run("hi")
        assert result == "(no response)"

    def test_multiple_tool_calls_in_one_response(self, loop, mock_llm, capbus):
        capbus.register(
            name="tool_a",
            cap_type="test",
            schema={},
            source_skill="test",
            executor=MagicMock(execute=lambda **kw: "result_a"),
        )
        capbus.register(
            name="tool_b",
            cap_type="test",
            schema={},
            source_skill="test",
            executor=MagicMock(execute=lambda **kw: "result_b"),
        )

        # v0.6 auto-continuation: after tools + text, loop continues until budget exhausted
        # v0.9: _MAX_CONSECUTIVE_TEXT_ONLY = 13, need >= 14 items to avoid StopIteration
        mock_llm.chat_with_tools.side_effect = [
            ToolCallResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="tool_a", arguments={}),
                    ToolCall(id="c2", name="tool_b", arguments={}),
                ],
            ),
            ToolCallResponse(text="Done.", tool_calls=[]),
        ] + [ToolCallResponse(text="Done.", tool_calls=[])] * 14

        result = loop.run("both")
        assert "Done." in result


class TestRunStream:
    def test_stream_text_only(self, loop, mock_llm):
        deltas = [
            StreamDelta(type="text", content="Hello"),
            StreamDelta(type="text", content=" world"),
        ]
        response = ToolCallResponse(text="Hello world", tool_calls=[])

        def gen():
            yield from deltas
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()

        chunks = list(loop.stream("hi"))
        assert chunks == ["Hello", " world"]

    def test_stream_with_tool_call(self, loop, mock_llm, capbus):
        capbus.register(
            name="fetch",
            cap_type="test",
            schema={},
            source_skill="test",
            executor=MagicMock(execute=lambda **kw: "data from fetch tool"),
        )

        deltas = [StreamDelta(type="tool_call_start", tool_name="fetch")]
        tool_response = ToolCallResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="fetch", arguments={"url": "x"})],
        )
        final_response = ToolCallResponse(text="Fetched data.", tool_calls=[])

        gen1 = (d for d in deltas)

        def make_gen1():
            yield from gen1
            return tool_response

        def make_gen2():
            yield StreamDelta(type="text", content="Fetched data.")
            return final_response

        mock_llm.stream_chat_with_tools.side_effect = [make_gen1(), make_gen2()]

        chunks = list(loop.stream("fetch data"))
        texts = [c for c in chunks if isinstance(c, str)]
        events = [c for c in chunks if isinstance(c, RichToolCallEvent)]

        assert len(events) >= 2
        assert events[0].phase == "start"
        assert events[0].tool_name == "fetch"
        assert "Fetched data" in "".join(texts)

    def test_stream_adds_history(self, loop, mock_llm, ctx):
        deltas = [StreamDelta(type="text", content="Streamed reply")]
        response = ToolCallResponse(text="Streamed reply", tool_calls=[])

        def gen():
            yield from deltas
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()

        list(loop.stream("question"))
        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "question"}
        assert ctx.history[1] == {"role": "assistant", "content": "Streamed reply"}


class TestRichToolCallEvent:
    def test_create_event(self):
        event = RichToolCallEvent(
            phase="start",
            tool_name="test_tool",
            tool_args={"key": "value"},
            elapsed=1.5,
            result_preview="preview text",
        )
        assert event.phase == "start"
        assert event.tool_name == "test_tool"
        assert event.tool_args == {"key": "value"}
        assert event.elapsed == 1.5
        assert event.result_preview == "preview text"

    def test_defaults(self):
        event = RichToolCallEvent(phase="done", tool_name="x")
        assert event.tool_args is None
        assert event.elapsed is None
        assert event.result_preview is None


class TestDirectLoopC2:
    """C2 fix: DirectLoop must use add_to_history, not add_assistant_message."""

    def test_run_calls_add_to_history(self):
        """DirectLoop.run() must call ctx.add_to_history('assistant', ...)."""
        from unittest.mock import MagicMock

        from agenthatch_core.bricks.loops import DirectLoop

        mock_llm = MagicMock()
        mock_llm.chat.return_value = "test response"
        mock_ctx = MagicMock()

        loop = DirectLoop(mock_llm, mock_ctx)
        _ = loop.run("hello")

        # C2 fix: must use add_to_history, not add_assistant_message
        mock_ctx.add_to_history.assert_called_with("assistant", "test response")
        # Old broken call must NOT be made
        assert not hasattr(mock_ctx.add_assistant_message, 'called') or \
            not mock_ctx.add_assistant_message.called, (
            "C2 regression: DirectLoop still calls add_assistant_message "
            "which does not exist on ContextManager"
        )

    def test_stream_calls_add_to_history(self):
        """DirectLoop.stream() must call ctx.add_to_history('assistant', ...)."""
        from unittest.mock import MagicMock

        from agenthatch_core.bricks.loops import DirectLoop

        mock_llm = MagicMock()

        mock_llm.chat_stream.return_value = iter(["Hello", " world"])
        mock_ctx = MagicMock()

        loop = DirectLoop(mock_llm, mock_ctx)
        chunks = list(loop.stream("hello"))
        result_text = "".join(chunks)

        assert result_text == "Hello world"
        mock_ctx.add_to_history.assert_called_with("assistant", "Hello world")
