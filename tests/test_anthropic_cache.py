"""Tests for Anthropic prompt caching (cache_control breakpoints).

Verifies that the adapter:
  1. Marks the last system block, last tool, and last message content
     block with cache_control: {"type": "ephemeral"}
  2. Never mutates the caller's message objects (they are reused across
     agent-loop turns)
  3. Folds cache_read/cache_creation tokens into prompt_tokens/total_tokens
     for both non-streaming and streaming calls
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agenthatch_core.llm.anthropic_adapter import (
    AnthropicAdapter,
    AnthropicChatCompletions,
    _apply_cache_control,
)

CC = {"type": "ephemeral"}


class _StubMessages:
    """Stub anthropic client .messages namespace capturing create() kwargs."""

    def __init__(self, response: Any):
        self._response = response
        self.captured: dict[str, Any] | None = None
        # AnthropicChatCompletions calls self._client.messages.create(...)
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        return self._response


def _fake_response(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test",
        model="claude-opus-4-8",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        ),
    )


def _make_completions(response: Any) -> tuple[AnthropicChatCompletions, _StubMessages]:
    stub = _StubMessages(response)
    adapter = object.__new__(AnthropicAdapter)
    adapter.chat = SimpleNamespace(completions=AnthropicChatCompletions(stub))
    return adapter.chat.completions, stub


class TestCacheControlMarkers:
    def test_string_system_gets_cached(self):
        completions, stub = _make_completions(_fake_response())
        completions.create(
            model="claude-opus-4-8",
            messages=[
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert stub.captured is not None
        system = stub.captured["system"]
        assert isinstance(system, list)
        assert system[-1]["cache_control"] == CC

    def test_block_system_last_block_marked(self):
        completions, stub = _make_completions(_fake_response())
        completions.create(
            model="claude-opus-4-8",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        )
        system = stub.captured["system"]
        assert len(system) == 2
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == CC

    def test_last_tool_marked_only(self):
        completions, stub = _make_completions(_fake_response())
        tools = [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
        ]
        completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        sent_tools = stub.captured["tools"]
        assert len(sent_tools) == 2
        assert "cache_control" not in sent_tools[0]
        assert sent_tools[1]["cache_control"] == CC

    def test_last_message_last_block_marked(self):
        completions, stub = _make_completions(_fake_response())
        completions.create(
            model="claude-opus-4-8",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                },
            ],
        )
        msgs = stub.captured["messages"]
        assert len(msgs) == 3
        # earlier messages unmarked
        assert "cache_control" not in msgs[0]["content"][0]
        assert "cache_control" not in msgs[1]["content"][0]
        # last message: first block unmarked, last block marked
        assert "cache_control" not in msgs[2]["content"][0]
        assert msgs[2]["content"][1]["cache_control"] == CC

    def test_no_messages_no_crash(self):
        system, messages, tools = _apply_cache_control([], [], None)
        assert system == []
        assert messages == []
        assert tools is None

    def test_non_dict_last_content_block_skipped(self):
        """String items pass through the translation layer verbatim; the
        marker must skip them instead of raising TypeError."""
        completions, stub = _make_completions(_fake_response())
        # user content list with a bare string as the last item
        completions.create(
            model="claude-opus-4-8",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": ["hello"]},
            ],
        )
        msgs = stub.captured["messages"]
        # no crash, last block unmarked, content preserved
        assert msgs[-1]["content"] == ["hello"]
        assert "cache_control" not in msgs[-1]["content"][0]
        # system (a plain string) is still marked
        assert stub.captured["system"][-1]["cache_control"] == CC

    def test_non_dict_last_system_block_skipped(self):
        """System list ending in a non-dict item must not crash the marker."""
        completions, stub = _make_completions(_fake_response())
        completions.create(
            model="claude-opus-4-8",
            messages=[
                # dict block first, bare string last — translation extends
                # system_parts verbatim for list content
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "part one"}, "tail"],
                },
                {"role": "user", "content": "Hi"},
            ],
        )
        system = stub.captured["system"]
        # no crash; dict block copied but unmarked (only the last block
        # would be marked, and it isn't a dict)
        assert system[0] == {"type": "text", "text": "part one"}
        assert system[-1] == "tail"
        assert "cache_control" not in system[0]

    def test_input_messages_not_mutated(self):
        completions, stub = _make_completions(_fake_response())
        system_blocks = [{"type": "text", "text": "sys"}]
        user_blocks = [{"type": "text", "text": "hi"}]
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": user_blocks},
        ]
        tools = [{"type": "function", "function": {"name": "a", "parameters": {}}}]
        completions.create(
            model="claude-opus-4-8",
            messages=messages,
            tools=tools,
        )
        # caller-side objects untouched
        assert "cache_control" not in system_blocks[0]
        assert "cache_control" not in user_blocks[0]
        assert "cache_control" not in tools[0]["function"]
        assert messages[0]["content"] is system_blocks
        assert messages[1]["content"] is user_blocks


class TestCacheUsageAccounting:
    def test_sync_usage_includes_cached_tokens(self):
        completions, _ = _make_completions(
            _fake_response(
                input_tokens=10,
                output_tokens=5,
                cache_read=100,
                cache_creation=50,
            )
        )
        resp = completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.usage.prompt_tokens == 10 + 100 + 50
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 10 + 100 + 50 + 5
        assert resp.usage.cache_read_input_tokens == 100
        assert resp.usage.cache_creation_input_tokens == 50

    def test_sync_usage_without_cache(self):
        completions, _ = _make_completions(
            _fake_response(input_tokens=10, output_tokens=5)
        )
        resp = completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.total_tokens == 15
        assert resp.usage.cache_read_input_tokens == 0

    def test_stream_usage_merges_input_side(self):
        """message_start input tokens (incl. cache) merge into message_delta usage."""

        class _StreamCtx:
            def __init__(self, events: list[Any]):
                self._events = events

            def __enter__(self):
                return iter(self._events)

            def __exit__(self, *args: Any) -> None:
                pass

        class _StubStreamMessages:
            def __init__(self, events: list[Any]):
                self._events = events
                self.captured: dict[str, Any] | None = None
                self.messages = self

            def stream(self, **kwargs: Any):
                self.captured = kwargs
                return _StreamCtx(self._events)

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=1,
                        cache_read_input_tokens=200,
                        cache_creation_input_tokens=0,
                    )
                ),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=0, output_tokens=7),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        stub = _StubStreamMessages(events)
        completions = AnthropicChatCompletions(stub)
        chunks = list(
            completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )
        usage_chunks = [c for c in chunks if c.usage]
        assert usage_chunks, "expected a usage-bearing chunk"
        final_usage = usage_chunks[-1].usage
        assert final_usage.prompt_tokens == 10 + 200
        assert final_usage.completion_tokens == 7
        assert final_usage.total_tokens == 10 + 200 + 7

    def test_stream_usage_propagates_cache_fields(self):
        """Streaming consumers (TokenCounter) must see cache fields like non-streaming."""

        class _StreamCtx:
            def __init__(self, events: list[Any]):
                self._events = events

            def __enter__(self):
                return iter(self._events)

            def __exit__(self, *args: Any) -> None:
                pass

        class _StubStreamMessages:
            def __init__(self, events: list[Any]):
                self._events = events
                self.captured: dict[str, Any] | None = None
                self.messages = self

            def stream(self, **kwargs: Any):
                self.captured = kwargs
                return _StreamCtx(self._events)

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=12,
                        output_tokens=1,
                        cache_read_input_tokens=300,
                        cache_creation_input_tokens=80,
                    )
                ),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=0, output_tokens=9),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        completions = AnthropicChatCompletions(_StubStreamMessages(events))
        chunks = list(
            completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )
        final_usage = [c for c in chunks if c.usage][-1].usage
        assert final_usage.cache_read_input_tokens == 300
        assert final_usage.cache_creation_input_tokens == 80
        assert final_usage.prompt_tokens == 12 + 300 + 80

    def test_stream_no_double_count_when_delta_reports_input(self):
        """Gateways that replay input_tokens in message_delta must not be double-counted."""

        class _StreamCtx:
            def __init__(self, events: list[Any]):
                self._events = events

            def __enter__(self):
                return iter(self._events)

            def __exit__(self, *args: Any) -> None:
                pass

        class _StubStreamMessages:
            def __init__(self, events: list[Any]):
                self._events = events
                self.captured: dict[str, Any] | None = None
                self.messages = self

            def stream(self, **kwargs: Any):
                self.captured = kwargs
                return _StreamCtx(self._events)

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=1,
                        cache_read_input_tokens=100,
                        cache_creation_input_tokens=0,
                    )
                ),
            ),
            # gateway replays full input-side usage here
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=110, output_tokens=7),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        completions = AnthropicChatCompletions(_StubStreamMessages(events))
        chunks = list(
            completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )
        final_usage = [c for c in chunks if c.usage][-1].usage
        # delta's own 110 stands; message_start's 110 not added on top
        assert final_usage.prompt_tokens == 110
        assert final_usage.total_tokens == 110 + 7
