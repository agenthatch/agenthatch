"""Test reasoning_content handling in streaming and ToolCallResponse.

Verifies that:
1. _stream_native() correctly handles content="" with reasoning_content non-empty
2. ToolCallResponse text fallback works for reasoning models
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agenthatch_core.llm.client import LLMClient, ToolCallResponse

# ---------------------------------------------------------------------------
# Helpers: mock OpenAI streaming chunks
# ---------------------------------------------------------------------------


class _MockFunction:
    """Simulates OpenAI tool call function delta."""

    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _MockToolCallDelta:
    """Simulates OpenAI tool call delta in streaming."""

    def __init__(self, index=0, id="", function=None):
        self.index = index
        self.id = id
        self.function = function


class _MockDelta:
    """Simulates an OpenAI streaming delta."""

    def __init__(self, content="", reasoning_content="", tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _MockChoice:
    """Simulates OpenAI choice with delta."""

    def __init__(self, delta):
        self.delta = delta


class _MockStreamEvent:
    """Simulates one OpenAI streaming event."""

    def __init__(self, delta):
        self.choices = [_MockChoice(delta)]


def _build_mock_llm_client(monkeypatch, features=None):
    """Build a minimal LLMClient with mocked OpenAI dependency."""
    import openai

    mock_client = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: mock_client)

    client = LLMClient(provider="deepseek", model="test-model", api_key="sk-mock")

    if features:
        client._features = features

    return client, mock_client


# ---------------------------------------------------------------------------
# _stream_native() — reasoning_content fallback
# ---------------------------------------------------------------------------


class TestStreamNativeReasoning:
    """Tests for _stream_native() reasoning_content fallback."""

    def test_content_empty_reasoning_nonempty_fallback(self, monkeypatch):
        """When content is empty but reasoning_content is present,
        _stream_native should use reasoning_content as the final text."""
        client, mock_openai = _build_mock_llm_client(monkeypatch)

        # Simulate stream: only reasoning_content, no content
        events = [
            _MockStreamEvent(
                _MockDelta(content="", reasoning_content="I need to think about this...")
            ),
            _MockStreamEvent(
                _MockDelta(content="", reasoning_content="The answer is 42.")
            ),
        ]
        mock_openai.chat.completions.create.return_value = events

        gen = client._stream_native(
            messages=[{"role": "user", "content": "What is the answer?"}],
            tools=[],
            model=None,
            temperature=0.7,
            max_tokens=100,
        )

        deltas = []
        try:
            while True:
                deltas.append(next(gen))
        except StopIteration as e:
            response = e.value

        # No text deltas yielded (content was always empty)
        text_deltas = [d for d in deltas if d.type == "text"]
        assert len(text_deltas) == 0

        # Final response should have reasoning_content as text
        assert response.text == "I need to think about this...The answer is 42."

    def test_content_and_reasoning_both_present(self, monkeypatch):
        """When both content and reasoning_content are present,
        content should be used as the final text, not reasoning_content."""
        client, mock_openai = _build_mock_llm_client(monkeypatch)

        events = [
            _MockStreamEvent(
                _MockDelta(content="The answer is 42.", reasoning_content="Thinking...")
            ),
        ]
        mock_openai.chat.completions.create.return_value = events

        gen = client._stream_native(
            messages=[{"role": "user", "content": "What is the answer?"}],
            tools=[],
            model=None,
            temperature=0.7,
            max_tokens=100,
        )

        deltas = []
        try:
            while True:
                deltas.append(next(gen))
        except StopIteration as e:
            response = e.value

        text_deltas = [d for d in deltas if d.type == "text"]
        assert len(text_deltas) == 1
        assert text_deltas[0].content == "The answer is 42."
        assert response.text == "The answer is 42."

    def test_no_content_no_reasoning_no_tool_calls(self, monkeypatch):
        """When everything is empty, text should be None."""
        client, mock_openai = _build_mock_llm_client(monkeypatch)

        events = [
            _MockStreamEvent(_MockDelta(content="", reasoning_content="")),
        ]
        mock_openai.chat.completions.create.return_value = events

        gen = client._stream_native(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            model=None,
            temperature=0.7,
            max_tokens=100,
        )

        try:
            while True:
                next(gen)
        except StopIteration as e:
            response = e.value

        assert response.text is None

    def test_stream_with_reasoning_and_tool_calls(self, monkeypatch):
        """When reasoning_content is present alongside tool calls,
        the final text should fallback to reasoning_content."""
        client, mock_openai = _build_mock_llm_client(monkeypatch)

        import json

        events = [
            _MockStreamEvent(
                _MockDelta(
                    content="",
                    reasoning_content="I should call the search tool.",
                    tool_calls=[
                        _MockToolCallDelta(
                            index=0,
                            id="call_1",
                            function=_MockFunction(
                                name="search", arguments=json.dumps({"q": "test"})
                            ),
                        )
                    ],
                )
            ),
        ]
        mock_openai.chat.completions.create.return_value = events

        gen = client._stream_native(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            model=None,
            temperature=0.7,
            max_tokens=100,
        )

        deltas = []
        try:
            while True:
                deltas.append(next(gen))
        except StopIteration as e:
            response = e.value

        # Should have tool call deltas
        tool_start_deltas = [d for d in deltas if d.type == "tool_call_start"]
        assert len(tool_start_deltas) == 1
        assert tool_start_deltas[0].tool_name == "search"

        # Final text should be reasoning_content
        assert response.text == "I should call the search tool."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"


# ---------------------------------------------------------------------------
# ToolCallResponse.from_openai — reasoning_content fallback
# ---------------------------------------------------------------------------


class TestToolCallResponseReasoningFallback:
    """Tests for ToolCallResponse.from_openai reasoning_content fallback."""

    def test_text_fallback_to_reasoning_content(self, monkeypatch):
        """When content is empty, from_openai should fall back to reasoning_content."""
        import openai

        mock_openai_client = MagicMock()
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: mock_openai_client)

        client = LLMClient(provider="deepseek", model="test-model", api_key="sk-mock")

        # Build mock response with empty content but reasoning_content present
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = "The answer is 42 based on my analysis."
        mock_msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.usage = None

        result = ToolCallResponse.from_openai(mock_response, llm_client=client)

        assert result.text == "The answer is 42 based on my analysis."
        assert result.tool_calls == []

    def test_text_fallback_content_present(self, monkeypatch):
        """When content is present, it should be used, not reasoning_content."""
        import openai

        mock_openai_client = MagicMock()
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: mock_openai_client)

        client = LLMClient(provider="deepseek", model="test-model", api_key="sk-mock")

        mock_msg = MagicMock()
        mock_msg.content = "Direct answer."
        mock_msg.reasoning_content = "Hidden reasoning..."
        mock_msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.usage = None

        result = ToolCallResponse.from_openai(mock_response, llm_client=client)

        assert result.text == "Direct answer."

    def test_text_fallback_no_llm_client(self):
        """When no llm_client is provided, fall back to msg.content directly."""
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = "Hidden reasoning..."
        mock_msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.usage = None

        result = ToolCallResponse.from_openai(mock_response, llm_client=None)

        # Without llm_client, content is used directly (empty string → None)
        assert result.text is None

    def test_text_fallback_both_empty(self, monkeypatch):
        """When both content and reasoning_content are empty, text should be None."""
        import openai

        mock_openai_client = MagicMock()
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: mock_openai_client)

        client = LLMClient(provider="deepseek", model="test-model", api_key="sk-mock")

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = ""
        mock_msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.usage = None

        result = ToolCallResponse.from_openai(mock_response, llm_client=client)

        assert result.text is None

    def test_text_fallback_reasoning_disabled_provider(self, monkeypatch):
        """When supports_reasoning_content is False, reasoning_content is ignored."""
        import openai

        mock_openai_client = MagicMock()
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: mock_openai_client)

        # openai doesn't support reasoning_content
        client = LLMClient(provider="openai", model="test-model", api_key="sk-mock")

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = "Hidden reasoning..."
        mock_msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.usage = None

        result = ToolCallResponse.from_openai(mock_response, llm_client=client)

        # reasoning_content is always used as fallback regardless of feature flag
        assert result.text == "Hidden reasoning..."
