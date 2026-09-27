"""Tests for Anthropic model-generation compatibility rules in the adapter.

Anthropic changed the Messages API surface across model generations:
  - Sampling params (temperature/top_p/top_k) were removed on Opus/Sonnet
    4.7+ and on every 5th-generation family (Fable, Mythos, Opus 5).
    Passing them returns a 400.
  - Fable/Mythos 5.1+ reject tool_choice "any"/"tool" with a 400.
  - v1.0.20: Opus 5.5 (2026-09-22) rejects them too, "as on Claude
    Fable 5.1" per the Opus 5.5 release notes.

These tests lock down that the adapter drops the offending params for the
new models and keeps sending them for the old ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agenthatch_core.llm.anthropic_adapter import (
    AnthropicAdapter,
    AnthropicChatCompletions,
    _forced_tool_choice_unsupported,
    _sampling_params_removed,
)


class _StubMessages:
    """Stub anthropic client .messages namespace capturing create() kwargs."""

    def __init__(self, response: Any):
        self._response = response
        self.captured: dict[str, Any] | None = None
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        return self._response


def _fake_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test",
        model="claude-test",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def _make_completions() -> tuple[AnthropicChatCompletions, _StubMessages]:
    stub = _StubMessages(_fake_response())
    adapter = object.__new__(AnthropicAdapter)
    adapter.chat = SimpleNamespace(completions=AnthropicChatCompletions(stub))
    return adapter.chat.completions, stub


class TestSamplingParamsRemoved:
    def test_old_models_keep_temperature(self):
        for model in ("claude-opus-4-5", "claude-opus-4-6", "claude-sonnet-4-6"):
            assert _sampling_params_removed(model) is False

    def test_47_and_48_drop_temperature(self):
        for model in ("claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-7"):
            assert _sampling_params_removed(model) is True

    def test_fifth_generation_families_drop_temperature(self):
        for model in ("claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1",
                      "claude-opus-5"):
            assert _sampling_params_removed(model) is True

    def test_snapshot_suffix_still_matched(self):
        assert _sampling_params_removed("claude-opus-4-8-20250714") is True
        assert _sampling_params_removed("claude-opus-4-6-20250514") is False

    def test_unknown_model_keeps_legacy_behavior(self):
        assert _sampling_params_removed("some-internal-model") is False


class TestTemperatureInRequests:
    def test_fable_51_request_has_no_temperature(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-fable-5-1",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert "temperature" not in stub.captured

    def test_opus_46_request_keeps_temperature(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert stub.captured["temperature"] == 0.7

    def test_opus_5_request_has_no_temperature(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert "temperature" not in stub.captured

    def test_opus_55_request_has_no_temperature(self):
        """v1.0.20: the major>=5 rule already covers Opus 5.5 — lock it."""
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert "temperature" not in stub.captured


class TestForcedToolChoice:
    def test_fable_mythos_51_unsupported(self):
        assert _forced_tool_choice_unsupported("claude-fable-5-1") is True
        assert _forced_tool_choice_unsupported("claude-mythos-5-1") is True

    def test_fable_5_and_older_still_supported(self):
        assert _forced_tool_choice_unsupported("claude-fable-5") is False
        assert _forced_tool_choice_unsupported("claude-opus-5") is False
        assert _forced_tool_choice_unsupported("claude-opus-4-8") is False

    def test_opus_55_unsupported(self):
        """v1.0.20: Opus 5.5 rejects any/tool "as on Claude Fable 5.1"."""
        assert _forced_tool_choice_unsupported("claude-opus-5-5") is True

    def test_future_opus_majors_assumed_unsupported(self):
        """Opus 6 keeps the restriction until a release says otherwise."""
        assert _forced_tool_choice_unsupported("claude-opus-6-0") is True
        assert _forced_tool_choice_unsupported("claude-opus-6") is True

    def test_required_downgraded_to_auto_on_opus_55(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
        )
        assert "tool_choice" not in stub.captured

    def test_required_downgraded_to_auto_on_fable_51(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-fable-5-1",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
        )
        assert "tool_choice" not in stub.captured

    def test_function_choice_downgraded_on_mythos_51(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-mythos-5-1",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        assert "tool_choice" not in stub.captured

    def test_none_choice_is_pass_through_even_on_fable_51(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-fable-5-1",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="none",
        )
        assert "tool_choice" not in stub.captured

    def test_required_still_works_on_opus_48(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
        )
        assert stub.captured["tool_choice"] == {"type": "any"}

    def test_function_choice_still_works_on_opus_5(self):
        completions, stub = _make_completions()
        completions.create(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        assert stub.captured["tool_choice"] == {"type": "tool", "name": "f"}


class TestVersionParsingEdgeCases:
    """v1.0.20 chain walk: real-world ID shapes must resolve predictably.

    The version-aware guards are the routing layer's judgment calls —
    these tests pin every shape that could be misjudged.
    """

    def test_snapshot_suffixes_ignored(self):
        """Dated snapshots share the base model's restrictions."""
        assert _forced_tool_choice_unsupported("claude-opus-5-5-20260922") is True
        assert _sampling_params_removed("claude-fable-5-1-20260901") is True

    def test_case_insensitive(self):
        assert _forced_tool_choice_unsupported("CLAUDE-OPUS-5-5") is True
        assert _sampling_params_removed("Claude-Opus-5-5") is True

    def test_gen3_ids_keep_legacy_behavior(self):
        """Pre-4.7 models still accept sampling and forced tool_choice."""
        assert _sampling_params_removed("claude-3-5-sonnet-20241022") is False
        assert _forced_tool_choice_unsupported("claude-3-5-sonnet-20241022") is False

    def test_sonnet_45_keeps_sampling(self):
        assert _sampling_params_removed("claude-sonnet-4-5") is False

    def test_sonnet_5_drops_sampling_but_keeps_forced_tools(self):
        """Every 5th-gen family drops sampling params; only Fable/Mythos
        5.1+ and Opus 5.5+ drop forced tool_choice (as documented today)."""
        assert _sampling_params_removed("claude-sonnet-5") is True
        assert _forced_tool_choice_unsupported("claude-sonnet-5") is False

    def test_unknown_names_keep_legacy_behavior(self):
        """Custom/renamed models must not be silently stripped."""
        assert _sampling_params_removed("my-private-claude") is False
        assert _forced_tool_choice_unsupported("my-private-claude") is False
