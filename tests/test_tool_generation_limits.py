"""Regression: AI tool generation must be able to finish (v1.0.21).

Tool implementations are delivered as ONE JSON document containing a full
function body per capability, generated from a ~50K-char skill context.
With the old 16384-token ceiling the response was cut mid-string, so
``json.loads`` raised "Unterminated string", ``_ai_generate_tool_impls``
returned ``{}``, and *every* tool silently degraded to a stub — while the
hatch still printed a successful summary. Thinking models compound this,
because reasoning tokens are drawn from the same budget.

These tests pin the budget so a future refactor cannot quietly reintroduce
the ceiling.
"""

from __future__ import annotations

from typing import Any

import agenthatch.providers as providers_mod
from agenthatch.cli.commands.hatch import (
    AI_TOOL_MAX_TOKENS,
    _create_ai_chat_fn,
)

# Enough headroom for several tool bodies plus thinking tokens.
_MIN_ACCEPTABLE_MAX_TOKENS = 32768


def _config(api_key: str = "test-key-not-used") -> dict[str, Any]:
    return {
        "agenthatch": {"default": "deepseek"},
        "providers": {
            "deepseek": {
                "api_key": api_key,
                "base_url": "https://api.deepseek.com/v1",
                "default_model": "deepseek-flash",
            }
        },
    }


class TestToolGenerationBudget:
    def test_budget_is_large_enough_for_multiple_tools(self) -> None:
        """A small budget truncates the JSON and stubs out every tool."""
        assert AI_TOOL_MAX_TOKENS >= _MIN_ACCEPTABLE_MAX_TOKENS, (
            "AI tool generation writes one large JSON document holding a "
            "function body per capability; too small a ceiling truncates it "
            "mid-string and turns every tool into a stub "
            f"(got {AI_TOOL_MAX_TOKENS})"
        )

    def test_budget_is_not_absurdly_large(self) -> None:
        """Guard the other direction: don't ask for a provider's whole window."""
        assert AI_TOOL_MAX_TOKENS <= 200_000


class TestChatFnConstruction:
    def test_no_api_key_returns_none(self, monkeypatch: Any) -> None:
        """Absent key → None, so the caller can warn instead of stubbing."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("AGENTHATCH_API_KEY", raising=False)
        monkeypatch.setattr(
            providers_mod, "resolve_api_key", lambda *a, **kw: None
        )

        assert _create_ai_chat_fn(_config(api_key="")) is None

    def test_with_api_key_returns_callable(self) -> None:
        """A usable key must yield a chat function, not None."""
        assert callable(_create_ai_chat_fn(_config()))
