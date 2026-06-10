"""Smoke tests for the agentic inference engine (skill/engine.py).

Covers the Orchestrator's LLMClient construction to prevent
regressions where api_key is not passed to LLMClient — a bug
pattern that affected both SkillAgent (Fix #7) and Orchestrator
(Fix #15) after the v0.7.5 llm_client migration to core.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agenthatch.skill.engine import Orchestrator


@pytest.fixture
def minimal_config() -> dict:
    """Minimal config with a default provider."""
    return {"providers": {"default": "deepseek"}}


def test_orchestrator_passes_api_key_to_llm_client(minimal_config):
    """Orchestrator must pass resolved api_key to both LLMClient instances.

    Regression test for Fix #15: after v0.7.5 migrated LLMClient to core,
    the Orchestrator (hatch command path) was constructing LLMClient without
    api_key, causing "No API key provided" errors during `agenthatch hatch`.
    """
    # Mock get_provider to return a minimal provider info
    mock_provider_info = MagicMock()
    mock_provider_info.default_model = "test-model"

    with (
        patch("agenthatch.skill.engine.LLMClient") as mock_llm,
        patch("agenthatch.providers.get_provider", return_value=mock_provider_info),
        patch("agenthatch.providers.resolve_api_key", return_value="test-api-key"),
    ):
        _ = Orchestrator(minimal_config)

    # H2 fix: when large_model == small_model, Orchestrator creates only 1
    # LLMClient and reuses it for both tiers.
    # minimal_config has no per-tier model config, so both default to the
    # same model → only 1 LLMClient is created.
    assert mock_llm.call_count in (1, 2), (
        f"Expected 1 or 2 LLMClient calls (1 when models match), "
        f"got {mock_llm.call_count}"
    )

    # Both calls (or the single call) should receive the resolved api_key
    for i, call in enumerate(mock_llm.call_args_list):
        assert call.kwargs.get("api_key") == "test-api-key", (
            f"LLMClient call {i} did not receive api_key: "
            f"kwargs={call.kwargs}"
        )
        assert call.kwargs.get("provider") == "deepseek", (
            f"LLMClient call {i} provider mismatch: {call.kwargs}"
        )
        assert call.kwargs.get("model") == "test-model", (
            f"LLMClient call {i} model mismatch: {call.kwargs}"
        )


def test_orchestrator_uses_large_model_override(minimal_config):
    """When large_model is provided via config, it should be used."""
    mock_provider_info = MagicMock()
    mock_provider_info.default_model = "test-model"

    with (
        patch("agenthatch.skill.engine.LLMClient") as mock_llm,
        patch("agenthatch.providers.get_provider", return_value=mock_provider_info),
        patch("agenthatch.providers.resolve_api_key", return_value="test-api-key"),
    ):
        Orchestrator(minimal_config, large_model="gpt-4o", small_model="gpt-4o-mini")

    # First call is large model, second is small model
    assert mock_llm.call_args_list[0].kwargs["model"] == "gpt-4o"
    assert mock_llm.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
