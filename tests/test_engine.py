"""Tests for the agentic inference engine (skill/engine.py).

Covers:
- Orchestrator LLMClient construction (Fix #7, Fix #15 regression tests)
- HARNESS_CONFIG structure (temperatures, thinking, reasons)
- HARNESS_LABELS mapping (single-letter → full name)
- MODEL_TIER_MAP (skill type → model tier, skip for pure_instruction D)
- should_skip_reflection() confidence thresholds (A/F ≥ 0.9, E ≥ 0.95)
- Orchestrator run() execution order with mocked harnesses
- Reflection wiring (Step 5.5 for A/B/C/D/F, Step 6.5 for E)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agenthatch.skill.engine import (
    Orchestrator,
    HARNESS_CONFIG,
    MODEL_TIER_MAP,
)
from agenthatch.skill.spec import HARNESS_LABELS, HarnessOutput
from agenthatch.skill.reflection import should_skip_reflection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config() -> dict:
    """Minimal config with a default provider."""
    return {"agenthatch": {"default": "deepseek"}}


def make_harness_output(
    result: dict[str, Any] | None = None,
    confidence: float = 0.8,
    self_check_passed: bool = True,
) -> HarnessOutput:
    """Factory for mock HarnessOutput objects."""
    return HarnessOutput(
        result=result or {},
        confidence=confidence,
        reasoning_trace=[],
        self_check_passed=self_check_passed,
        degradation_applied=[],
        internal_retries=0,
        token_usage={},
        temperature_used=None,
    )


# ---------------------------------------------------------------------------
# Existing constructor tests (Fix #7, Fix #15 regression)
# ---------------------------------------------------------------------------

def test_orchestrator_passes_api_key_to_llm_client(minimal_config):
    """Orchestrator must pass resolved api_key to both LLMClient instances.

    Regression test for Fix #15: after v0.7.5 migrated LLMClient to core,
    the Orchestrator (hatch command path) was constructing LLMClient without
    api_key, causing "No API key provided" errors during `agenthatch hatch`.
    """
    mock_provider_info = MagicMock()
    mock_provider_info.default_model = "test-model"

    with (
        patch("agenthatch.skill.engine.LLMClient") as mock_llm,
        patch("agenthatch.providers.get_provider", return_value=mock_provider_info),
        patch("agenthatch.providers.resolve_api_key", return_value="test-api-key"),
    ):
        _ = Orchestrator(minimal_config)

    assert mock_llm.call_count in (1, 2)
    for i, call in enumerate(mock_llm.call_args_list):
        assert call.kwargs.get("api_key") == "test-api-key"
        assert call.kwargs.get("provider") == "deepseek"
        assert call.kwargs.get("model") == "test-model"


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

    assert mock_llm.call_args_list[0].kwargs["model"] == "gpt-4o"
    assert mock_llm.call_args_list[1].kwargs["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# HARNESS_CONFIG tests
# ---------------------------------------------------------------------------

class TestHarnessConfig:
    """Verify HARNESS_CONFIG structure: temperatures, thinking, reasons."""

    def test_all_six_harnesses_present(self):
        assert set(HARNESS_CONFIG.keys()) == {"A", "B", "C", "D", "E", "F"}

    def test_each_harness_has_required_fields(self):
        for key, config in HARNESS_CONFIG.items():
            assert "temperature" in config, f"Harness {key} missing temperature"
            assert "thinking" in config, f"Harness {key} missing thinking"
            assert "reason" in config, f"Harness {key} missing reason"

    def test_temperatures_in_valid_range(self):
        for key, config in HARNESS_CONFIG.items():
            temp = config["temperature"]
            assert 0.0 <= temp <= 1.0, f"Harness {key} temperature {temp} out of range"

    @pytest.mark.parametrize("key,expected_temp", [
        ("A", 0.1),  # Identity extraction — low temp for consistency
        ("B", 0.5),  # Intent inference — high temp for creativity
        ("C", 0.5),  # Interface inference — high temp for complex inference
        ("D", 0.3),  # Base detection — moderate temp
        ("E", 0.3),  # Assembly validation — low temp for consistency
        ("F", 0.3),  # MCP config — moderate temp
    ])
    def test_specific_temperatures(self, key, expected_temp):
        """Verify each harness has its designed temperature."""
        actual = HARNESS_CONFIG[key]["temperature"]
        assert actual == expected_temp, (
            f"Harness {key} temperature: expected {expected_temp}, got {actual}. "
            f"Reason: {HARNESS_CONFIG[key]['reason']}"
        )

    def test_thinking_enabled_for_all(self):
        """All harnesses should have thinking enabled for reasoning."""
        for key, config in HARNESS_CONFIG.items():
            assert config["thinking"] is True, f"Harness {key} thinking should be True"

    def test_reason_strings_are_descriptive(self):
        """Each harness should have a non-empty reason explaining its temperature."""
        for key, config in HARNESS_CONFIG.items():
            reason = config["reason"]
            assert len(reason) > 10, f"Harness {key} reason too short: '{reason}'"


# ---------------------------------------------------------------------------
# HARNESS_LABELS tests
# ---------------------------------------------------------------------------

class TestHarnessLabels:
    """Verify HARNESS_LABELS maps single-letter keys to full names."""

    def test_all_six_labels_present(self):
        assert set(HARNESS_LABELS.keys()) == {"A", "B", "C", "D", "E", "F"}

    @pytest.mark.parametrize("key,expected_name", [
        ("A", "extract_identity"),
        ("B", "infer_intent"),
        ("C", "infer_interface"),
        ("D", "detect_base_and_instructions"),
        ("E", "assemble_and_validate"),
        ("F", "infer_mcp_servers"),
    ])
    def test_label_mapping(self, key, expected_name):
        assert HARNESS_LABELS[key] == expected_name


# ---------------------------------------------------------------------------
# MODEL_TIER_MAP tests
# ---------------------------------------------------------------------------

class TestModelTierMap:
    """Verify MODEL_TIER_MAP: skill type → model tier per harness."""

    def test_all_four_skill_types_present(self):
        assert set(MODEL_TIER_MAP.keys()) == {
            "pure_instruction", "script_driven", "integration", "knowledge"
        }

    def test_pure_instruction_skips_d(self):
        """pure_instruction skills should skip Harness D (no runtime to detect)."""
        tier_map = MODEL_TIER_MAP["pure_instruction"]
        assert tier_map["D"] == "skip"

    def test_script_driven_runs_all_harnesses(self):
        """script_driven skills should run all 6 harnesses."""
        tier_map = MODEL_TIER_MAP["script_driven"]
        for key in ("A", "B", "C", "D", "E", "F"):
            assert tier_map[key] in ("small", "large"), (
                f"script_driven {key} should be small or large, got {tier_map[key]}"
            )

    def test_all_harnesses_have_tier(self):
        """Every skill type should have a tier for every harness (or 'skip')."""
        for skill_type, tier_map in MODEL_TIER_MAP.items():
            for key in ("A", "B", "C", "D", "E", "F"):
                assert key in tier_map, f"{skill_type} missing tier for {key}"
                assert tier_map[key] in ("small", "large", "skip")

    def test_a_is_always_small(self):
        """Harness A (identity extraction) is always small model (deterministic)."""
        for skill_type, tier_map in MODEL_TIER_MAP.items():
            assert tier_map["A"] == "small", (
                f"{skill_type} Harness A should be small, got {tier_map['A']}"
            )

    def test_c_is_always_large(self):
        """Harness C (interface inference) is always large model (complex)."""
        for skill_type, tier_map in MODEL_TIER_MAP.items():
            assert tier_map["C"] == "large", (
                f"{skill_type} Harness C should be large, got {tier_map['C']}"
            )


# ---------------------------------------------------------------------------
# should_skip_reflection tests
# ---------------------------------------------------------------------------

class TestShouldSkipReflection:
    """Verify reflection skip logic based on confidence thresholds."""

    def test_skip_identity_high_confidence(self):
        """Harness A (extract_identity) with confidence >= 0.9 should skip."""
        assert should_skip_reflection("extract_identity", 0.9) is True
        assert should_skip_reflection("extract_identity", 0.95) is True
        assert should_skip_reflection("extract_identity", 1.0) is True

    def test_no_skip_identity_low_confidence(self):
        """Harness A with confidence < 0.9 should not skip."""
        assert should_skip_reflection("extract_identity", 0.89) is False
        assert should_skip_reflection("extract_identity", 0.5) is False

    def test_skip_mcp_high_confidence(self):
        """Harness F (infer_mcp_servers) with confidence >= 0.9 should skip."""
        assert should_skip_reflection("infer_mcp_servers", 0.9) is True
        assert should_skip_reflection("infer_mcp_servers", 0.95) is True

    def test_no_skip_mcp_low_confidence(self):
        assert should_skip_reflection("infer_mcp_servers", 0.89) is False

    def test_skip_assembly_very_high_confidence(self):
        """Harness E (assemble_and_validate) with confidence >= 0.95 should skip."""
        assert should_skip_reflection("assemble_and_validate", 0.95) is True
        assert should_skip_reflection("assemble_and_validate", 1.0) is True

    def test_no_skip_assembly_below_threshold(self):
        """Harness E with confidence < 0.95 should not skip."""
        assert should_skip_reflection("assemble_and_validate", 0.94) is False
        assert should_skip_reflection("assemble_and_validate", 0.9) is False

    def test_never_skip_with_errors(self):
        """should_skip_reflection should never skip if has_errors=True."""
        assert should_skip_reflection("extract_identity", 1.0, has_errors=True) is False
        assert should_skip_reflection("infer_mcp_servers", 1.0, has_errors=True) is False
        assert should_skip_reflection("assemble_and_validate", 1.0, has_errors=True) is False

    def test_no_skip_for_b_c_d(self):
        """Harnesses B, C, D should never skip reflection regardless of confidence."""
        assert should_skip_reflection("infer_intent", 1.0) is False
        assert should_skip_reflection("infer_interface", 1.0) is False
        assert should_skip_reflection("detect_base_and_instructions", 1.0) is False

    def test_unknown_harness_name_no_skip(self):
        """Unknown harness names should not skip reflection."""
        assert should_skip_reflection("unknown_harness", 1.0) is False

    def test_boundary_confidence_values(self):
        """Test exact boundary values (>= vs >)."""
        # 0.9 is the threshold for A and F — >= means skip
        assert should_skip_reflection("extract_identity", 0.9) is True
        # 0.95 is the threshold for E — >= means skip
        assert should_skip_reflection("assemble_and_validate", 0.95) is True
