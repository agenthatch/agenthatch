"""Tests for Phase 2 harness self-reflection (v0.9.20).

Covers the reflection-correction loop (``reflect_and_correct_harness``)
and the CP2 fidelity checkpoint (``run_fidelity_checkpoint``).

Design follows the project's mock pattern: ``MagicMock`` for the LLM
client, lightweight fakes for the harness contract. No real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agenthatch.skill.reflection import (
    CorrectionItem,
    FidelityCheckpointOutput,
    HarnessReflectionOutput,
    apply_corrections,
    reflect_and_correct_harness,
    run_fidelity_checkpoint,
    should_skip_reflection,
)
from agenthatch.skill.spec import HarnessOutput

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _FakeHarness:
    """Minimal harness stand-in for reflection tests.

    Provides the attributes ``reflect_and_correct_harness`` reads:
    ``name``, ``client``, ``model``, ``validate_output``.
    """

    def __init__(
        self,
        name: str = "B",
        validate_fn=None,
    ) -> None:
        self.name = name
        self.client = MagicMock()
        self.client.last_usage = None
        self.model = "test-model"
        self._validate_fn = validate_fn or (lambda r: (True, ""))

    def validate_output(self, result: dict) -> tuple[bool, str]:
        return self._validate_fn(result)


def _reflection_output(
    *,
    self_consistency: str = "pass",
    cross_harness: str = "pass",
    fidelity: str = "pass",
    corrections: list[CorrectionItem] | None = None,
    confidence: float = 0.9,
) -> HarnessReflectionOutput:
    return HarnessReflectionOutput(
        self_consistency=self_consistency,
        self_consistency_detail="",
        cross_harness=cross_harness,
        cross_harness_detail="",
        fidelity=fidelity,
        fidelity_detail="",
        corrections=corrections or [],
        overall_confidence=confidence,
    )


def _make_output(
    result: dict | None = None,
    confidence: float = 0.8,
    self_check: bool = True,
) -> HarnessOutput:
    return HarnessOutput(
        result=result or {"intent": {"summary": "a" * 30}},
        confidence=confidence,
        reasoning_trace=[],
        self_check_passed=self_check,
    )


# ---------------------------------------------------------------------------
# should_skip_reflection
# ---------------------------------------------------------------------------


class TestShouldSkipReflection:
    def test_skip_a_high_confidence(self):
        assert should_skip_reflection("extract_identity", 0.95) is True

    def test_skip_f_high_confidence(self):
        assert should_skip_reflection("infer_mcp_servers", 0.92) is True

    def test_skip_e_very_high_confidence(self):
        assert should_skip_reflection("assemble_and_validate", 0.96) is True

    def test_no_skip_e_below_threshold(self):
        assert should_skip_reflection("assemble_and_validate", 0.94) is False

    def test_no_skip_b_regardless_of_confidence(self):
        # B has no skip threshold — always reflects
        assert should_skip_reflection("infer_intent", 0.99) is False

    def test_no_skip_when_errors(self):
        assert should_skip_reflection("extract_identity", 0.95, has_errors=True) is False


# ---------------------------------------------------------------------------
# reflect_and_correct_harness
# ---------------------------------------------------------------------------


class TestReflectAndCorrectHarness:
    def test_skip_high_confidence_harness_a(self):
        """Harness A with confidence >= 0.9 bypasses reflection."""
        harness = _FakeHarness(name="extract_identity")
        output = _make_output(confidence=0.95)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
        )

        assert corrections == []
        assert "skipped" in result.reasoning_trace[-1]
        harness.client.chat_structured.assert_not_called()

    def test_skip_no_validate_output_override(self):
        """Harness without validate_output override is skipped."""
        def raise_not_implemented(_r: dict) -> tuple[bool, str]:
            raise NotImplementedError()

        harness = _FakeHarness(name="B", validate_fn=raise_not_implemented)
        output = _make_output(confidence=0.5)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
        )

        assert corrections == []
        assert "no validate_output" in result.reasoning_trace[-1]
        harness.client.chat_structured.assert_not_called()

    def test_pass_clean_no_corrections(self):
        """Reflection passes all dimensions with no corrections → output unchanged."""
        harness = _FakeHarness(name="B")
        harness.client.chat_structured.return_value = _reflection_output(
            confidence=0.92,
        )
        original_result = {"intent": {"summary": "a" * 30}}
        output = _make_output(result=original_result, confidence=0.8)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
        )

        assert corrections == []
        assert result.result == original_result
        assert any("passed" in t for t in result.reasoning_trace)
        # Confidence updated conservatively (min of original and reflection's)
        assert result.confidence == pytest.approx(0.8)

    def test_corrections_committed(self):
        """Corrections that pass post-validation are committed to result."""
        harness = _FakeHarness(name="B")
        harness.client.chat_structured.return_value = _reflection_output(
            fidelity="fail",
            corrections=[
                CorrectionItem(
                    field="intent.summary",
                    new_value="corrected summary that is long enough",
                    reason="SKILL.md says otherwise",
                )
            ],
            confidence=0.85,
        )
        output = _make_output(
            result={"intent": {"summary": "a" * 30}},
            confidence=0.7,
        )

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
            max_rounds=1,
        )

        assert len(corrections) == 1
        assert result.result["intent"]["summary"] == "corrected summary that is long enough"
        assert result.confidence == pytest.approx(0.85)
        assert any("committed" in t for t in result.reasoning_trace)

    def test_corrections_rolled_back_on_validation_failure(self):
        """Corrections that fail post-validation are rolled back."""
        # validate_output rejects the corrected value
        def strict_validate(r):
            summary = r.get("intent", {}).get("summary", "")
            if "BAD" in summary:
                return False, "summary contains BAD"
            return True, ""

        harness = _FakeHarness(name="B", validate_fn=strict_validate)
        harness.client.chat_structured.return_value = _reflection_output(
            fidelity="fail",
            corrections=[
                CorrectionItem(
                    field="intent.summary",
                    new_value="BAD value",
                    reason="wrong correction",
                )
            ],
            confidence=0.6,
        )
        original = {"intent": {"summary": "a" * 30}}
        output = _make_output(result=original, confidence=0.7)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
        )

        # Rollback: result unchanged, corrections empty
        assert corrections == []
        assert result.result == original
        assert any("rolled back" in t for t in result.reasoning_trace)

    def test_llm_failure_returns_output_unchanged(self):
        """When chat_structured raises, output is returned unchanged."""
        harness = _FakeHarness(name="B")
        harness.client.chat_structured.side_effect = RuntimeError("API down")
        original = {"intent": {"summary": "a" * 30}}
        output = _make_output(result=original, confidence=0.7)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
        )

        # run_harness_reflection catches the exception and returns all-pass sentinel
        assert corrections == []
        assert result.result == original

    def test_two_rounds_converge(self):
        """Round 1 produces corrections, round 2 passes clean."""
        harness = _FakeHarness(name="B")
        # Round 1: fail + corrections; Round 2: pass clean
        harness.client.chat_structured.side_effect = [
            _reflection_output(
                fidelity="fail",
                corrections=[
                    CorrectionItem(
                        field="intent.summary",
                        new_value="corrected summary long enough",
                        reason="fix",
                    )
                ],
                confidence=0.85,
            ),
            _reflection_output(confidence=0.9),
        ]
        output = _make_output(
            result={"intent": {"summary": "a" * 30}},
            confidence=0.7,
        )

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
            max_rounds=2,
        )

        assert len(corrections) == 1
        assert result.result["intent"]["summary"] == "corrected summary long enough"
        assert harness.client.chat_structured.call_count == 2

    def test_flagged_but_no_corrections_stops(self):
        """Reflection flags failure but proposes no correction → stop iterating."""
        harness = _FakeHarness(name="B")
        harness.client.chat_structured.return_value = _reflection_output(
            self_consistency="fail",
            corrections=[],  # flagged but no fixable field
            confidence=0.5,
        )
        output = _make_output(confidence=0.7)

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
            max_rounds=2,
        )

        assert corrections == []
        # Should stop after 1 round, not loop
        assert harness.client.chat_structured.call_count == 1
        assert any("no correctable field" in t for t in result.reasoning_trace)


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    def test_simple_dot_path(self):
        output = {"identity": {"id": "old", "display_name": "X"}}
        corrections = [
            CorrectionItem(field="identity.id", new_value="new-name", reason="r"),
        ]
        result = apply_corrections(output, corrections)
        assert result["identity"]["id"] == "new-name"

    def test_array_index_path(self):
        output = {"provides": [{"capability": "a"}, {"capability": "b"}]}
        corrections = [
            CorrectionItem(
                field="provides[1].capability", new_value="c", reason="r"
            ),
        ]
        result = apply_corrections(output, corrections)
        assert result["provides"][1]["capability"] == "c"

    def test_missing_path_silently_skipped(self):
        output = {"identity": {"id": "x"}}
        corrections = [
            CorrectionItem(field="nonexistent.field", new_value="v", reason="r"),
        ]
        result = apply_corrections(output, corrections)
        # Original unchanged, no crash
        assert result == output

    def test_does_not_mutate_original(self):
        output = {"identity": {"id": "old"}}
        corrections = [
            CorrectionItem(field="identity.id", new_value="new", reason="r"),
        ]
        apply_corrections(output, corrections)
        assert output["identity"]["id"] == "old"  # original intact

    def test_type_auto_conversion_list_from_json_string(self):
        """LLM returns JSON string for a list field — should be parsed."""
        output = {"intent": {"triggers": ["weather", "forecast"]}}
        corrections = [
            CorrectionItem(
                field="intent.triggers",
                old_value='["weather", "forecast"]',
                new_value='["weather", "forecast", "temperature"]',
                reason="add temperature",
            ),
        ]
        result = apply_corrections(output, corrections)
        assert result["intent"]["triggers"] == ["weather", "forecast", "temperature"]
        assert isinstance(result["intent"]["triggers"], list)

    def test_type_auto_conversion_dict_from_json_string(self):
        """LLM returns JSON string for a dict field — should be parsed."""
        output = {"interface": {"provides": [{"input_schema": {"city": "string"}}]}}
        corrections = [
            CorrectionItem(
                field="interface.provides[0].input_schema",
                old_value='{"city": "string"}',
                new_value='{"city": "string", "country": "string"}',
                reason="add country",
            ),
        ]
        result = apply_corrections(output, corrections)
        assert result["interface"]["provides"][0]["input_schema"] == {
            "city": "string",
            "country": "string",
        }
        assert isinstance(result["interface"]["provides"][0]["input_schema"], dict)

    def test_old_value_mismatch_skips_correction(self):
        """When old_value doesn't match current value, correction is skipped."""
        output = {"identity": {"id": "actual-id"}}
        corrections = [
            CorrectionItem(
                field="identity.id",
                old_value="wrong-id",  # doesn't match actual
                new_value="new-id",
                reason="fix",
            ),
        ]
        result = apply_corrections(output, corrections)
        # Correction skipped, original preserved
        assert result["identity"]["id"] == "actual-id"

    def test_old_value_match_applies_correction(self):
        """When old_value matches current value, correction is applied."""
        output = {"identity": {"id": "old-id"}}
        corrections = [
            CorrectionItem(
                field="identity.id",
                old_value="old-id",  # matches
                new_value="new-id",
                reason="fix",
            ),
        ]
        result = apply_corrections(output, corrections)
        assert result["identity"]["id"] == "new-id"

    def test_empty_old_value_always_applies(self):
        """Empty old_value means no precondition — correction always applies."""
        output = {"identity": {"id": "anything"}}
        corrections = [
            CorrectionItem(
                field="identity.id",
                old_value="",  # no precondition
                new_value="new-id",
                reason="fix",
            ),
        ]
        result = apply_corrections(output, corrections)
        assert result["identity"]["id"] == "new-id"


# ---------------------------------------------------------------------------
# Type preservation rollback (reflect_and_correct_harness)
# ---------------------------------------------------------------------------


class TestTypePreservationRollback:
    def test_list_to_string_rollback(self):
        """Correction that changes list → str is rolled back by type check."""
        # Permissive harness: validate_output always passes
        harness = _FakeHarness(name="B")
        # Correction replaces triggers list with a non-JSON string
        # _coerce_value won't salvage it (doesn't start with [)
        harness.client.chat_structured.return_value = _reflection_output(
            fidelity="fail",
            corrections=[
                CorrectionItem(
                    field="intent.triggers",
                    new_value="not a json string",
                    reason="wrong type",
                )
            ],
            confidence=0.6,
        )
        output = _make_output(
            result={"intent": {"summary": "a" * 30, "triggers": ["a", "b"]}},
            confidence=0.7,
        )

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
            max_rounds=1,
        )

        # validate_output passes (permissive), but type check catches list→str
        assert corrections == []
        assert result.result["intent"]["triggers"] == ["a", "b"]  # original preserved
        assert any("rolled back" in t for t in result.reasoning_trace)

    def test_str_value_change_not_flagged(self):
        """str → str value change is NOT a type change — no rollback."""
        harness = _FakeHarness(name="B")
        harness.client.chat_structured.return_value = _reflection_output(
            fidelity="fail",
            corrections=[
                CorrectionItem(
                    field="intent.summary",
                    new_value="a better summary that is long enough",
                    reason="improve",
                )
            ],
            confidence=0.85,
        )
        output = _make_output(
            result={"intent": {"summary": "a" * 30}},
            confidence=0.7,
        )

        result, corrections = reflect_and_correct_harness(
            harness=harness,
            output=output,
            skill_md_body="body",
            peer_outputs={},
            max_rounds=1,
        )

        # str → str is fine, correction should commit
        assert len(corrections) == 1
        assert result.result["intent"]["summary"] == "a better summary that is long enough"


# ---------------------------------------------------------------------------
# run_fidelity_checkpoint
# ---------------------------------------------------------------------------


class TestRunFidelityCheckpoint:
    def test_success(self):
        client = MagicMock()
        expected = FidelityCheckpointOutput(
            coverage="pass",
            hallucination="pass",
            fidelity_score=0.95,
            recommendations=[],
        )
        client.chat_structured.return_value = expected

        result = run_fidelity_checkpoint(
            client=client,
            model="test-model",
            skill_md="skill body",
            ahspec_json='{"identity": {}}',
        )

        assert result is expected
        client.chat_structured.assert_called_once()

    def test_llm_failure_returns_sentinel(self):
        client = MagicMock()
        client.chat_structured.side_effect = RuntimeError("timeout")

        result = run_fidelity_checkpoint(
            client=client,
            model="test-model",
            skill_md="body",
            ahspec_json="{}",
        )

        assert result.fidelity_score == 0.5
        assert result.coverage == "pass"
        assert result.hallucination == "pass"
        assert "failed" in result.coverage_detail.lower()
