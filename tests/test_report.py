"""Tests for HatchReport (v0.9.17).

Covers:
- Verdict computation (PASS / WARN — no FAIL, never blocks)
- JSON schema stability (CI consumers depend on field names)
- Token aggregation across harnesses and phases
- Terminal rendering (smoke test — no exception)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agenthatch.skill.report import (
    HarnessReport,
    HatchReport,
    PhaseReport,
    ReadinessSummary,
    build_hatch_report,
)
from agenthatch.skill.spec import HARNESS_LABELS, HarnessOutput

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _make_harness_output(
    key: str,
    *,
    confidence: float = 0.9,
    self_check_passed: bool = True,
    degradations: list[str] | None = None,
    retries: int = 0,
    tokens: dict[str, int] | None = None,
) -> HarnessOutput:
    """Build a minimal HarnessOutput for testing."""
    return HarnessOutput(
        result={},
        confidence=confidence,
        reasoning_trace=[f"[{key}] analyze: inputs received"],
        self_check_passed=self_check_passed,
        degradation_applied=degradations or [],
        internal_retries=retries,
        token_usage=tokens or {},
    )


def _make_readiness(status: str = "READY") -> Any:
    """Build a minimal ReadinessVerdict-like object for testing."""
    from agenthatch.generate.readiness import ReadinessVerdict

    return ReadinessVerdict(
        status=status,
        missing_optional=[] if status == "READY" else ["mcporter CLI not found"],
        fix_suggestions=[] if status == "READY" else ["npm install -g mcporter"],
    )


# ─────────────────────────────────────────────────────────────────────────
# Verdict tests
# ─────────────────────────────────────────────────────────────────────────


class TestVerdict:
    """Verdict is PASS or WARN only — no FAIL, never blocks."""

    def test_pass_when_all_clean(self):
        """All harnesses pass, readiness READY → PASS."""
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(key="A", label="extract_identity", confidence=0.9),
                HarnessReport(key="B", label="infer_intent", confidence=0.85),
            ],
            readiness=ReadinessSummary(status="READY"),
        )
        assert report.compute_verdict() == "PASS"

    def test_warn_when_degradation_applied(self):
        """Any harness with degradation → WARN."""
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(
                    key="A",
                    label="extract_identity",
                    confidence=0.5,
                    degradation_applied=["validation failed"],
                ),
            ],
            readiness=ReadinessSummary(status="READY"),
        )
        assert report.compute_verdict() == "WARN"

    def test_warn_when_self_check_failed(self):
        """Any harness with self_check_passed=False → WARN."""
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(
                    key="C",
                    label="infer_interface",
                    confidence=0.3,
                    self_check_passed=False,
                ),
            ],
            readiness=ReadinessSummary(status="READY"),
        )
        assert report.compute_verdict() == "WARN"

    def test_warn_when_readiness_warn(self):
        """Readiness WARN → WARN even if harnesses all pass."""
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(key="A", label="extract_identity", confidence=0.9),
            ],
            readiness=ReadinessSummary(status="WARN", missing_optional=["x"]),
        )
        assert report.compute_verdict() == "WARN"

    def test_no_fail_state_exists(self):
        """Verdict type is Literal['PASS', 'WARN'] — FAIL is impossible."""
        # Build a worst-case scenario: all harnesses fail
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(
                    key="A",
                    label="extract_identity",
                    confidence=0.0,
                    self_check_passed=False,
                    degradation_applied=["total failure"],
                ),
            ],
            readiness=ReadinessSummary(status="WARN", missing_optional=["everything"]),
        )
        verdict = report.compute_verdict()
        assert verdict == "WARN"
        assert verdict != "FAIL"

    def test_verdict_never_blocks(self):
        """Verdict is informational — it does not raise or block."""
        report = HatchReport(
            skill_id="test",
            skill_name="Test",
            generated_at=datetime.now(),
            harnesses=[
                HarnessReport(
                    key="A",
                    label="extract_identity",
                    confidence=0.0,
                    self_check_passed=False,
                    degradation_applied=["fatal"],
                ),
            ],
            readiness=ReadinessSummary(status="WARN"),
        )
        # compute_verdict should return a string, never raise
        result = report.compute_verdict()
        assert isinstance(result, str)
        assert result in ("PASS", "WARN")


# ─────────────────────────────────────────────────────────────────────────
# JSON schema tests
# ─────────────────────────────────────────────────────────────────────────


class TestJsonSchema:
    """JSON output schema stability for CI consumers."""

    def test_json_contains_required_fields(self):
        """JSON output must contain all top-level fields CI depends on."""
        tokens_a = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        report = build_hatch_report(
            skill_id="weather-reporter",
            skill_name="Weather Reporter",
            provider="deepseek",
            model="deepseek-v4-pro",
            phases=[
                PhaseReport(
                    name="phase_1_context",
                    label="Phase 1",
                    elapsed_seconds=0.5,
                ),
            ],
            harness_outputs={
                "A": _make_harness_output("A", tokens=tokens_a),
            },
            readiness=_make_readiness("READY"),
            agent_output_dir="/tmp/weather-agent",
            file_count=10,
            archetype="api_consumer",
            archetype_confidence=0.95,
        )
        data = json.loads(report.to_json())

        # Top-level fields
        assert data["skill_id"] == "weather-reporter"
        assert data["skill_name"] == "Weather Reporter"
        assert data["verdict"] == "PASS"
        assert data["provider"] == "deepseek"
        assert data["model"] == "deepseek-v4-pro"
        assert data["agent_output_dir"] == "/tmp/weather-agent"
        assert data["file_count"] == 10
        assert data["archetype"] == "api_consumer"
        assert "generated_at" in data

        # Phases
        assert len(data["phases"]) == 1
        assert data["phases"][0]["name"] == "phase_1_context"
        assert data["phases"][0]["elapsed_seconds"] == 0.5

        # Harnesses
        assert len(data["harnesses"]) == 1
        assert data["harnesses"][0]["key"] == "A"
        assert data["harnesses"][0]["label"] == "extract_identity"
        assert data["harnesses"][0]["token_usage"]["total_tokens"] == 150

        # Readiness
        assert data["readiness"]["status"] == "READY"

        # Total tokens
        assert data["total_tokens"]["total_tokens"] == 150

    def test_json_verdict_warn(self):
        """WARN verdict propagates to JSON."""
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={
                "A": _make_harness_output("A", degradations=["failed"]),
            },
            readiness=_make_readiness("WARN"),
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        data = json.loads(report.to_json())
        assert data["verdict"] == "WARN"

    def test_json_is_valid_json(self):
        """to_json() returns valid JSON parseable by json.loads."""
        report = build_hatch_report(
            skill_id="x",
            skill_name="X",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        parsed = json.loads(report.to_json())
        assert isinstance(parsed, dict)


# ─────────────────────────────────────────────────────────────────────────
# Token aggregation tests
# ─────────────────────────────────────────────────────────────────────────


class TestTokenAggregation:
    """Token usage is correctly aggregated across harnesses and phases."""

    def test_harness_tokens_summed(self):
        """Total tokens = sum of all harness token_usage."""
        tokens_a = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        tokens_b = {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
        tokens_c = {"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450}
        harness_outputs = {
            "A": _make_harness_output("A", tokens=tokens_a),
            "B": _make_harness_output("B", tokens=tokens_b),
            "C": _make_harness_output("C", tokens=tokens_c),
        }
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs=harness_outputs,
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        assert report.total_tokens["prompt_tokens"] == 600
        assert report.total_tokens["completion_tokens"] == 300
        assert report.total_tokens["total_tokens"] == 900

    def test_phase_tokens_added(self):
        """Phase 3 tokens are added to the total."""
        phases = [
            PhaseReport(
                name="phase_3_generation",
                label="Phase 3",
                elapsed_seconds=1.0,
                token_usage={
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                },
            ),
        ]
        tokens_a = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        harness_outputs = {
            "A": _make_harness_output("A", tokens=tokens_a),
        }
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=phases,
            harness_outputs=harness_outputs,
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        assert report.total_tokens["prompt_tokens"] == 600
        assert report.total_tokens["completion_tokens"] == 250
        assert report.total_tokens["total_tokens"] == 850

    def test_empty_tokens_when_no_llm(self):
        """No LLM calls → all token counts are 0."""
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={
                "A": _make_harness_output("A", tokens={}),
            },
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        assert report.total_tokens == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


# ─────────────────────────────────────────────────────────────────────────
# Terminal rendering smoke test
# ─────────────────────────────────────────────────────────────────────────


class TestTerminalRendering:
    """to_terminal() returns a Rich Group without raising."""

    def test_renders_without_error(self):
        """to_terminal() builds Rich renderables without exception."""
        report = build_hatch_report(
            skill_id="weather-reporter",
            skill_name="Weather Reporter",
            provider="deepseek",
            model="deepseek-v4-pro",
            phases=[
                PhaseReport(name="phase_1", label="Phase 1", elapsed_seconds=0.5),
                PhaseReport(name="phase_2", label="Phase 2", elapsed_seconds=2.3),
                PhaseReport(
                    name="phase_3",
                    label="Phase 3",
                    elapsed_seconds=1.1,
                    token_usage={"total_tokens": 500},
                ),
            ],
            harness_outputs={
                "A": _make_harness_output("A", confidence=0.95, tokens={"total_tokens": 200}),
                "B": _make_harness_output("B", confidence=0.88, tokens={"total_tokens": 300}),
                "C": _make_harness_output("C", confidence=0.92, tokens={"total_tokens": 400}),
            },
            readiness=_make_readiness("WARN"),
            agent_output_dir="/tmp/weather-agent",
            file_count=12,
            archetype="api_consumer",
            archetype_confidence=0.95,
        )
        group = report.to_terminal()
        assert group is not None

    def test_renders_minimal_report(self):
        """to_terminal() works with minimal data (no phases, no harnesses)."""
        report = build_hatch_report(
            skill_id="minimal",
            skill_name="Minimal",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
        )
        group = report.to_terminal()
        assert group is not None


# ─────────────────────────────────────────────────────────────────────────
# HARNESS_LABELS constant tests
# ─────────────────────────────────────────────────────────────────────────


class TestHarnessLabels:
    """HARNESS_LABELS is the single source of truth for harness labels."""

    def test_all_six_harnesses_present(self):
        """All 6 harness keys (A-F) are mapped."""
        assert set(HARNESS_LABELS.keys()) == {"A", "B", "C", "D", "E", "F"}

    def test_labels_match_expected(self):
        """Labels match the canonical names."""
        assert HARNESS_LABELS["A"] == "extract_identity"
        assert HARNESS_LABELS["B"] == "infer_intent"
        assert HARNESS_LABELS["C"] == "infer_interface"
        assert HARNESS_LABELS["D"] == "detect_base_and_instructions"
        assert HARNESS_LABELS["E"] == "assemble_and_validate"
        assert HARNESS_LABELS["F"] == "infer_mcp_servers"


# ─────────────────────────────────────────────────────────────────────────
# HarnessOutput token_usage field tests
# ─────────────────────────────────────────────────────────────────────────


class TestHarnessOutputTokenUsage:
    """HarnessOutput.token_usage field works as expected."""

    def test_default_empty_dict(self):
        """token_usage defaults to empty dict."""
        out = HarnessOutput(
            result={},
            confidence=0.9,
            reasoning_trace=[],
            self_check_passed=True,
        )
        assert out.token_usage == {}

    def test_can_set_token_usage(self):
        """token_usage can be set at construction."""
        out = HarnessOutput(
            result={},
            confidence=0.9,
            reasoning_trace=[],
            self_check_passed=True,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert out.token_usage["total_tokens"] == 150
