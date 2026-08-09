"""Regression tests for skill/validate.py targeted-repair routing.

Covers:
- Bug #17: _retarget_harness("A") must pass full body (not [:2500])
- _retarget_harness coverage for B/C/D/E/F call signatures
- _map_errors_to_harnesses field → harness routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Bug #17: _retarget_harness("A") passes truncated body ([:2500])
# ---------------------------------------------------------------------------

@dataclass
class _FakeManifest:
    """Minimal FileManifest stand-in for _retarget_harness tests."""

    entries: list = field(default_factory=list)
    entrypoint: str = ""

    def content_bundle(self) -> list:
        return []


@dataclass
class _FakeContext:
    """Minimal ContextPack stand-in."""

    frontmatter: dict | None = None
    body: str = ""
    file_manifest: Any = None
    dir_name: str = "test-skill"
    parse_warnings: list = field(default_factory=list)
    skill_dir: Path | None = None


class _FakeHarness:
    """Captures kwargs passed to run() for assertion."""

    def __init__(self, result: dict | None = None) -> None:
        self.captured_kwargs: dict[str, Any] = {}
        self._result = result or {
            "identity": {"id": "test-agent", "display_name": "Test"},
            "intent": {"triggers": [], "satisfies": [], "summary": ""},
            "interface": {"provides": [], "requires": [], "compatible_with": []},
            "base": {"runtime": "python3.11"},
            "instructions": {"workflow": "", "rules": [], "output_template": ""},
            "ahs_spec": {},
        }

    def run(self, **kwargs: Any) -> Any:
        from agenthatch.skill.spec import HarnessOutput

        self.captured_kwargs = kwargs
        return HarnessOutput(
            result=self._result,
            confidence=0.9,
            reasoning_trace=[],
            self_check_passed=True,
        )


class TestBug17RetargetHarnessAPassesFullBody:
    """Bug #17: ``_retarget_harness("A")`` must pass the full body, not ``[:2500]``.

    The Orchestrator main path (engine.py L1133) calls Harness A with
    ``body_first_50_lines=context.body`` (the full SKILL.md body).  But
    ``_retarget_harness`` in validate.py was passing
    ``body_first_50_lines=context.body[:2500]`` — a 2500-char truncation
    that doesn't exist in the main path.

    Impact: when SKILL.md body > 2500 chars (common for medium-size
    skills) and a Pydantic validation error triggers Harness A re-run,
    the LLM sees less context than the original run.  Identity fields
    referenced later in the body (e.g. an ``id`` declared in a code
    block past char 2500) become invisible, and the repair may fail
    even though the original Harness A run had no trouble extracting
    them.  This manifests as a spurious ``SchemaValidationError`` after
    exhausting ``max_targeted_retries``.

    Fix: ``_retarget_harness`` must pass ``context.body`` unchanged,
    matching the Orchestrator main path.
    """

    def test_retarget_a_passes_full_body_not_truncated(self) -> None:
        from agenthatch.skill.validate import _retarget_harness

        # Body longer than 2500 chars — the truncation point.
        long_body = "x" * 5000
        context = _FakeContext(
            frontmatter={"name": "test"},
            body=long_body,
            file_manifest=_FakeManifest(),
        )
        fake_a = _FakeHarness()
        harnesses = {"A": fake_a}
        outputs: dict[str, Any] = {}

        _retarget_harness("A", harnesses, outputs, context)

        captured = fake_a.captured_kwargs.get("body_first_50_lines", "")
        assert captured == long_body, (
            f"_retarget_harness('A') must pass full body ({len(long_body)} chars), "
            f"got truncated body ({len(captured)} chars). "
            f"Orchestrator main path passes context.body unchanged; "
            f"retarget must match."
        )

    def test_retarget_a_short_body_unchanged(self) -> None:
        """Sanity: short body (< 2500 chars) is also passed unchanged."""
        from agenthatch.skill.validate import _retarget_harness

        short_body = "short body"
        context = _FakeContext(
            frontmatter={"name": "test"},
            body=short_body,
            file_manifest=_FakeManifest(),
        )
        fake_a = _FakeHarness()
        harnesses = {"A": fake_a}
        outputs: dict[str, Any] = {}

        _retarget_harness("A", harnesses, outputs, context)

        assert fake_a.captured_kwargs["body_first_50_lines"] == short_body

    def test_retarget_a_signature_matches_main_path(self) -> None:
        """Static guard: retarget call must match Orchestrator's call signature.

        Orchestrator (engine.py) calls Harness A with:
            frontmatter, dir_name, body_first_50_lines, file_contents

        Retarget must pass the same four kwargs with the same semantics.
        """
        import ast

        engine_path = (
            Path(__file__).parent.parent
            / "src" / "agenthatch" / "skill" / "engine.py"
        )
        validate_path = (
            Path(__file__).parent.parent
            / "src" / "agenthatch" / "skill" / "validate.py"
        )

        engine_src = engine_path.read_text()
        validate_src = validate_path.read_text()

        # Main path must NOT use [:2500] on body_first_50_lines
        assert "body_first_50_lines=context.body[:2500]" not in engine_src, (
            "Orchestrator main path must not truncate body — if it does, "
            "the bug description changes"
        )
        assert "body_first_50_lines=context.body," in engine_src, (
            "Orchestrator main path must pass full context.body"
        )

        # Retarget path must also NOT use [:2500]
        assert "body_first_50_lines=context.body[:2500]" not in validate_src, (
            "BUG REGRESSION: _retarget_harness('A') truncates body to "
            "[:2500] — must pass full context.body to match Orchestrator"
        )


# ---------------------------------------------------------------------------
# _retarget_harness call signature coverage for B/C/D/E/F
# ---------------------------------------------------------------------------

class TestRetargetHarnessSignatures:
    """Verify each _retarget_harness call passes the right kwargs.

    These are smoke tests ensuring the retarget path doesn't silently
    drop or mutate inputs that the main path provides.
    """

    def test_retarget_b_passes_full_body(self) -> None:
        from agenthatch.skill.validate import _retarget_harness

        long_body = "y" * 5000
        context = _FakeContext(
            frontmatter={"name": "test", "description": "desc"},
            body=long_body,
            file_manifest=_FakeManifest(),
        )
        fake_b = _FakeHarness(result={"intent": {"summary": "x", "triggers": [], "satisfies": []}})
        harnesses = {"B": fake_b}
        outputs: dict[str, Any] = {}

        _retarget_harness("B", harnesses, outputs, context)

        assert fake_b.captured_kwargs["body"] == long_body
        assert fake_b.captured_kwargs["description"] == "desc"

    def test_retarget_c_passes_full_body(self) -> None:
        from agenthatch.skill.validate import _retarget_harness

        long_body = "z" * 5000
        context = _FakeContext(
            frontmatter={"name": "test", "allowed_tools": ["tool1"]},
            body=long_body,
            file_manifest=_FakeManifest(),
        )
        fake_c = _FakeHarness(result={"interface": {"provides": [], "requires": []}})
        harnesses = {"C": fake_c}
        outputs: dict[str, Any] = {}

        _retarget_harness("C", harnesses, outputs, context)

        assert fake_c.captured_kwargs["body"] == long_body

    def test_retarget_d_passes_full_body(self) -> None:
        from agenthatch.skill.validate import _retarget_harness

        long_body = "w" * 5000
        context = _FakeContext(
            frontmatter={"name": "test"},
            body=long_body,
            file_manifest=_FakeManifest(),
        )
        fake_d = _FakeHarness(result={"base": {"runtime": "python3.11"}, "instructions": {}})
        harnesses = {"D": fake_d}
        outputs: dict[str, Any] = {}

        _retarget_harness("D", harnesses, outputs, context)

        assert fake_d.captured_kwargs["body"] == long_body


# ---------------------------------------------------------------------------
# _map_errors_to_harnesses routing
# ---------------------------------------------------------------------------

class TestMapErrorsToHarnesses:
    """Verify field → harness routing is correct."""

    def test_identity_field_maps_to_a(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["identity", "id"], "msg": "bad", "type": "value_error"}]
        result = _map_errors_to_harnesses(errors)
        assert result == ["A"]

    def test_intent_field_maps_to_b(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["intent", "summary"], "msg": "bad", "type": "value_error"}]
        result = _map_errors_to_harnesses(errors)
        assert result == ["B"]

    def test_interface_field_maps_to_c(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["interface", "provides"], "msg": "bad", "type": "value_error"}]
        result = _map_errors_to_harnesses(errors)
        assert result == ["C"]

    def test_base_and_instructions_map_to_d(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [
            {"loc": ["base", "runtime"], "msg": "bad", "type": "value_error"},
            {"loc": ["instructions", "workflow"], "msg": "bad", "type": "value_error"},
        ]
        result = _map_errors_to_harnesses(errors)
        # base and instructions both → D, deduplicated
        assert result == ["D"]

    def test_root_parse_error_maps_to_e(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["__root__"], "msg": "parse error", "type": "parse_error"}]
        result = _map_errors_to_harnesses(errors)
        assert result == ["E"]

    def test_sub_field_resolution_via_parent_map(self) -> None:
        """Pydantic errors with loc[0] as a sub-field (e.g. 'id' not 'identity')."""
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["id"], "msg": "bad", "type": "value_error"}]
        result = _map_errors_to_harnesses(errors)
        # 'id' → parent 'identity' → Harness A
        assert result == ["A"]

    def test_unknown_field_returns_empty(self) -> None:
        from agenthatch.skill.validate import _map_errors_to_harnesses

        errors = [{"loc": ["unknown_field"], "msg": "bad", "type": "value_error"}]
        result = _map_errors_to_harnesses(errors)
        assert result == []
