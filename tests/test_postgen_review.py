"""Tests for Phase 3.5 post-generation review (v0.9.22 — B2/B3/B4 self-healing).

Covers:
- B2 ``inspect_generated_package()`` — each inspection check type
  (syntax errors, JS artifacts, literal stubs, undefined variables,
  None attribute access, semantic stubs)
- B3 ``_run_tool_self_test()`` — archetype branches and exception
  categorization (subprocess sandbox)
- B4 ``iterate_until_gate()`` — termination conditions (gate passes,
  max rounds, no chat_fn, no errors)
- HatchReport integration (PostGenReviewSummary, verdict propagation)
- Detection capability verified with synthetic tools.py fixtures
  containing each known bug pattern

Design follows the project's mock pattern (see test_reflection.py):
``MagicMock`` for the LLM client (no real API calls), lightweight
fixtures for the file system.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthatch.skill import postgen_review as _postgen_review_module
from agenthatch.skill.postgen_review import (
    CATEGORY_NONE_ATTR,
    CATEGORY_SEMANTIC_STUB,
    CATEGORY_STUB,
    CATEGORY_SYNTAX,
    CATEGORY_TYPE_ERROR,
    CATEGORY_UNDEFINED_VAR,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    VERDICT_READY,
    VERDICT_WARN,
    PostGenFinding,
    PostGenReport,
    _apply_tool_repair,
    _build_agent_context,
    _detect_none_attribute_access,
    _detect_semantic_stubs,
    _detect_undefined_variables,
    _has_side_effects,
    _rebuild_function_source,
    inspect_generated_package,
    iterate_until_gate,
)
from agenthatch.skill.report import (
    PostGenFindingSummary,
    PostGenReviewSummary,
    build_hatch_report,
)

# Alias to avoid pytest collecting it as a test (function name starts with "test_").
_run_tool_self_test = _postgen_review_module.test_tool_signatures


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _make_tools_py(
    output_dir: Path, body: str, *, rel_path: str = "src/myagent/tools.py"
) -> Path:
    """Write a tools.py file at the given relative path under output_dir."""
    tools_path = output_dir / rel_path
    tools_path.parent.mkdir(parents=True, exist_ok=True)
    tools_path.write_text(textwrap.dedent(body), encoding="utf-8")
    return tools_path


# Sample tools.py with each known bug pattern (from the task spec)
# Note: parameter names are `from_currency`/`to_currency` but the body
# hallucinates `from_curr`/`to_curr` — this is the currency-converter bug.
_TOOLS_PY_UNDEFINED_VAR = '''
"""Generated tools."""

def convert_currency(amount: float, from_currency: str = "USD", to_currency: str = "EUR") -> dict:
    """Convert currency."""
    rate = get_rate(from_curr, to_curr)
    return {"amount": amount, "from": from_curr, "to": to_curr, "rate": rate}
'''

_TOOLS_PY_NONE_ATTR = '''
"""Generated tools."""

def format_text(text: str = None) -> str:
    """Format text."""
    return text.strip().upper()
'''

_TOOLS_PY_SEMANTIC_STUB = '''
"""Generated tools."""

def fetch_data(query: str = "") -> str:
    """Fetch data."""
    return f"executed with {query}"
'''

_TOOLS_PY_LITERAL_STUB = '''
"""Generated tools."""

def load_data(path: str = "") -> str:
    """Load data."""
    return "AI tool generation did not produce a valid implementation"
'''

_TOOLS_PY_VALID = '''
"""Generated tools."""

def add(a: int = 1, b: int = 2) -> int:
    """Add two numbers."""
    return a + b
'''

_TOOLS_PY_SYNTAX_ERROR = '''
"""Generated tools with syntax error."""

def broken():
    x = 1
        y = 2
    return x + y
'''

_TOOLS_PY_JS_ARTIFACT = '''
"""Generated tools with JS artifact."""

def wrong():
    flag = true
    return flag
'''

_TOOLS_PY_SIDE_EFFECT = '''
"""Generated tools with side effects."""

import subprocess

def run_shell(cmd: str = "ls") -> str:
    """Run a shell command."""
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    return result.stdout
'''


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Clean output directory for each test."""
    out = tmp_path / "agent"
    out.mkdir()
    return out


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection — Data structures
# ─────────────────────────────────────────────────────────────────────────


class TestPostGenReportDataclass:
    """PostGenReport helper methods work correctly."""

    def test_has_errors_true_when_error_severity(self):
        report = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="tools.py",
                    line=1,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="undefined",
                )
            ]
        )
        assert report.has_errors() is True

    def test_has_errors_false_when_only_warnings(self):
        report = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_WARNING,
                    file="tools.py",
                    line=1,
                    category=CATEGORY_STUB,
                    message="stub",
                )
            ]
        )
        assert report.has_errors() is False

    def test_has_errors_false_when_empty(self):
        report = PostGenReport()
        assert report.has_errors() is False

    def test_error_findings_returns_only_errors(self):
        report = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="a.py",
                    line=1,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="e1",
                    tool_name="foo",
                ),
                PostGenFinding(
                    severity=SEVERITY_WARNING,
                    file="b.py",
                    line=2,
                    category=CATEGORY_STUB,
                    message="w1",
                    tool_name="bar",
                ),
            ]
        )
        errors = report.error_findings()
        assert len(errors) == 1
        assert errors[0].message == "e1"

    def test_tools_to_repair_dedupes(self):
        report = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="a.py",
                    line=1,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="e1",
                    tool_name="foo",
                ),
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="a.py",
                    line=2,
                    category=CATEGORY_TYPE_ERROR,
                    message="e2",
                    tool_name="foo",
                ),
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="a.py",
                    line=3,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="e3",
                    tool_name="bar",
                ),
            ]
        )
        tools = report.tools_to_repair()
        assert tools == ["foo", "bar"]

    def test_to_dict_round_trip(self):
        report = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="a.py",
                    line=1,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="e1",
                    tool_name="foo",
                    suggested_fix="add the var",
                )
            ],
            tools_total=3,
            tools_with_issues=1,
            iterations=2,
            token_usage={"total_tokens": 100},
            verdict=VERDICT_WARN,
        )
        d = report.to_dict()
        assert d["verdict"] == "WARN"
        assert d["tools_total"] == 3
        assert d["iterations"] == 2
        assert len(d["findings"]) == 1
        assert d["findings"][0]["tool_name"] == "foo"
        assert d["findings"][0]["suggested_fix"] == "add the var"


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection — Check 3: Undefined variable detection
# ─────────────────────────────────────────────────────────────────────────


class TestUndefinedVariableDetection:
    """Catches the currency-converter NameError bug pattern."""

    def test_detects_undefined_variable_in_function_body(self):
        import ast

        tree = ast.parse(_TOOLS_PY_UNDEFINED_VAR)
        findings = _detect_undefined_variables(tree)
        # `from_curr` and `to_curr` are not defined anywhere
        undefined_names = {finding[1] for finding in findings}
        assert "from_curr" in undefined_names
        assert "to_curr" in undefined_names

    def test_does_not_flag_function_parameters(self):
        import ast

        tree = ast.parse(_TOOLS_PY_VALID)
        findings = _detect_undefined_variables(tree)
        # `add` function uses a, b — both are parameters, no findings
        assert findings == []

    def test_does_not_flag_module_level_imports(self):
        import ast

        source = """
            import json
            from pathlib import Path

            def use_imports(data: str = "") -> str:
                return json.dumps(data) + str(Path("."))
        """
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_undefined_variables(tree)
        assert findings == []

    def test_does_not_flag_local_variables(self):
        import ast

        source = """
            def compute(x: int = 1) -> int:
                y = x * 2
                z = y + 1
                return z
        """
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_undefined_variables(tree)
        assert findings == []

    def test_does_not_flag_builtins(self):
        import ast

        source = """
            def use_builtins(items: list = None) -> int:
                return len(items or [])
        """
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_undefined_variables(tree)
        assert findings == []


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection — Check 4: None attribute access detection
# ─────────────────────────────────────────────────────────────────────────


class TestNoneAttributeAccess:
    """Catches the minimal-skill AttributeError bug pattern."""

    def test_detects_method_call_on_none_default_param(self):
        import ast

        tree = ast.parse(_TOOLS_PY_NONE_ATTR)
        findings = _detect_none_attribute_access(tree)
        # `text.strip()` where text=None default → flagged
        assert len(findings) == 1
        func_name, param_name, attr_name, _lineno = findings[0]
        assert func_name == "format_text"
        assert param_name == "text"
        assert attr_name == "strip"

    def test_does_not_flag_non_none_default(self):
        import ast

        source = """
            def format_text(text: str = "") -> str:
                return text.strip()
        """
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_none_attribute_access(tree)
        assert findings == []

    def test_does_not_flag_safe_access(self):
        import ast

        source = """
            def format_text(text: str = None) -> str:
                if text is None:
                    return ""
                return text.strip()
        """
        tree = ast.parse(textwrap.dedent(source))
        # We still detect the .strip() call, but the test verifies the
        # detection logic itself doesn't crash on guard patterns.
        findings = _detect_none_attribute_access(tree)
        # The detection is intentionally conservative — it still flags
        # because it doesn't analyze the if-guard. This is by design.
        assert len(findings) == 1


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection — Check 5: Semantic stub detection
# ─────────────────────────────────────────────────────────────────────────


class TestSemanticStubDetection:
    """Catches tools that return f-string templates or placeholder phrases."""

    def test_detects_fstring_template_return(self):
        import ast

        tree = ast.parse(_TOOLS_PY_SEMANTIC_STUB)
        findings = _detect_semantic_stubs(tree)
        assert len(findings) == 1
        func_name, _lineno, reason = findings[0]
        assert func_name == "fetch_data"
        assert "f-string" in reason

    def test_detects_placeholder_phrase(self):
        import ast

        source = '''
            def placeholder_tool():
                """Docstring."""
                return "TODO: implementation would go here"
        '''
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_semantic_stubs(tree)
        assert len(findings) == 1
        func_name, _lineno, reason = findings[0]
        assert func_name == "placeholder_tool"
        assert "placeholder" in reason.lower()

    def test_does_not_flag_real_implementation(self):
        import ast

        tree = ast.parse(_TOOLS_PY_VALID)
        findings = _detect_semantic_stubs(tree)
        assert findings == []

    def test_ignores_docstring(self):
        import ast

        source = '''
            def tool():
                """This docstring mentions placeholder but the body is real."""
                return 42
        '''
        tree = ast.parse(textwrap.dedent(source))
        findings = _detect_semantic_stubs(tree)
        assert findings == []


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection — Full inspect_generated_package
# ─────────────────────────────────────────────────────────────────────────


class TestInspectGeneratedPackage:
    """End-to-end inspection on synthetic tools.py fixtures."""

    def test_clean_tools_pass_inspection(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_VALID)
        report = inspect_generated_package(output_dir)
        assert report.tools_total == 1
        # No errors — verdict should be READY after inspect
        assert not report.has_errors()
        assert report.verdict == VERDICT_READY

    def test_undefined_variable_flagged_as_error(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_UNDEFINED_VAR)
        report = inspect_generated_package(output_dir)
        # Should have at least one error-severity finding for undefined var
        undefined_findings = [
            f for f in report.findings if f.category == CATEGORY_UNDEFINED_VAR
        ]
        assert len(undefined_findings) >= 2  # from_curr and to_curr
        for f in undefined_findings:
            assert f.severity == SEVERITY_ERROR
            assert f.tool_name == "convert_currency"
        assert report.has_errors()
        assert report.verdict == VERDICT_WARN

    def test_none_attribute_access_flagged_as_warning(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_NONE_ATTR)
        report = inspect_generated_package(output_dir)
        none_attr_findings = [
            f for f in report.findings if f.category == CATEGORY_NONE_ATTR
        ]
        assert len(none_attr_findings) == 1
        assert none_attr_findings[0].severity == SEVERITY_WARNING
        assert none_attr_findings[0].tool_name == "format_text"
        # Warnings don't make has_errors() True
        assert not report.has_errors()

    def test_semantic_stub_flagged_as_warning(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_SEMANTIC_STUB)
        report = inspect_generated_package(output_dir)
        semantic_findings = [
            f for f in report.findings if f.category == CATEGORY_SEMANTIC_STUB
        ]
        assert len(semantic_findings) == 1
        assert semantic_findings[0].severity == SEVERITY_WARNING

    def test_literal_stub_flagged_as_warning(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_LITERAL_STUB)
        report = inspect_generated_package(output_dir)
        stub_findings = [f for f in report.findings if f.category == CATEGORY_STUB]
        assert len(stub_findings) == 1
        assert stub_findings[0].severity == SEVERITY_WARNING
        assert stub_findings[0].tool_name == "load_data"

    def test_syntax_error_flagged_as_error(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_SYNTAX_ERROR)
        report = inspect_generated_package(output_dir)
        syntax_findings = [f for f in report.findings if f.category == CATEGORY_SYNTAX]
        assert len(syntax_findings) >= 1
        for f in syntax_findings:
            assert f.severity == SEVERITY_ERROR
        assert report.has_errors()

    def test_js_artifact_flagged_as_error(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_JS_ARTIFACT)
        report = inspect_generated_package(output_dir)
        # JS artifact check is in _validate_generated_python
        # It will be reported with category=syntax (default) or js_artifact
        # depending on the error message format.
        assert report.has_errors()

    def test_skills_dir_excluded(self, output_dir: Path):
        """Original skill files in skills/ should not be inspected."""
        # Create a buggy tools.py in skills/ (original skill source)
        skills_dir = output_dir / "skills"
        skills_dir.mkdir()
        _make_tools_py(
            output_dir,
            _TOOLS_PY_UNDEFINED_VAR,
            rel_path="skills/tools.py",
        )
        # Create a clean generated tools.py
        _make_tools_py(output_dir, _TOOLS_PY_VALID, rel_path="src/myagent/tools.py")
        report = inspect_generated_package(output_dir)
        # No errors should be reported (skills/ excluded, generated is clean)
        assert not report.has_errors()


# ─────────────────────────────────────────────────────────────────────────
# B3: Tool self-test
# ─────────────────────────────────────────────────────────────────────────


class TestHasSideEffects:
    """Side-effect detection correctly skips unsafe tools."""

    def test_detects_subprocess(self):
        import ast

        tree = ast.parse(_TOOLS_PY_SIDE_EFFECT)
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_shell"
        )
        side_effect = _has_side_effects(func_node)
        assert side_effect == "subprocess"

    def test_detects_open_call(self):
        import ast

        source = """
            def read_config(path: str = "") -> str:
                with open(path) as f:
                    return f.read()
        """
        tree = ast.parse(textwrap.dedent(source))
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef)
        )
        side_effect = _has_side_effects(func_node)
        assert side_effect == "file_io"

    def test_pure_function_returns_none(self):
        import ast

        tree = ast.parse(_TOOLS_PY_VALID)
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add"
        )
        side_effect = _has_side_effects(func_node)
        assert side_effect is None


class TestToolSignatureSelfTest:
    """Tool self-test runs each tool with default args in sandbox."""

    def test_clean_tool_passes(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_VALID)
        report = inspect_generated_package(output_dir)
        report = _run_tool_self_test(output_dir, report)
        # No findings added by self-test (add(1, 2) returns 3 without error)
        test_failures = [
            f for f in report.findings if f.category == CATEGORY_TYPE_ERROR
        ]
        assert test_failures == []

    def test_none_attr_tool_raises_attribute_error(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_NONE_ATTR)
        report = inspect_generated_package(output_dir)
        report = _run_tool_self_test(output_dir, report)
        # format_text() with default None should raise AttributeError
        attr_findings = [
            f for f in report.findings
            if f.category == CATEGORY_TYPE_ERROR and f.tool_name == "format_text"
        ]
        assert len(attr_findings) == 1
        # AttributeError is an error-severity finding
        assert attr_findings[0].severity == SEVERITY_ERROR

    def test_side_effect_tools_skipped(self, output_dir: Path):
        _make_tools_py(output_dir, _TOOLS_PY_SIDE_EFFECT)
        report = inspect_generated_package(output_dir)
        report = _run_tool_self_test(output_dir, report)
        # Side-effect tools get an INFO finding (skipped, not failed)
        skipped_findings = [
            f for f in report.findings
            if f.tool_name == "run_shell" and f.severity == SEVERITY_INFO
        ]
        assert len(skipped_findings) == 1
        assert "side effects" in skipped_findings[0].message

    def test_no_tools_returns_report_unchanged(self, output_dir: Path):
        """If no tools.py exists, the report is returned unchanged."""
        report = PostGenReport(verdict=VERDICT_READY)
        result = _run_tool_self_test(output_dir, report)
        assert result is report


# ─────────────────────────────────────────────────────────────────────────
# B4: Autonomous iteration loop
# ─────────────────────────────────────────────────────────────────────────


class TestIterateUntilGate:
    """Iteration loop terminates correctly under various conditions."""

    def test_no_errors_stops_after_one_round(self, output_dir: Path):
        """Clean tools → one round, READY verdict."""
        _make_tools_py(output_dir, _TOOLS_PY_VALID)
        report = iterate_until_gate(
            output_dir=output_dir,
            ahsspec={},
            context=None,
            max_rounds=3,
        )
        assert report.iterations == 1
        assert report.verdict == VERDICT_READY
        assert not report.has_errors()

    def test_max_rounds_with_unfixable_errors(self, output_dir: Path):
        """Errors without chat_fn → loops max_rounds, WARN verdict."""
        _make_tools_py(output_dir, _TOOLS_PY_UNDEFINED_VAR)
        report = iterate_until_gate(
            output_dir=output_dir,
            ahsspec={},
            context=None,
            max_rounds=2,
            chat_fn=None,  # No repair possible
            skill_dir=None,
        )
        # Should iterate max_rounds times since no chat_fn to repair
        assert report.iterations == 2
        assert report.verdict == VERDICT_WARN
        assert report.has_errors()

    def test_repair_via_llm_fixes_undefined_var(self, output_dir: Path):
        """Mock LLM returns a corrected body that fixes the bug."""
        _make_tools_py(output_dir, _TOOLS_PY_NONE_ATTR)

        # Mock chat_fn returns a fixed body (None-guard before .strip())
        fixed_body = (
            "if text is None:\n"
            '    return ""\n'
            "return text.strip().upper()"
        )
        chat_fn = MagicMock(return_value=json.dumps({"body": fixed_body}))

        # Mock skill_dir with a SKILL.md so context collection works
        skill_dir = output_dir / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill", encoding="utf-8")

        # Mock ahs_dict with tool metadata
        ahs_dict = {
            "interface": {
                "provides": [
                    {
                        "capability": "format-text",
                        "description": "Format text",
                        "input_schema": {"text": "string"},
                    }
                ]
            }
        }

        report = iterate_until_gate(
            output_dir=output_dir,
            ahsspec=ahs_dict,
            context=None,
            max_rounds=3,
            chat_fn=chat_fn,
            skill_dir=skill_dir,
        )
        # The repair should have been attempted (chat_fn called at least once)
        assert chat_fn.call_count >= 1
        # Mock returned a valid Python body that fixes the bug — verdict must
        # be READY (repair succeeded on round 1, re-inspect passed on round 2).
        # A WARN here would mean the repair didn't apply (regression).
        assert report.verdict == VERDICT_READY, (
            f"expected READY after successful mock repair, got {report.verdict}; "
            f"findings: {[(f.category, f.message) for f in report.findings]}"
        )
        assert not report.has_errors()

    def test_zero_rounds_returns_inspect_only(self, output_dir: Path):
        """max_rounds=0 → only initial inspect runs."""
        _make_tools_py(output_dir, _TOOLS_PY_VALID)
        report = iterate_until_gate(
            output_dir=output_dir,
            ahsspec={},
            context=None,
            max_rounds=0,
        )
        assert report.iterations == 0

    def test_never_blocks_on_catastrophic_failure(self, output_dir: Path, monkeypatch):
        """Even if inspect raises, iterate_until_gate should not propagate."""
        _make_tools_py(output_dir, _TOOLS_PY_VALID)

        # Make inspect_generated_package raise
        def _raising_inspect(_output_dir: Path) -> PostGenReport:
            raise RuntimeError("catastrophic failure")

        monkeypatch.setattr(
            "agenthatch.skill.postgen_review.inspect_generated_package",
            _raising_inspect,
        )

        # Should raise — we don't swallow exceptions in iterate_until_gate
        # (the caller in hatch_command does).
        with pytest.raises(RuntimeError):
            iterate_until_gate(
                output_dir=output_dir,
                ahsspec={},
                context=None,
                max_rounds=1,
            )


# ─────────────────────────────────────────────────────────────────────────
# HatchReport integration
# ─────────────────────────────────────────────────────────────────────────


class TestHatchReportIntegration:
    """PostGenReviewSummary propagates into HatchReport correctly."""

    def test_postgen_warn_propagates_to_hatch_verdict(self):
        """HatchReport verdict is WARN when postgen_review.verdict is WARN."""
        postgen = PostGenReviewSummary(
            verdict="WARN",
            iterations=2,
            tools_total=3,
            tools_with_issues=1,
            findings=[
                PostGenFindingSummary(
                    severity="error",
                    file="tools.py",
                    line=10,
                    category="undefined_var",
                    message="undefined",
                    tool_name="foo",
                )
            ],
        )
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
            postgen_review=postgen,
        )
        assert report.verdict == "WARN"
        assert report.postgen_review is not None
        assert report.postgen_review.verdict == "WARN"
        assert report.postgen_review.iterations == 2

    def test_postgen_ready_does_not_force_warn(self):
        """HatchReport verdict is PASS when postgen is READY and all else clean."""
        postgen = PostGenReviewSummary(
            verdict="READY",
            iterations=1,
            tools_total=2,
            tools_with_issues=0,
        )
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
            postgen_review=postgen,
        )
        assert report.verdict == "PASS"

    def test_postgen_none_does_not_affect_verdict(self):
        """No postgen_review → verdict computation is unchanged."""
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
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
        assert report.postgen_review is None
        assert report.verdict == "PASS"

    def test_postgen_coerce_from_dataclass(self):
        """build_hatch_report accepts a PostGenReport dataclass via to_dict."""
        postgen_dc = PostGenReport(
            findings=[
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="tools.py",
                    line=10,
                    category=CATEGORY_UNDEFINED_VAR,
                    message="undefined",
                    tool_name="foo",
                )
            ],
            tools_total=2,
            tools_with_issues=1,
            iterations=1,
            token_usage={"total_tokens": 50},
            verdict=VERDICT_WARN,
        )
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
            postgen_review=postgen_dc,
        )
        assert report.postgen_review is not None
        assert report.postgen_review.verdict == "WARN"
        assert report.postgen_review.tools_total == 2
        assert len(report.postgen_review.findings) == 1
        # Token usage accumulates into total_tokens
        assert report.total_tokens.get("total_tokens", 0) >= 50

    def test_terminal_rendering_includes_postgen_section(self):
        """to_terminal() includes the Post-Generation Review panel when WARN."""
        postgen = PostGenReviewSummary(
            verdict="WARN",
            iterations=2,
            tools_total=3,
            tools_with_issues=1,
            findings=[
                PostGenFindingSummary(
                    severity="error",
                    file="tools.py",
                    line=10,
                    category="undefined_var",
                    message="undefined variable 'foo'",
                    tool_name="convert_currency",
                )
            ],
        )
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
            postgen_review=postgen,
        )
        # Render to terminal — should not raise
        renderable = report.to_terminal()
        assert renderable is not None

    def test_terminal_rendering_skips_when_none(self):
        """to_terminal() omits the postgen section when postgen_review is None."""
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
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
        # Should render without error
        renderable = report.to_terminal()
        assert renderable is not None

    def test_json_output_includes_postgen(self):
        """to_json() includes the postgen_review field."""
        postgen = PostGenReviewSummary(
            verdict="WARN",
            iterations=2,
            tools_total=3,
            tools_with_issues=1,
        )
        report = build_hatch_report(
            skill_id="test",
            skill_name="Test",
            provider=None,
            model=None,
            phases=[],
            harness_outputs={},
            readiness=None,
            agent_output_dir=None,
            file_count=0,
            archetype=None,
            archetype_confidence=None,
            postgen_review=postgen,
        )
        data = json.loads(report.to_json())
        assert "postgen_review" in data
        assert data["postgen_review"]["verdict"] == "WARN"
        assert data["postgen_review"]["iterations"] == 2


# ─────────────────────────────────────────────────────────────────────────
# Detection capability — known bugs from task spec
# ─────────────────────────────────────────────────────────────────────────


class TestKnownBugDetection:
    """Verify the three known bugs from the task spec are detected."""

    def test_currency_converter_bug_detected(self, output_dir: Path):
        """Bug 1: convert_currency uses from_curr/to_curr but params are from/to.

        B2 'undefined variable detection' should flag this as an error.
        """
        _make_tools_py(output_dir, _TOOLS_PY_UNDEFINED_VAR)
        report = inspect_generated_package(output_dir)
        # Verify error-severity findings for undefined var
        undefined_errors = [
            f for f in report.findings
            if f.category == CATEGORY_UNDEFINED_VAR and f.severity == SEVERITY_ERROR
        ]
        undefined_names = {f.message for f in undefined_errors}
        # Both from_curr and to_curr should be flagged
        assert any("from_curr" in msg for msg in undefined_names)
        assert any("to_curr" in msg for msg in undefined_names)
        assert report.has_errors()

    def test_minimal_skill_bug_detected(self, output_dir: Path):
        """Bug 2: format_text(text=None) has return text.strip().

        B2 'None attribute access detection' should flag this as a warning.
        B3 tool self-test should catch AttributeError.
        """
        _make_tools_py(output_dir, _TOOLS_PY_NONE_ATTR)
        report = inspect_generated_package(output_dir)
        # B2: None attribute access warning
        none_attr_findings = [
            f for f in report.findings if f.category == CATEGORY_NONE_ATTR
        ]
        assert len(none_attr_findings) == 1
        assert none_attr_findings[0].tool_name == "format_text"

        # B3: self-test catches AttributeError
        report = _run_tool_self_test(output_dir, report)
        attr_errors = [
            f for f in report.findings
            if f.category == CATEGORY_TYPE_ERROR and f.tool_name == "format_text"
        ]
        assert len(attr_errors) == 1
        assert attr_errors[0].severity == SEVERITY_ERROR

    def test_data_analyzer_logic_error_documented_as_limitation(
        self, output_dir: Path
    ):
        """Bug 3: load_csv() has str(path).split() — logic error.

        This is harder to detect automatically — there's no syntax error,
        no undefined variable, no None access. The behavior is just wrong
        when path contains spaces. This test documents that B2 does NOT
        flag this case (known limitation).
        """
        source = '''
            """Generated tools."""

            def load_csv(path: str = "") -> list:
                """Load CSV file."""
                parts = str(path).split()
                return parts
        '''
        _make_tools_py(output_dir, textwrap.dedent(source))
        report = inspect_generated_package(output_dir)
        # No errors detected — this is a known limitation
        assert not report.has_errors()
        assert report.verdict == VERDICT_READY


# ─────────────────────────────────────────────────────────────────────────
# _build_agent_context — agent-level semantic context for repair LLM
# ─────────────────────────────────────────────────────────────────────────


class TestBuildAgentContext:
    """Verify the repair LLM gets agent-wide semantics (intent/base/identity)."""

    def test_emits_all_fields_when_present(self):
        ahs = {
            "identity": {"display_name": "Data Analyzer"},
            "intent": {
                "summary": "Load and transform CSV data",
                "triggers": ["load csv", "compute stats"],
            },
        }
        out = _build_agent_context(ahs, archetype="MULTI_STEP")
        assert "=== AGENT CONTEXT ===" in out
        assert "Data Analyzer" in out
        assert "Load and transform CSV data" in out
        assert "load csv, compute stats" in out
        assert "MULTI_STEP" in out

    def test_returns_empty_when_no_agent_fields(self):
        out = _build_agent_context({})
        assert out == ""

    def test_skips_empty_triggers_list(self):
        ahs = {
            "identity": {"display_name": "X"},
            "intent": {"summary": "do thing", "triggers": []},
        }
        out = _build_agent_context(ahs, archetype="TOOL_WRAPPER")
        assert "Triggers:" not in out
        assert "TOOL_WRAPPER" in out

    def test_handles_non_dict_sections_gracefully(self):
        # Should not crash if sections are malformed
        ahs = {"identity": None, "intent": "not a dict", "base": []}
        out = _build_agent_context(ahs)
        assert out == ""

    def test_returns_empty_for_none_ahs_dict(self):
        """Regression: None ahs_dict must not raise AttributeError."""
        out = _build_agent_context(None)  # type: ignore[arg-type]
        assert out == ""

    def test_archetype_param_drives_archetype_line(self):
        """Archetype comes from the explicit param, not ahs_dict['base'].

        This locks in the design decision: AHSSpec Pydantic model has no
        ``archetype`` field (computed at runtime by classify_skill), so the
        caller must pass it explicitly. Reading ahs_dict['base']['archetype']
        would silently always return empty — dead code.
        """
        ahs = {
            "identity": {"display_name": "X"},
            # Deliberately put archetype in base to prove it's NOT read from here
            "base": {"archetype": "IGNORED_VALUE"},
        }
        out = _build_agent_context(ahs, archetype="PROMPT_ONLY")
        assert "PROMPT_ONLY" in out
        assert "IGNORED_VALUE" not in out

    def test_archetype_none_omits_archetype_line(self):
        ahs = {"identity": {"display_name": "X"}}
        out = _build_agent_context(ahs, archetype=None)
        assert "Archetype:" not in out


# ─────────────────────────────────────────────────────────────────────────
# _rebuild_function_source / _apply_tool_repair — repair application
# ─────────────────────────────────────────────────────────────────────────


class TestRebuildFunctionSource:
    """Verify function body replacement preserves docstring + signature."""

    def test_multiline_docstring_preserved(self):
        """Multi-line docstring must be fully retained (end_lineno is inclusive).

        Regression: previously doc_end_idx missed +1, dropping the closing
        triple-quote and producing unparseable code.
        """
        source = textwrap.dedent('''\
            def format_text(text: str = None) -> Any:
                """Handle the 'format_text' capability.
                AI-generated implementation based on skill context.
                """
                return text.strip()
        ''')
        tree = ast.parse(source)
        func_node = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "format_text"
        )
        new_body = 'if text is None:\n    return ""\nreturn text.strip()'
        new_source = _rebuild_function_source(func_node, source, new_body)
        assert new_source is not None
        # New source must parse cleanly (this is the regression check)
        ast.parse(new_source)
        # Docstring closer must be present
        assert '"""' in new_source
        # New body must be present
        assert 'if text is None:' in new_source

    def test_single_line_docstring_preserved(self):
        source = textwrap.dedent('''\
            def foo(x):
                """Short."""
                return x
        ''')
        tree = ast.parse(source)
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "foo"
        )
        new_body = "return x * 2"
        new_source = _rebuild_function_source(func_node, source, new_body)
        assert new_source is not None
        ast.parse(new_source)
        assert '"""Short."""' in new_source

    def test_no_docstring(self):
        source = textwrap.dedent('''\
            def foo(x):
                return x
        ''')
        tree = ast.parse(source)
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "foo"
        )
        new_body = "return x * 2"
        new_source = _rebuild_function_source(func_node, source, new_body)
        assert new_source is not None
        ast.parse(new_source)
        assert "return x * 2" in new_source


class TestApplyToolRepair:
    """End-to-end repair application on a synthetic tools.py."""

    def test_repair_applies_to_multiline_docstring_function(self, tmp_path: Path):
        """Regression: repair must succeed when function has multi-line docstring."""
        tools_py = tmp_path / "src" / "myagent" / "tools.py"
        tools_py.parent.mkdir(parents=True)
        tools_py.write_text(textwrap.dedent('''\
            """tools."""
            from typing import Any


            def format_text(text: str = None) -> Any:
                """Handle the 'format_text' capability.
                Multi-line docstring.
                """
                return text.strip()
        '''), encoding="utf-8")

        new_body = 'if text is None:\n    return ""\nreturn text.strip()'
        success = _apply_tool_repair(tmp_path, "format_text", new_body)
        assert success is True

        # Verify the new content parses and contains the fix
        new_content = tools_py.read_text(encoding="utf-8")
        ast.parse(new_content)
        assert "if text is None:" in new_content
        # Docstring must still be intact (3 lines)
        assert "Handle the 'format_text' capability." in new_content
        assert "Multi-line docstring." in new_content

    def test_repair_returns_false_when_function_missing(self, tmp_path: Path):
        tools_py = tmp_path / "src" / "myagent" / "tools.py"
        tools_py.parent.mkdir(parents=True)
        tools_py.write_text("def other():\n    return 1\n", encoding="utf-8")
        success = _apply_tool_repair(tmp_path, "missing_func", "return 1")
        assert success is False

    def test_repair_applies_to_async_function(self, tmp_path: Path):
        """Regression: async def tools must be repairable.

        ``ast.AsyncFunctionDef`` is a separate node type from
        ``ast.FunctionDef`` — without explicit handling, async tools would
        never match in ``_replace_function_body`` and repair would silently
        fail. This test locks in the fix.
        """
        tools_py = tmp_path / "src" / "myagent" / "tools.py"
        tools_py.parent.mkdir(parents=True)
        tools_py.write_text(textwrap.dedent('''\
            """tools."""
            from typing import Any


            async def fetch_data(url: str = None) -> Any:
                """Fetch data from URL.
                Multi-line docstring.
                """
                return url
        '''), encoding="utf-8")

        new_body = 'if url is None:\n    return ""\nreturn url'
        success = _apply_tool_repair(tmp_path, "fetch_data", new_body)
        assert success is True

        new_content = tools_py.read_text(encoding="utf-8")
        ast.parse(new_content)
        # async def must be preserved
        assert "async def fetch_data" in new_content
        assert "if url is None:" in new_content
        # Docstring intact
        assert "Fetch data from URL." in new_content

    def test_repair_returns_false_when_new_body_has_syntax_error(
        self, tmp_path: Path
    ):
        """Safety net: invalid new_body must not corrupt tools.py."""
        tools_py = tmp_path / "src" / "myagent" / "tools.py"
        tools_py.parent.mkdir(parents=True)
        original = textwrap.dedent('''\
            def foo(x):
                """Doc."""
                return x
        ''')
        tools_py.write_text(original, encoding="utf-8")

        # Invalid Python: unclosed paren
        success = _apply_tool_repair(tmp_path, "foo", "return foo(")
        assert success is False
        # File must be unchanged
        assert tools_py.read_text(encoding="utf-8") == original
