"""Post-generation review (Phase 3.5) — B2/B3/B4 self-healing.

Inspects generated agent code, self-tests tool signatures, and iterates
LLM-driven repair until the quality gate passes (max 3 rounds).

Design (see docs/agenthatch-v0.9.22-postgen-review-design.md):

- B2 ``inspect_generated_package()`` — reuses
  :meth:`GenerateEngine._validate_generated_python` (AST syntax + JS
  artifact detection) and :meth:`GenerateEngine._check_tool_stubs`
  (literal stub detection), then adds:
    * undefined-variable detection (catches ``NameError`` bugs)
    * None attribute-access detection (catches ``AttributeError`` bugs)
    * semantic stub detection (catches placeholder/template returns)

- B3 ``test_tool_signatures()`` — imports the generated ``tools.py`` in a
  subprocess sandbox (10s timeout) and calls each tool with default
  params. Catches ``NameError``/``TypeError``/``AttributeError``.

- B4 ``iterate_until_gate()`` — inspect → test → repair → re-inspect
  loop. Max 3 rounds. Never blocks (WARN verdict proceeds).

Follows the "never block" philosophy (see
:meth:`runtime_readiness_gate` in readiness.py:302): self-review
produces READY/WARN, never BLOCK. The agent is always generated.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

CATEGORY_SYNTAX = "syntax"
CATEGORY_JS_ARTIFACT = "js_artifact"
CATEGORY_STUB = "stub"
CATEGORY_SEMANTIC_STUB = "semantic_stub"
CATEGORY_UNDEFINED_VAR = "undefined_var"
CATEGORY_NONE_ATTR = "none_attr_access"
CATEGORY_TYPE_ERROR = "type_error"
CATEGORY_TEST_FAILURE = "test_failure"

VERDICT_READY = "READY"
VERDICT_WARN = "WARN"
VERDICT_NEEDS_HUMAN = "NEEDS_HUMAN"

# Subprocess timeout for tool self-test (seconds)
TOOL_TEST_TIMEOUT = 10

# Phrases that indicate a semantic stub
_PLACEHOLDER_PHRASES: tuple[str, ...] = (
    "would go here",
    "placeholder",
    "not implemented",
    "todo:",
    "fixme:",
    "executed with",
    "would be",
    "stub",
)

# Names that are always available in Python functions
_BUILTIN_NAMES: set[str] = set(dir(builtins)) | {
    "True", "False", "None", "__name__", "__file__", "__doc__",
}


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PostGenFinding:
    """A single inspection or test finding."""

    severity: str
    file: str
    line: int
    category: str
    message: str
    tool_name: str | None = None
    suggested_fix: str | None = None


@dataclass
class PostGenReport:
    """Aggregate report from inspect + test + iterate."""

    findings: list[PostGenFinding] = field(default_factory=list)
    tools_total: int = 0
    tools_with_issues: int = 0
    iterations: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    verdict: str = VERDICT_WARN

    def has_errors(self) -> bool:
        """True if any finding has severity 'error'."""
        return any(f.severity == SEVERITY_ERROR for f in self.findings)

    def error_findings(self) -> list[PostGenFinding]:
        """Return only error-severity findings."""
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    def tools_to_repair(self) -> list[str]:
        """Return unique tool names that have error findings."""
        seen: set[str] = set()
        result: list[str] = []
        for f in self.findings:
            if f.severity == SEVERITY_ERROR and f.tool_name and f.tool_name not in seen:
                seen.add(f.tool_name)
                result.append(f.tool_name)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for HatchReport integration)."""
        return {
            "findings": [
                {
                    "severity": f.severity,
                    "file": f.file,
                    "line": f.line,
                    "category": f.category,
                    "message": f.message,
                    "tool_name": f.tool_name,
                    "suggested_fix": f.suggested_fix,
                }
                for f in self.findings
            ],
            "tools_total": self.tools_total,
            "tools_with_issues": self.tools_with_issues,
            "iterations": self.iterations,
            "token_usage": dict(self.token_usage),
            "verdict": self.verdict,
        }


# ─────────────────────────────────────────────────────────────────────────
# B2: Inspection module
# ─────────────────────────────────────────────────────────────────────────


def inspect_generated_package(output_dir: Path) -> PostGenReport:
    """Inspect the generated agent package for code quality issues.

    Reuses :meth:`GenerateEngine._validate_generated_python` (AST syntax +
    JS artifact detection) and :meth:`GenerateEngine._check_tool_stubs`
    (literal stub detection), then adds:

    - Undefined variable detection (catches NameError bugs)
    - None attribute access detection (catches AttributeError bugs)
    - Semantic stub detection (catches placeholder/template returns)
    """
    from agenthatch.generate.engine import GenerateEngine

    report = PostGenReport(verdict=VERDICT_WARN)
    skills_dir = output_dir / "skills"

    # ── Check 1: AST syntax + JS artifacts (reuse engine logic) ─────────
    validation_errors = GenerateEngine._validate_generated_python(output_dir)
    for err in validation_errors:
        # Parse error format: "file:line: message"
        match = re.match(r"(.+?):(\d+):\s*(.+)", err)
        if match:
            file_str, line_str, message = match.groups()
            category = CATEGORY_SYNTAX if "SyntaxError" in err else CATEGORY_JS_ARTIFACT
            report.findings.append(
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file=file_str,
                    line=int(line_str),
                    category=category,
                    message=message,
                )
            )
        else:
            report.findings.append(
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file="<unknown>",
                    line=0,
                    category=CATEGORY_SYNTAX,
                    message=err,
                )
            )

    # ── Check 2: Literal stubs (reuse engine logic) ────────────────────
    stub_tools = GenerateEngine._check_tool_stubs(output_dir)
    tools_py_relpath = _find_tools_py_relpath(output_dir)
    for tool_name in stub_tools:
        report.findings.append(
            PostGenFinding(
                severity=SEVERITY_WARNING,
                file=tools_py_relpath,
                line=0,
                category=CATEGORY_STUB,
                message=(
                    f"Tool '{tool_name}' is a non-functional stub "
                    "(pass/NotImplementedError)"
                ),
                tool_name=tool_name,
                suggested_fix=(
                    "Re-run hatch with working LLM provider, or implement manually"
                ),
            )
        )

    # ── Find tools.py for deeper inspection ─────────────────────────────
    tools_files: list[tuple[Path, str]] = []  # (abs_path, rel_path)
    for py_file in output_dir.rglob("tools.py"):
        # Exclude skills/ directory (original source)
        if skills_dir in py_file.parents or py_file.parent == skills_dir:
            continue
        rel = str(py_file.relative_to(output_dir))
        tools_files.append((py_file, rel))

    # ── Counts and deeper checks ───────────────────────────────────────
    for tools_py, rel_path in tools_files:
        try:
            content = tools_py.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, OSError, UnicodeDecodeError):
            # Already reported by Check 1
            continue

        # Count top-level functions (skip configure_mcp_auth helper)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name != "configure_mcp_auth"
            ):
                report.tools_total += 1

        # ── Check 3: Undefined variable detection ──────────────────────
        undefined_findings = _detect_undefined_variables(tree)
        for func_name, var_name, lineno in undefined_findings:
            report.findings.append(
                PostGenFinding(
                    severity=SEVERITY_ERROR,
                    file=rel_path,
                    line=lineno,
                    category=CATEGORY_UNDEFINED_VAR,
                    message=(
                        f"Tool '{func_name}' references undefined variable "
                        f"'{var_name}'. This will cause NameError at runtime."
                    ),
                    tool_name=func_name,
                    suggested_fix=(
                        f"Add '{var_name}' as a function parameter, define it "
                        f"as a local variable, or import it."
                    ),
                )
            )

        # ── Check 4: None attribute access detection ───────────────────
        none_attr_findings = _detect_none_attribute_access(tree)
        for func_name, param_name, attr_name, lineno in none_attr_findings:
            report.findings.append(
                PostGenFinding(
                    severity=SEVERITY_WARNING,
                    file=rel_path,
                    line=lineno,
                    category=CATEGORY_NONE_ATTR,
                    message=(
                        f"Tool '{func_name}' calls .{attr_name}() on parameter "
                        f"'{param_name}' which has default None. This will raise "
                        f"AttributeError when called with the default value."
                    ),
                    tool_name=func_name,
                    suggested_fix=(
                        f"Add a None check: 'if {param_name} is None: return ...' "
                        f"before accessing .{attr_name}()."
                    ),
                )
            )

        # ── Check 5: Semantic stub detection ───────────────────────────
        semantic_findings = _detect_semantic_stubs(tree)
        for func_name, lineno, reason in semantic_findings:
            report.findings.append(
                PostGenFinding(
                    severity=SEVERITY_WARNING,
                    file=rel_path,
                    line=lineno,
                    category=CATEGORY_SEMANTIC_STUB,
                    message=(
                        f"Tool '{func_name}' may be a semantic stub: {reason}. "
                        f"It returns a placeholder without performing real work."
                    ),
                    tool_name=func_name,
                    suggested_fix=(
                        "Generate a real implementation that performs the tool's "
                        "intended work, not just returns a template string."
                    ),
                )
            )

    # Count tools with issues
    tools_with_issues = {f.tool_name for f in report.findings if f.tool_name}
    report.tools_with_issues = len(tools_with_issues)

    # Initial verdict (B4 will finalize)
    report.verdict = VERDICT_WARN if report.has_errors() else VERDICT_READY
    return report


def _find_tools_py_relpath(output_dir: Path) -> str:
    """Find relative path of tools.py (for findings)."""
    skills_dir = output_dir / "skills"
    for py_file in output_dir.rglob("tools.py"):
        if skills_dir in py_file.parents or py_file.parent == skills_dir:
            continue
        try:
            return str(py_file.relative_to(output_dir))
        except ValueError:
            return str(py_file)
    return "tools.py"


def _detect_undefined_variables(
    tree: ast.Module,
) -> list[tuple[str, str, int]]:
    """For each top-level FunctionDef, find undefined Name references.

    A Name is "undefined" if it is not:
    - A function parameter
    - A local variable (assigned within the function)
    - A module-level import or constant
    - A Python builtin

    Returns list of (func_name, var_name, lineno).
    """
    findings: list[tuple[str, str, int]] = []

    # Collect module-level imports and assignments
    module_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                module_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name
        param_names: set[str] = {
            arg.arg for arg in node.args.args + node.args.kwonlyargs
        }
        if node.args.vararg:
            param_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            param_names.add(node.args.kwarg.arg)

        # Collect local names defined within the function body
        local_names: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        local_names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                local_names.add(elt.id)
            elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                local_names.add(sub.target.id)
            elif isinstance(
                sub, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
            ):
                local_names.add(sub.name)
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    local_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    local_names.add(alias.asname or alias.name)
            elif isinstance(sub, ast.For) and isinstance(sub.target, ast.Name):
                local_names.add(sub.target.id)
            elif isinstance(sub, ast.For) and isinstance(sub.target, ast.Tuple):
                for elt in sub.target.elts:
                    if isinstance(elt, ast.Name):
                        local_names.add(elt.id)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                local_names.add(sub.name)
            elif isinstance(sub, ast.With):
                for item in sub.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        local_names.add(item.optional_vars.id)
            elif isinstance(
                sub, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
            ):
                for gen in sub.generators:
                    if isinstance(gen.target, ast.Name):
                        local_names.add(gen.target.id)
                    elif isinstance(gen.target, ast.Tuple):
                        for elt in gen.target.elts:
                            if isinstance(elt, ast.Name):
                                local_names.add(elt.id)
            elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                local_names.add(sub.target.id)
            elif isinstance(sub, ast.MatchAs) and sub.name is not None:
                local_names.add(sub.name)
            elif isinstance(sub, ast.MatchStar) and sub.name is not None:
                local_names.add(sub.name)
            elif isinstance(sub, ast.MatchMapping) and sub.rest is not None:
                local_names.add(sub.rest)

        available = param_names | local_names | module_names | _BUILTIN_NAMES

        # Find Name references in Load context that aren't available
        seen_pairs: set[tuple[str, str]] = set()  # dedupe (func, var)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in available:
                    pair = (func_name, sub.id)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        findings.append((func_name, sub.id, sub.lineno))

    return findings


def _detect_none_attribute_access(
    tree: ast.Module,
) -> list[tuple[str, str, str, int]]:
    """Detect ``param.method()`` where ``param`` has default ``None``.

    Returns list of (func_name, param_name, attr_name, lineno).
    """
    findings: list[tuple[str, str, str, int]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name
        none_params: set[str] = set()

        # Positional args with defaults
        args = node.args.args
        defaults = node.args.defaults
        n_args = len(args)
        n_defaults = len(defaults)
        for i, default in enumerate(defaults):
            arg_idx = n_args - n_defaults + i
            if 0 <= arg_idx < n_args:
                if _is_none_constant(default):
                    none_params.add(args[arg_idx].arg)

        # Kwonly args
        for i, kw_default in enumerate(node.args.kw_defaults):
            if kw_default is not None and _is_none_constant(kw_default):
                if i < len(node.args.kwonlyargs):
                    none_params.add(node.args.kwonlyargs[i].arg)

        if not none_params:
            continue

        # Find ``param.method()`` patterns in function body
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                attr = sub.func
                if isinstance(attr.value, ast.Name) and attr.value.id in none_params:
                    findings.append((func_name, attr.value.id, attr.attr, attr.lineno))

    return findings


def _is_none_constant(node: ast.AST) -> bool:
    """True if node is ``ast.Constant`` with value ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def _detect_semantic_stubs(
    tree: ast.Module,
) -> list[tuple[str, int, str]]:
    """Detect function bodies that only return placeholder strings.

    Returns list of (func_name, lineno, reason).
    """
    findings: list[tuple[str, int, str]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name
        body = list(node.body)

        # Skip docstring
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        if not body:
            continue

        # Check if body is just a single return statement
        if len(body) == 1 and isinstance(body[0], ast.Return):
            ret = body[0]
            if ret.value is None:
                continue

            # Pattern 1: return f"..." (f-string template only)
            if isinstance(ret.value, ast.JoinedStr):
                findings.append(
                    (func_name, ret.value.lineno, "returns f-string template only")
                )
                continue

            # Pattern 2: return "..." with placeholder phrase
            if isinstance(ret.value, ast.Constant) and isinstance(
                ret.value.value, str
            ):
                text = ret.value.value.lower()
                for phrase in _PLACEHOLDER_PHRASES:
                    if phrase in text:
                        findings.append(
                            (
                                func_name,
                                ret.value.lineno,
                                f"contains placeholder phrase '{phrase}'",
                            )
                        )
                        break
                continue

    return findings


# ─────────────────────────────────────────────────────────────────────────
# B3: Tool self-test
# ─────────────────────────────────────────────────────────────────────────


# Python script template for testing a single tool in a subprocess
_TOOL_TEST_SCRIPT = """\
import importlib.util
import sys
import traceback

tools_path = {tools_path!r}
tool_name = {tool_name!r}

spec = importlib.util.spec_from_file_location("_tools_under_test", tools_path)
if spec is None or spec.loader is None:
    print("IMPORT_ERROR: cannot load module spec")
    sys.exit(2)

mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print("IMPORT_ERROR:", type(e).__name__, str(e)[:300])
    sys.exit(3)

fn = getattr(mod, tool_name, None)
if fn is None or not callable(fn):
    print("NOT_FOUND")
    sys.exit(4)

# Call with default args (no args = use defaults)
try:
    result = fn()
    print("OK", repr(result)[:200])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300])
    sys.exit(5)
"""


def test_tool_signatures(
    output_dir: Path, report: PostGenReport
) -> PostGenReport:
    """Call each tool once with default params, capture exceptions.

    Strategy:
    - Skip PROMPT_ONLY skills (no tools)
    - Skip tools with subprocess/network/file-IO calls (side effects)
    - Run each remaining tool in a subprocess sandbox with 10s timeout
    - Catch NameError/TypeError/AttributeError → log as error finding
    - Other exceptions (e.g. ValueError on bad default args) → INFO

    Uses the agenthatch_core Sandbox executor for isolation.
    """
    from agenthatch_core.sandbox.executor import Sandbox, SandboxConfig

    skills_dir = output_dir / "skills"

    # Find tools.py
    tools_files: list[Path] = []
    for py_file in output_dir.rglob("tools.py"):
        if skills_dir in py_file.parents or py_file.parent == skills_dir:
            continue
        tools_files.append(py_file)

    if not tools_files:
        return report

    sandbox = Sandbox(config=SandboxConfig(timeout=f"{TOOL_TEST_TIMEOUT}s"))

    for tools_py in tools_files:
        try:
            rel_path = str(tools_py.relative_to(output_dir))
        except ValueError:
            rel_path = str(tools_py)
        try:
            content = tools_py.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue

        # Identify all tool functions (skip MCP auth helper)
        tool_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "configure_mcp_auth"
        ]

        for func_node in tool_funcs:
            func_name = func_node.name

            # Skip tools with side effects (subprocess, network, file IO)
            side_effect = _has_side_effects(func_node)
            if side_effect:
                report.findings.append(
                    PostGenFinding(
                        severity=SEVERITY_INFO,
                        file=rel_path,
                        line=func_node.lineno,
                        category=CATEGORY_TEST_FAILURE,
                        message=(
                            f"Tool '{func_name}' has side effects ({side_effect}), "
                            f"self-test skipped"
                        ),
                        tool_name=func_name,
                    )
                )
                continue

            # Run in sandbox
            script = _TOOL_TEST_SCRIPT.format(
                tools_path=str(tools_py),
                tool_name=func_name,
            )
            result = sandbox.run(
                [sys.executable, "-c", script],
                timeout=TOOL_TEST_TIMEOUT,
            )

            stdout = result.stdout or ""
            stdout_stripped = stdout.strip()

            if result.timed_out:
                report.findings.append(
                    PostGenFinding(
                        severity=SEVERITY_WARNING,
                        file=rel_path,
                        line=func_node.lineno,
                        category=CATEGORY_TEST_FAILURE,
                        message=(
                            f"Tool '{func_name}' timed out after {TOOL_TEST_TIMEOUT}s "
                            f"during self-test"
                        ),
                        tool_name=func_name,
                    )
                )
                continue

            # Check for OK marker (tool ran successfully)
            if stdout_stripped.startswith("OK"):
                continue

            if stdout_stripped.startswith("FAIL:"):
                # Tool raised an exception
                exc_info = stdout_stripped[len("FAIL:"):].strip()
                parts = exc_info.split(" ", 1)
                exc_type = parts[0] if parts else "Exception"
                exc_msg = parts[1] if len(parts) > 1 else ""

                if exc_type in ("NameError", "AttributeError", "TypeError"):
                    severity = SEVERITY_ERROR
                    category = CATEGORY_TYPE_ERROR
                else:
                    # Value errors etc. may be expected for default args
                    severity = SEVERITY_INFO
                    category = CATEGORY_TEST_FAILURE

                report.findings.append(
                    PostGenFinding(
                        severity=severity,
                        file=rel_path,
                        line=func_node.lineno,
                        category=category,
                        message=(
                            f"Tool '{func_name}' raised {exc_type} when called "
                            f"with default args: {exc_msg[:200]}"
                        ),
                        tool_name=func_name,
                        suggested_fix=_suggest_fix_for_exception(exc_type, exc_msg),
                    )
                )
            elif stdout_stripped.startswith("IMPORT_ERROR"):
                report.findings.append(
                    PostGenFinding(
                        severity=SEVERITY_ERROR,
                        file=rel_path,
                        line=func_node.lineno,
                        category=CATEGORY_TEST_FAILURE,
                        message=(
                            f"Tool module failed to import during self-test: "
                            f"{stdout_stripped}"
                        ),
                        tool_name=func_name,
                    )
                )
            elif stdout_stripped.startswith("NOT_FOUND"):
                # Tool function doesn't exist — skip silently
                continue
            # else: empty/unknown output — skip silently

    # Recompute tools_with_issues
    tools_with_issues = {f.tool_name for f in report.findings if f.tool_name}
    report.tools_with_issues = len(tools_with_issues)

    # Re-evaluate verdict
    report.verdict = VERDICT_WARN if report.has_errors() else VERDICT_READY
    return report


def _has_side_effects(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Detect if a function has side effects (subprocess, network, file IO).

    Returns the side-effect kind as a string, or ``None`` if the function
    appears pure (safe to self-test).
    """
    for sub in ast.walk(func_node):
        # subprocess.run / subprocess.Popen
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            attr = sub.func
            if isinstance(attr.value, ast.Name):
                if attr.value.id == "subprocess":
                    return "subprocess"
                if attr.value.id == "requests":
                    return "network"
                if attr.value.id == "urllib":
                    return "network"
                if attr.value.id == "httpx":
                    return "network"
        # open(...) calls (file IO)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id == "open":
                return "file_io"
            if sub.func.id in ("socket",):
                return "network"

    return None


def _suggest_fix_for_exception(exc_type: str, exc_msg: str) -> str:
    """Suggest a fix based on the exception type."""
    if exc_type == "NameError":
        match = re.search(r"name '(\w+)' is not defined", exc_msg)
        if match:
            var = match.group(1)
            return f"Define '{var}' as a parameter, local variable, or import."
        return "Define the missing variable before using it."
    if exc_type == "AttributeError":
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", exc_msg)
        if match:
            obj, attr = match.group(1), match.group(2)
            return (
                f"Add a None check before accessing '.{attr}' on '{obj}', "
                f"or use a different attribute."
            )
        return "Add a None check before accessing attributes."
    if exc_type == "TypeError":
        return "Check argument types and counts in function calls."
    return "Investigate the exception and add proper handling."


# ─────────────────────────────────────────────────────────────────────────
# B4: Autonomous iteration loop
# ─────────────────────────────────────────────────────────────────────────


def iterate_until_gate(
    output_dir: Path,
    ahsspec: Any,
    context: Any,
    max_rounds: int = 3,
    *,
    chat_fn: Any | None = None,
    skill_dir: Path | None = None,
    archetype: str | None = None,
) -> PostGenReport:
    """inspect → test → fix → re-inspect loop (max 3 rounds).

    Each round only fixes tools with error-severity findings, not full
    regeneration. Reuses :meth:`GenerateEngine._ai_generate_tool_impls`
    LLM call patterns for repair (system prompt structure, JSON parsing,
    indent normalization).

    Never blocks: even if all rounds fail, returns WARN verdict and
    the agent is still generated.
    """
    ahs_dict = _coerce_ahs_dict(ahsspec)

    total_tokens: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    final_report: PostGenReport | None = None

    for round_num in range(1, max_rounds + 1):
        # 1. Inspect
        report = inspect_generated_package(output_dir)

        # 2. Test tools
        report = test_tool_signatures(output_dir, report)

        report.iterations = round_num
        report.token_usage = dict(total_tokens)
        final_report = report

        logger.info(
            "postgen_review round %d: %d findings (%d errors, %d warnings)",
            round_num,
            len(report.findings),
            len(report.error_findings()),
            sum(1 for f in report.findings if f.severity == SEVERITY_WARNING),
        )

        # 3. Quality gate check
        if not report.has_errors():
            report.verdict = VERDICT_READY
            break

        # 4. Repair (need chat_fn + skill_dir + ahs_dict)
        if not chat_fn or not skill_dir or not ahs_dict:
            # Cannot repair; mark as WARN if errors remain
            report.verdict = VERDICT_WARN
            continue

        tools_to_repair = report.tools_to_repair()
        if not tools_to_repair:
            # Errors without associated tools — can't surgically repair
            report.verdict = VERDICT_WARN
            continue

        # Collect findings for each tool to repair
        tool_findings: dict[str, list[PostGenFinding]] = {}
        for f in report.findings:
            if f.tool_name in tools_to_repair:
                tool_findings.setdefault(f.tool_name, []).append(f)

        # 5. Regenerate tool implementations
        regenerated: dict[str, str] = {}
        for tool_name in tools_to_repair:
            new_body, repair_tokens = _regenerate_tool_via_llm(
                ahs_dict=ahs_dict,
                skill_dir=skill_dir,
                tool_name=tool_name,
                findings=tool_findings.get(tool_name, []),
                chat_fn=chat_fn,
                archetype=archetype,
            )
            for k in total_tokens:
                total_tokens[k] += repair_tokens.get(k, 0)

            if new_body:
                applied = _apply_tool_repair(output_dir, tool_name, new_body)
                if applied:
                    regenerated[tool_name] = new_body
                    logger.info(
                        "postgen_review: regenerated tool '%s' in round %d",
                        tool_name,
                        round_num,
                    )

        if not regenerated:
            # No tools could be repaired — stop iterating
            logger.warning(
                "postgen_review: no tools could be repaired in round %d, stopping",
                round_num,
            )
            report.verdict = VERDICT_WARN
            break
        # Loop continues — re-inspect with the new code

    # Finalize
    if final_report is None:
        # No iterations ran (max_rounds=0)
        final_report = inspect_generated_package(output_dir)
        final_report = test_tool_signatures(output_dir, final_report)
        final_report.iterations = 0

    final_report.token_usage = dict(total_tokens)

    if final_report.has_errors():
        final_report.verdict = VERDICT_WARN
    else:
        final_report.verdict = VERDICT_READY

    return final_report


def _coerce_ahs_dict(ahsspec: Any) -> dict[str, Any]:
    """Coerce ahsspec (model object or dict) to a plain dict."""
    if isinstance(ahsspec, dict):
        return dict(ahsspec)
    if hasattr(ahsspec, "model_dump"):
        try:
            dumped = ahsspec.model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    return {}


def _regenerate_tool_via_llm(
    ahs_dict: dict[str, Any],
    skill_dir: Path,
    tool_name: str,
    findings: list[PostGenFinding],
    chat_fn: Any,
    archetype: str | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Use LLM to regenerate a single tool's body.

    Reuses the same prompt structure and JSON parsing as
    :meth:`GenerateEngine._ai_generate_tool_impls`, but targeted at one
    tool with bug context.

    Returns (new_body, token_usage). new_body is None on failure.
    """
    # Collect context files (subset of _ai_generate_tool_impls)
    context_block = _collect_skill_context(skill_dir)
    if not context_block:
        logger.warning("repair tool '%s': no skill context collected", tool_name)
        return None, {}

    # Build tool metadata block for the failing tool
    tool_metadata = _find_tool_metadata(ahs_dict, tool_name)
    if not tool_metadata:
        # Debug: log available capability names to diagnose mismatch
        interface = ahs_dict.get("interface", {}) or {}
        provides = interface.get("provides", []) or []
        cap_names = [
            cap.get("capability", "") for cap in provides if isinstance(cap, dict)
        ]
        logger.warning(
            "repair tool '%s': no metadata found in ahs_dict. "
            "Available capabilities: %s",
            tool_name,
            cap_names,
        )
        return None, {}

    # Build findings description
    findings_desc = "\n".join(
        f"- [{f.category}] line {f.line}: {f.message}"
        + (f"\n  Suggested fix: {f.suggested_fix}" if f.suggested_fix else "")
        for f in findings
    )

    system_prompt = (
        "You are an expert Python code generator for agent tool implementations. "
        "A tool implementation has bugs detected by automated inspection. "
        "Regenerate the tool's function body to fix the bugs.\n\n"
        "Rules:\n"
        "- Generate ONLY the function body (code INSIDE the function, after the "
        "  signature and docstring).\n"
        "- Start your code at indent 0 (no leading spaces).\n"
        "- Use the EXACT parameter names from the tool definition.\n"
        "- Do NOT use **kwargs — use exact parameter names.\n"
        "- Include proper error handling and return meaningful results.\n"
        "- Import only stdlib or packages from the skill context.\n"
        "- Honor the AGENT CONTEXT: the repair must fit the agent's intent, "
        "  triggers, and archetype (e.g. MULTI_STEP agents should keep state "
        "  across calls; PROMPT_ONLY agents have no tools to repair).\n"
        '- Output JSON: {"body": "<function body code>"}\n'
    )

    # Build agent-level context (intent + base + identity) so the LLM has the
    # same semantic view as the original _ai_generate_tool_impls call. Without
    # this, repairs degrade to single-tool reasoning and miss agent-wide
    # semantics (e.g. triggers, archetype constraints).
    agent_context = _build_agent_context(ahs_dict, archetype=archetype)

    user_prompt = (
        f"Regenerate the Python function body for tool '{tool_name}'.\n\n"
        f"{agent_context}"
        f"=== TOOL DEFINITION ===\n"
        f"Name: {tool_metadata.get('name', tool_name)}\n"
        f"Func name: {tool_metadata.get('func_name', tool_name)}\n"
        f"Description: {tool_metadata.get('description', '')}\n"
        f"Backend: {tool_metadata.get('backend_kind', 'none')}\n"
        f"Params: {tool_metadata.get('params', [])}\n\n"
        f"=== SKILL CONTEXT ===\n{context_block}\n\n"
        f"=== DETECTED BUGS ===\n{findings_desc}\n\n"
        f'Generate a corrected function body. Output JSON: {{"body": "..."}}\n'
    )

    try:
        response = chat_fn(system_prompt, user_prompt)
    except Exception as e:
        logger.warning("LLM repair call failed for tool '%s': %s", tool_name, e)
        return None, {}

    tokens = _extract_token_usage(chat_fn)

    if not response or not response.strip():
        return None, tokens

    # Parse JSON response
    try:
        json_text = response
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0]

        parsed = json.loads(json_text.strip())
        body = parsed.get("body") if isinstance(parsed, dict) else None
        if not isinstance(body, str) or len(body) < 5:
            return None, tokens

        # Validate the body compiles
        body_lines = body.strip().split("\n")
        indented = "\n".join("    " + line for line in body_lines)
        wrapped = f"def _validate():\n{indented}\n"
        try:
            compile(wrapped, f"<repair:{tool_name}>", "exec")
        except SyntaxError as se:
            # Attempt normalization (reuse engine helper)
            try:
                from agenthatch.generate.engine import GenerateEngine

                fixed = GenerateEngine._normalize_indentation(
                    ["    " + line for line in body_lines],
                    [se.lineno - 1 if se.lineno else 0],
                )
                fixed_str = "\n".join(fixed)
                fixed_wrapper = f"def _validate():\n{fixed_str}\n"
                compile(fixed_wrapper, f"<repair:{tool_name}>", "exec")
                body = "\n".join(line[4:] if line.startswith("    ") else line for line in fixed)
            except (SyntaxError, Exception):
                logger.warning(
                    "LLM repair for tool '%s' has syntax error: %s",
                    tool_name,
                    se,
                )
                return None, tokens

        return body, tokens
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(
            "Failed to parse LLM repair response for '%s': %s", tool_name, e
        )
        return None, tokens


def _build_agent_context(
    ahs_dict: dict[str, Any], archetype: str | None = None
) -> str:
    """Build agent-level semantic context for repair LLM.

    Mirrors the agent-wide view that the original ``_ai_generate_tool_impls``
    call has, so repairs aren't limited to single-tool reasoning. Includes:
    - identity.display_name
    - intent.summary / intent.triggers
    - archetype (PROMPT_ONLY / TOOL_WRAPPER / MULTI_STEP / MCP_CONNECTOR)

    ``archetype`` is passed explicitly because it is computed at runtime by
    ``classify_skill()`` and is NOT persisted in the AHSSpec Pydantic model
    (``BaseSpec`` has no ``archetype`` field). The caller (``hatch_command``)
    owns the ``classification`` object and passes ``classification.archetype.value``
    here. Reading ``ahs_dict["base"]["archetype"]`` would always return empty
    (silent dead code) — see v0.9.22 review.
    """
    if not ahs_dict:
        return ""

    parts: list[str] = ["=== AGENT CONTEXT ===\n"]

    identity = ahs_dict.get("identity", {}) or {}
    if isinstance(identity, dict):
        display_name = identity.get("display_name", "")
        if display_name:
            parts.append(f"Agent name: {display_name}\n")

    intent = ahs_dict.get("intent", {}) or {}
    if isinstance(intent, dict):
        summary = intent.get("summary", "")
        triggers = intent.get("triggers", [])
        if summary:
            parts.append(f"Intent: {summary}\n")
        if isinstance(triggers, list) and triggers:
            parts.append(f"Triggers: {', '.join(str(t) for t in triggers)}\n")

    if archetype:
        parts.append(f"Archetype: {archetype}\n")

    # Only emit the section if we gathered at least one field beyond the header
    if len(parts) <= 1:
        return ""
    parts.append("\n")
    return "".join(parts)


def _collect_skill_context(skill_dir: Path) -> str:
    """Collect context files from skill directory.

    Subset of :meth:`GenerateEngine._ai_generate_tool_impls` context
    collection — gathers SKILL.md, reference files, and scripts.
    """
    parts: list[str] = []

    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        try:
            parts.append(
                f"--- SKILL.md ---\n{skill_md.read_text(encoding='utf-8')}"
            )
        except Exception:
            pass

    for refs_dir in (skill_dir / "skills" / "references", skill_dir):
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.glob("*")):
                if ref_file.is_file() and ref_file.suffix in (".md", ".txt"):
                    if ref_file.name in ("SKILL.md", "agenthatch.yaml"):
                        continue
                    try:
                        content = ref_file.read_text(encoding="utf-8")
                        if content:
                            rel = ref_file.relative_to(skill_dir)
                            parts.append(f"--- {rel} ---\n{content}")
                    except Exception:
                        pass

    for scripts_dir in (skill_dir / "skills" / "scripts", skill_dir / "scripts"):
        if scripts_dir.is_dir():
            for script_file in sorted(scripts_dir.glob("*")):
                if script_file.is_file():
                    try:
                        content = script_file.read_text(encoding="utf-8")
                        if content:
                            rel = script_file.relative_to(skill_dir)
                            parts.append(f"--- {rel} ---\n{content}")
                    except Exception:
                        pass

    return "\n\n".join(parts)


def _find_tool_metadata(
    ahs_dict: dict[str, Any], tool_name: str
) -> dict[str, Any] | None:
    """Find tool metadata from AHSSPEC by tool_name (or func_name)."""
    interface = ahs_dict.get("interface", {})
    provides = interface.get("provides", [])
    for cap in provides:
        if not isinstance(cap, dict):
            continue
        name = cap.get("capability", "")
        func_name = name.replace("-", "_") if isinstance(name, str) else ""
        if func_name == tool_name or name == tool_name:
            params: list[tuple[str, str, str]] = []
            input_schema = cap.get("input_schema", {}) or {}
            if isinstance(input_schema, dict):
                for param_name, param_type in input_schema.items():
                    params.append((param_name, str(param_type), "None"))

            return {
                "name": name,
                "func_name": func_name,
                "description": cap.get("description", ""),
                "backend_kind": cap.get("backend_kind", "none"),
                "params": params,
            }
    return None


def _apply_tool_repair(
    output_dir: Path, tool_name: str, new_body: str
) -> bool:
    """Replace a tool function's body in tools.py with new_body.

    Returns True if successful.
    """
    skills_dir = output_dir / "skills"
    candidate_tools: list[Path] = []
    for tools_py in output_dir.rglob("tools.py"):
        if skills_dir in tools_py.parents or tools_py.parent == skills_dir:
            continue
        candidate_tools.append(tools_py)
        if _replace_function_body(tools_py, tool_name, new_body):
            return True
    logger.warning(
        "apply_tool_repair: could not replace body for '%s'. "
        "Candidate tools.py files: %s",
        tool_name,
        [str(p.relative_to(output_dir)) for p in candidate_tools],
    )
    return False


def _replace_function_body(
    tools_py: Path, func_name: str, new_body: str
) -> bool:
    """Replace the body of a function in tools.py.

    ``new_body`` is the function body code at indent 0 (will be indented
    to 4 spaces when inserted).
    """
    content = tools_py.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.warning(
            "replace_function_body: tools.py %s has syntax error: %s",
            tools_py,
            e,
        )
        return False

    for node in tree.body:
        # Handle both sync (FunctionDef) and async (AsyncFunctionDef) tools.
        # AsyncFunctionDef is a separate AST node type — without this check,
        # async tools would never match and repair would silently fail.
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            continue

        old_source = ast.get_source_segment(content, node)
        if old_source is None:
            logger.warning(
                "replace_function_body: could not extract source for '%s' in %s",
                func_name,
                tools_py,
            )
            return False

        # Build new function source
        new_source = _rebuild_function_source(node, old_source, new_body)
        if new_source is None:
            logger.warning(
                "replace_function_body: could not rebuild source for '%s'",
                func_name,
            )
            return False

        # Replace
        new_content = content.replace(old_source, new_source, 1)
        if new_content == content:
            logger.warning(
                "replace_function_body: replace had no effect for '%s' in %s",
                func_name,
                tools_py,
            )
            return False

        # Validate the new content compiles
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            logger.warning(
                "replace_function_body: new content for '%s' has syntax error: %s",
                func_name,
                e,
            )
            return False

        tools_py.write_text(new_content, encoding="utf-8")
        return True

    # function not found in this particular tools.py is a normal control-flow
    # outcome when _apply_tool_repair iterates multiple candidate files —
    # debug, not warning, to avoid log noise.
    logger.debug(
        "replace_function_body: function '%s' not found in %s",
        func_name,
        tools_py,
    )
    return False


def _rebuild_function_source(
    node: ast.FunctionDef | ast.AsyncFunctionDef, old_source: str, new_body: str
) -> str | None:
    """Build a new function source from old signature+docstring + new body."""
    lines = old_source.split("\n")

    # Find signature end (line ending with `:`)
    sig_end = 0
    for i, line in enumerate(lines):
        if line.rstrip().endswith(":"):
            sig_end = i
            break
    signature_lines = lines[: sig_end + 1]

    # Find docstring (if any)
    docstring_lines: list[str] = []
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        doc_node = node.body[0]
        # doc_node.lineno and node.lineno are 1-indexed
        doc_start_idx = doc_node.lineno - node.lineno  # 0-indexed within function source
        # end_lineno is inclusive — +1 to make it an exclusive upper bound
        doc_end_idx = (
            doc_node.end_lineno or doc_node.lineno
        ) - node.lineno + 1
        docstring_lines = lines[doc_start_idx:doc_end_idx]

    # Build new function source
    new_lines = list(signature_lines)
    new_lines.extend(docstring_lines)

    # Add new body (indented 4 spaces)
    for line in new_body.strip().split("\n"):
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append("")

    return "\n".join(new_lines)


def _extract_token_usage(chat_fn: Any) -> dict[str, int]:
    """Extract token usage from chat_fn closure's LLMClient.last_usage.

    Recursively searches nested closures to handle wrapped chat functions
    (e.g. the token-accumulator wrapper in hatch_command).
    """
    try:
        return _find_llm_client_usage(chat_fn, depth=0)
    except Exception as e:
        logger.debug("Token usage extraction failed: %s", e)
        return {}


def _find_llm_client_usage(fn: Any, depth: int = 0) -> dict[str, int]:
    """Recursively search closure for LLMClient.last_usage."""
    if depth > 5:
        return {}

    closure_cells = getattr(fn, "__closure__", None) or []
    for cell in closure_cells:
        try:
            obj = cell.cell_contents
        except Exception:
            continue
        # Check if this object has last_usage (LLMClient)
        if hasattr(obj, "last_usage") and obj.last_usage is not None:
            usage = obj.last_usage
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            total = int(
                getattr(usage, "total_tokens", 0) or (prompt + completion)
            )
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
        # Recursively search callable objects (nested closures)
        if callable(obj) and hasattr(obj, "__closure__"):
            result = _find_llm_client_usage(obj, depth + 1)
            if result:
                return result
    return {}
