"""Regression tests for bugs in generated agent code (template-level guards).

Covers:
- R4-V23: Generated chat_stream uses ``return (yield from ...)`` (not bare ``yield from``)
- R4-V22: kb_max_text not stored as ``1`` (typo that disabled meta-narration stripping)
- R4-V16: ``sys.modules[spec.name]`` registered before exec_module (KB package resolution)
- python_escape: Jinja2 filter handles null bytes, control chars, triple quotes
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# R4-V23: ``return (yield from ...)`` vs bare ``yield from``
# ---------------------------------------------------------------------------


class TestR4V23YieldFromReturn:
    """P0 #1: ``chat_stream()`` must use ``return (yield from ...)``.

    Bare ``yield from`` discards the child generator's return value
    (the final answer text), so API callers that inspect
    ``StopIteration.value`` get an empty response even though the real
    answer was streamed to the user.
    """

    def test_chat_stream_uses_return_yield_from(self) -> None:
        """Render agent.py.j2 and assert the guard pattern appears exactly once."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("agent.py.j2")
        rendered = tpl.render(
            agent_name="test-agent",
            agent_class="TestAgent",
            display_name="Test Agent",
            version="0.1.0",
            package_name="test_agent",
            description="test",
            workflow="",
            workflow_steps=[],
            output_tpl="",
            rules=[],
            tools=[],
            tool_metadata=[],
            mcp_servers=[],
            api_templates=[],
            script_map={},
            requires=[],
            brick_manifest=None,
            loop_workflow=None,
            ai_tool_impls={},
            ai_references={},
            dependencies=[],
            kb=None,
            kb_enabled=False,
            base_runtime="python3.11",
            llm_provider="openai",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        # The template must use ``return (yield from ...)`` exactly once.
        count = rendered.count("return (yield from super().chat_stream(user_input))")
        assert count == 1, (
            f"Expected exactly 1 occurrence of 'return (yield from super().chat_stream(user_input))', "
            f"got {count}. Someone may have changed it to bare 'yield from'."
        )

    def test_chat_stream_uses_return_yield_from_with_kb(self) -> None:
        """Same guard, but with kb_enabled=True — pattern must still hold."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("agent.py.j2")
        rendered = tpl.render(
            agent_name="test-agent-kb",
            agent_class="TestAgentKB",
            display_name="Test Agent KB",
            version="0.1.0",
            package_name="test_agent_kb",
            description="test",
            workflow="",
            workflow_steps=[],
            output_tpl="",
            rules=[],
            tools=[],
            tool_metadata=[],
            mcp_servers=[],
            api_templates=[],
            script_map={},
            requires=[],
            brick_manifest=None,
            loop_workflow=None,
            ai_tool_impls={},
            ai_references={},
            dependencies=[],
            kb={},
            kb_enabled=True,
            base_runtime="python3.11",
            llm_provider="openai",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        count = rendered.count("return (yield from super().chat_stream(user_input))")
        assert count == 1, (
            f"Expected exactly 1 occurrence with kb_enabled=True, got {count}."
        )


# ---------------------------------------------------------------------------
# python_escape filter regression tests
# ---------------------------------------------------------------------------


class TestBugPythonEscape:
    """P1 #15: ``python_escape`` Jinja2 filter correctness.

    The filter must escape control characters, null bytes, triple quotes,
    and backslash-triple-quote combos so rendered Python source always
    compiles cleanly via ``compile()``.
    """

    def test_empty_string(self) -> None:
        """Empty string passes through unchanged."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        assert python_escape("") == ""

    def test_normal_string_unchanged(self) -> None:
        """Plain text with no special characters is unchanged."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        assert python_escape("hello world") == "hello world"

    def test_null_byte_escaped(self) -> None:
        """Null bytes (\\\\x00) prevent compile() — must be escaped."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        result = python_escape("\x00")
        assert result == "\\x00", f"Expected '\\\\x00', got {result!r}"

    def test_control_chars_escaped(self) -> None:
        """Control chars (\\\\x01–\\\\x1f, excluding \\\\t/\\\\n/\\\\r) must be escaped."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        result = python_escape("\x01\x02\x0b")
        assert result == "\\x01\\x02\\x0b", f"Expected '\\\\x01\\\\x02\\\\x0b', got {result!r}"

    def test_tab_newline_cr_preserved(self) -> None:
        """Whitespace tab/newline/carriage-return is preserved as-is."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        result = python_escape("\t\n\r")
        assert result == "\t\n\r", f"Expected literal whitespace, got {result!r}"

    def test_triple_double_quote_escaped(self) -> None:
        """Triple-quote sequence is escaped so it does not close the string literal."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        result = python_escape('some """ text')
        # The escaped form (which uses \\\" to break the triple-quote)
        # must compile cleanly when embedded in a triple-quoted string.
        snippet = f'x = """{result}"""'
        compile(snippet, "<test>", "exec")
        # Also verify escaping actually happened — the literal '"""'
        # must not pass through unchanged.  The function replaces """
        # with \\\""" which itself contains """ at the tail, but the
        # leading \\\" escapes the first quote so the remaining ""
        # is only 2 quotes and cannot terminate the string.
        assert result != 'some """ text', (
            f"Triple-quote was not escaped at all: {result!r}"
        )

    def test_backslash_escaped(self) -> None:
        """Backslashes are escaped to prevent unintended escape sequences."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        result = python_escape("a\\b")
        assert result == "a\\\\b", f"Expected 'a\\\\\\\\b', got {result!r}"

    def test_backslash_then_triple_quote(self) -> None:
        """Backslash before triple-quote: backslash escaping must happen first.

        Input is a Python literal with two backslashes followed by three
        double-quotes.  If triple-quotes are escaped first, the result
        would contain an unescaped triple-quote that, when embedded in a
        triple-quoted Python string, terminates the literal and causes
        a SyntaxError.

        The correct order is backslash-escaping first, then triple-quote
        escaping, producing a safe escaped form.
        """
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        python_escape = engine._env.filters["python_escape"]
        # Input: two literal backslashes followed by three double-quotes
        result = python_escape('\\"""')
        # Verify the result, when embedded in a triple-quoted Python
        # string, compiles without SyntaxError.  This is the true
        # correctness test — if the escaping order is wrong, the
        # embedded string would contain an unescaped """ that terminates
        # the literal.
        snippet = f'x = """{result}"""'
        compile(snippet, "<test>", "exec")
        # Also verify escaping actually happened.
        assert result != '\\"""', (
            f"Backslash-then-triple-quote was not escaped: {result!r}"
        )


# ---------------------------------------------------------------------------
# R4-V22: kb_max_text typo guard
# ---------------------------------------------------------------------------


class TestR4V22KbMaxTextNotOne:
    """P0 #2: ``kb_max_text`` must not be ``1`` in generated agent code.

    A typo in ``chat_stream()``'s KB auto-continuation cap stored
    ``kb_max_text = 1`` instead of ``0`` (the correct value used by
    ``chat()``).  With ``1``, ``self._max_consecutive_text_only == 0``
    never holds in the streaming path, so ``_strip_trailing_meta_narration``
    is never called and meta-narration ("前已详答" / "前问已答毕") leaks
    through to the user.

    The fix lives in the runtime (``agenthatch_core.agent.AHCoreAgent``),
    not in the template.  This test guards against the value ever
    appearing in the rendered template output as well.
    """

    def test_kb_max_text_not_one(self) -> None:
        """Render agent template and verify ``kb_max_text = 1`` does not appear."""
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("agent.py.j2")
        rendered = tpl.render(
            agent_name="test-agent",
            agent_class="TestAgent",
            display_name="Test Agent",
            version="0.1.0",
            package_name="test_agent",
            description="test",
            workflow="",
            workflow_steps=[],
            output_tpl="",
            rules=[],
            tools=[],
            tool_metadata=[],
            mcp_servers=[],
            api_templates=[],
            script_map={},
            requires=[],
            brick_manifest=None,
            loop_workflow=None,
            ai_tool_impls={},
            ai_references={},
            dependencies=[],
            kb=None,
            kb_enabled=False,
            base_runtime="python3.11",
            llm_provider="openai",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )

        # If ``kb_max_text`` appears at all in the rendered output,
        # ``= 1`` must NOT follow it.  The variable shouldn't be in the
        # template to begin with (it's a runtime concern), but if someone
        # adds it later this test catches the typo.
        import re
        for m in re.finditer(r"kb_max_text\s*=\s*(\d+)", rendered):
            val = int(m.group(1))
            assert val != 1, (
                f"kb_max_text = {val} found in rendered output. "
                f"Expected 0 (or absent), not 1 — this would disable "
                f"meta-narration stripping in chat_stream()."
            )


# ---------------------------------------------------------------------------
# R4-V16: sys.modules registered before exec_module
# ---------------------------------------------------------------------------


class TestR4V16SysModulesRegistered:
    """P0 #3: ``sys.modules[spec.name]`` registered BEFORE ``exec_module``.

    The CLI ``run.py`` must register the agent module in ``sys.modules``
    before calling ``spec.loader.exec_module(module)`` so that
    ``KnowledgeBaseBrick``'s package detection code can resolve the
    agent's package name via ``sys.modules.get(type(self).__module__)``.

    Without this registration, ``spec_from_file_location`` +
    ``exec_module`` leaves the module unregistered, and the KB init code
    falls back to deriving the package from ``__module__`` (which works
    but is fragile).
    """

    def test_sys_modules_registered_for_agent_module(self) -> None:
        """Verify ``sys.modules[spec.name] = module`` appears before ``exec_module``.

        Uses ``ast.parse`` on ``run.py`` to statically verify the
        ordering — no runtime execution needed.
        """
        import ast
        from pathlib import Path

        run_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "agenthatch" / "cli" / "commands" / "run.py"
        )
        source = run_path.read_text()

        # Sanity: file exists and is readable
        assert source, f"run.py is empty or unreadable at {run_path}"

        tree = ast.parse(source)

        # Collect all relevant statement line numbers
        assign_line: int | None = None
        exec_module_line: int | None = None

        for node in ast.walk(tree):
            # Look for: sys.modules[spec.name] = module
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                    ):
                        attr = target.value
                        if (
                            isinstance(attr.value, ast.Name)
                            and attr.value.id == "sys"
                            and attr.attr == "modules"
                        ):
                            # Found ``sys.modules[...] = ...``
                            assign_line = node.lineno

            # Look for: spec.loader.exec_module(module)
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "exec_module"
                    and isinstance(node.func.value, ast.Attribute)
                ):
                    inner = node.func.value
                    if (
                        isinstance(inner.value, ast.Name)
                        and inner.value.id == "spec"
                        and inner.attr == "loader"
                    ):
                        exec_module_line = node.lineno

        assert assign_line is not None, (
            "Could not find 'sys.modules[spec.name] = module' in run.py. "
            "The R4-V16 guard may have been removed."
        )
        assert exec_module_line is not None, (
            "Could not find 'spec.loader.exec_module(module)' in run.py."
        )
        assert assign_line < exec_module_line, (
            f"sys.modules[spec.name] = module (line {assign_line}) "
            f"must appear BEFORE exec_module (line {exec_module_line}). "
            f"Otherwise KnowledgeBaseBrick cannot resolve the agent's "
            f"package name via sys.modules."
        )
