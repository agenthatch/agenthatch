"""Tests for ContextManager — system prompt construction and history management."""

from __future__ import annotations

import pytest
from agenthatch_core.context.manager import ContextManager

from agenthatch.skill.spec import (
    AHSSpec,
    BaseSpec,
    Identity,
    Instructions,
    Intent,
    Interface,
    Safety,
    WorkflowStep,
)


@pytest.fixture
def minimal_spec() -> AHSSpec:
    """Minimal valid AHSSpec for testing."""
    return AHSSpec(
        identity=Identity(
            id="test-skill",
            display_name="Test Skill",
            version="1.0.0",
            author="Tester",
        ),
        intent=Intent(
            triggers=["test", "verify"],
            satisfies=["verify {thing}"],
            summary="A test skill for testing.",
        ),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def full_spec() -> AHSSpec:
    """Full-featured AHSSpec with workflow, rules, output template."""
    return AHSSpec(
        identity=Identity(
            id="full-skill",
            display_name="Full Skill",
            version="2.0.0",
            author="Dev",
        ),
        intent=Intent(
            triggers=["analyze", "report", "generate"],
            satisfies=["analyze {data}", "generate {report}"],
            summary="A full-featured skill for testing.",
        ),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(
            workflow=[
                WorkflowStep(step=1, description="Load data"),
                WorkflowStep(step=2, description="Analyze data", script="analyze.py"),
                WorkflowStep(step=3, description="Generate report"),
            ],
            rules=["Rule 1: Validate inputs", "Rule 2: Never expose secrets"],
            safety=Safety(plan_required=True),
            output_template="# Report:\n{content}\n## Summary:\n{summary}",
        ),
    )


class TestContextManagerInit:
    """Test ContextManager initialization."""

    def test_init_stores_spec(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        assert ctx.spec is minimal_spec

    def test_init_empty_history(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        assert ctx.history == []

    def test_default_max_turns(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        assert ctx.max_history_turns == 20


class TestBuildSystemPrompt:
    """Test system prompt construction."""

    def test_minimal_prompt_contains_identity(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        prompt = ctx.build_system_prompt()
        assert "Test Skill" in prompt
        assert "Tester" in prompt

    def test_minimal_prompt_contains_intent(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        prompt = ctx.build_system_prompt()
        assert "A test skill for testing" in prompt
        assert "test" in prompt
        assert "verify" in prompt

    def test_minimal_prompt_has_domain_guard(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        prompt = ctx.build_system_prompt()
        assert "outside your domain" in prompt.lower()

    def test_prompt_includes_workflow(self, full_spec):
        ctx = ContextManager(full_spec)
        prompt = ctx.build_system_prompt()
        assert "Load data" in prompt
        assert "Analyze data" in prompt
        assert "Generate report" in prompt
        assert "run_skill_script" in prompt

    def test_prompt_includes_rules(self, full_spec):
        ctx = ContextManager(full_spec)
        prompt = ctx.build_system_prompt()
        assert "Rule 1: Validate inputs" in prompt
        assert "Rule 2: Never expose secrets" in prompt

    def test_prompt_includes_safety_plan(self, full_spec):
        ctx = ContextManager(full_spec)
        prompt = ctx.build_system_prompt()
        assert "create a plan before executing" in prompt.lower()

    def test_prompt_includes_output_template(self, full_spec):
        ctx = ContextManager(full_spec)
        prompt = ctx.build_system_prompt()
        assert "Report:" in prompt
        assert "{content}" in prompt
        assert "{summary}" in prompt
        assert "MANDATORY" in prompt

    def test_prompt_without_author(self):
        spec = AHSSpec(
            identity=Identity(id="no-author", display_name="No Author", version="1.0.0"),
            intent=Intent(triggers=["a"], satisfies=["b"], summary="c"),
            interface=Interface(provides=[], requires=[]),
            base=BaseSpec(),
            instructions=Instructions(),
        )
        ctx = ContextManager(spec)
        prompt = ctx.build_system_prompt()
        assert "Author:" not in prompt


class TestBuildMessages:
    """Test message list construction."""

    def test_includes_system_message(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        messages = ctx.build_messages("hello")
        assert messages[0]["role"] == "system"
        assert "Test Skill" in messages[0]["content"]

    def test_includes_user_message(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        messages = ctx.build_messages("hello world")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "hello world"

    def test_no_history_includes_only_system_and_user(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        messages = ctx.build_messages("hi")
        assert len(messages) == 2

    def test_with_history_includes_prior_messages(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        ctx.add_to_history("user", "previous question")
        ctx.add_to_history("assistant", "previous answer")
        messages = ctx.build_messages("new question")
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "previous answer"


class TestAddToHistory:
    """Test history management."""

    def test_add_user_message(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        ctx.add_to_history("user", "hello")
        assert len(ctx.history) == 1
        assert ctx.history[0] == {"role": "user", "content": "hello"}

    def test_add_assistant_message(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        ctx.add_to_history("assistant", "hi there")
        assert len(ctx.history) == 1
        assert ctx.history[0] == {"role": "assistant", "content": "hi there"}

    def test_multiple_messages_accumulate(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        for i in range(5):
            ctx.add_to_history("user", f"msg {i}")
        assert len(ctx.history) == 5


class TestHistoryWindow:
    """Test max_history_turns truncation."""

    def test_history_respects_window(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        ctx.max_history_turns = 2
        for i in range(10):
            ctx.add_to_history("user", f"msg {i}")
            ctx.add_to_history("assistant", f"reply {i}")
        messages = ctx.build_messages("final")
        history_in_messages = [m for m in messages if m["role"] in ("user", "assistant")]
        assert len(history_in_messages) <= 4 + 1
        assert messages[-1]["content"] == "final"


class TestCompress:
    """Test context compression stub."""

    def test_compress_returns_stub(self, minimal_spec):
        ctx = ContextManager(minimal_spec)
        result = ctx.compress()
        assert "not yet available" in result
