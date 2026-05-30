"""Tests for SkillAgent — runtime assembly and config resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agenthatch.agent.runtime import SkillAgent, SkillBrick
from agenthatch.base.sandbox import Sandbox
from agenthatch.skill.spec import (
    AgentConfig,
    AgentRuntimeConfig,
    AHSSpec,
    BaseSpec,
    Capability,
    Identity,
    Instructions,
    Intent,
    Interface,
    WorkflowStep,
)


@pytest.fixture
def minimal_spec() -> AHSSpec:
    return AHSSpec(
        identity=Identity(id="test", display_name="Test", version="1.0"),
        intent=Intent(triggers=["a"], satisfies=["b"], summary="c"),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def spec_with_agent_config() -> AHSSpec:
    return AHSSpec(
        identity=Identity(id="configured", display_name="Configured", version="1.0"),
        intent=Intent(triggers=["x"], satisfies=["y"], summary="z"),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
        agent=AgentConfig(
            runtime=AgentRuntimeConfig(
                provider="custom-provider",
                model="custom-model",
                env={"CUSTOM_KEY": "custom_value"},
                temperature=0.5,
                max_tokens=2048,
            )
        ),
    )


@pytest.fixture
def spec_with_capabilities() -> AHSSpec:
    return AHSSpec(
        identity=Identity(id="cap-skill", display_name="Cap Skill", version="1.0"),
        intent=Intent(triggers=["cap"], satisfies=["cap {x}"], summary="cap test"),
        interface=Interface(
            provides=[
                Capability(
                    capability="process_data",
                    type="data",
                    input_schema={"type": "object"},
                ),
                Capability(
                    capability="generate_report",
                    type="renderer",
                    input_schema={"type": "object"},
                ),
            ],
            requires=[
                Capability(capability="http_client", type="data"),
            ],
        ),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def spec_with_scripts(tmp_path) -> AHSSpec:
    script_path = tmp_path / "hello.sh"
    script_path.write_text("#!/bin/bash\necho hello")
    spec = AHSSpec(
        identity=Identity(id="script-skill", display_name="Script Skill", version="1.0"),
        intent=Intent(triggers=["run"], satisfies=["run script"], summary="script test"),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(runtime="bash"),
        instructions=Instructions(
            workflow=[
                WorkflowStep(step=1, description="Run hello", script="hello.sh"),
            ]
        ),
    )
    return spec


@pytest.fixture
def ahs_yaml_path(tmp_path, minimal_spec) -> Path:
    ahs_path = tmp_path / "agenthatch.yaml"
    data = {
        "identity": {
            "id": "test",
            "display_name": "Test",
            "version": "1.0",
        },
        "intent": {
            "triggers": ["a"],
            "satisfies": ["b"],
            "summary": "c",
        },
        "interface": {
            "provides": [],
            "requires": [],
            "compatible_with": [],
        },
        "base": {
            "runtime": None,
            "sandbox": False,
            "timeout": "60s",
            "env": [],
            "dependencies": [],
        },
        "instructions": {
            "workflow": [],
            "rules": [],
            "safety": {},
            "output_template": None,
        },
        "resources": {
            "scripts": [],
            "references": [],
            "assets": [],
        },
        "composition": {
            "event_listeners": [],
        },
        "agent": None,
        "harness_traces": [],
    }
    ahs_path.write_text(yaml.dump(data))
    return ahs_path


def _make_patches(stack):
    """Create the standard set of patches for SkillAgent tests."""
    patches = (
        stack.enter_context(patch("agenthatch.agent.runtime.LLMClient")),
        stack.enter_context(patch("agenthatch.agent.runtime.Sandbox")),
        stack.enter_context(patch("agenthatch.agent.runtime.CapBus")),
        stack.enter_context(patch("agenthatch.agent.runtime.ConversationLoop")),
        stack.enter_context(patch("agenthatch.agent.runtime.ContextManager")),
        stack.enter_context(
            patch("agenthatch.agent.runtime.is_builtin", return_value=False)
        ),
    )
    # v0.4.1: mock get_default_provider and get_provider for config resolution
    stack.enter_context(
        patch(
            "agenthatch.agent.runtime.get_default_provider",
            return_value="openai",
        )
    )
    stack.enter_context(
        patch(
            "agenthatch.agent.runtime.get_provider",
            return_value=_make_mock_provider_info(),
        )
    )
    return patches


def _make_mock_provider_info():
    """Create a mock ProviderInfo with ProviderFeatures for tests."""
    from unittest.mock import MagicMock

    features = MagicMock()
    features.supports_tools = True
    features.supports_stream_tools = True
    features.supports_json_mode = True
    features.supports_parallel_tool_calls = True
    features.supports_reasoning_content = False
    features.requires_anthropic_adapter = False
    features.available_models = ()
    info = MagicMock()
    info.features = features
    return info


class TestSkillAgentConfigResolution:
    def test_cli_override_passed_to_llm_client(self, spec_with_agent_config):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mock_llm, _, _, _, _, _ = _make_patches(stack)
            SkillAgent(
                spec_with_agent_config,
                skill_dir=Path("/tmp"),
                provider="cli-provider",
                api_key="cli-key",
                model="cli-model",
            )
            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["provider_name"] == "cli-provider"
            assert call_kwargs["model"] == "cli-model"

    def test_yaml_config_fallback(self, spec_with_agent_config):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mock_llm, _, _, _, _, _ = _make_patches(stack)
            SkillAgent(spec_with_agent_config, skill_dir=Path("/tmp"))
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["provider_name"] == "custom-provider"
            assert call_kwargs["model"] == "custom-model"

    def test_no_config_creates_agent(self, minimal_spec):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mock_llm, _, _, _, _, _ = _make_patches(stack)
            agent = SkillAgent(minimal_spec, skill_dir=Path("/tmp"))
            assert agent is not None
            mock_llm.assert_called_once()


class TestSkillAgentFromAhspec:
    def test_from_ahspec_loads_and_returns_agent(self, ahs_yaml_path):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _, _, _, _, _, _ = _make_patches(stack)
            agent = SkillAgent.from_ahspec(ahs_yaml_path)
            assert agent is not None
            assert agent.spec.identity.id == "test"


class TestSkillAgentAssemble:
    def test_capabilities_registered(self, spec_with_capabilities):
        import agenthatch.agent.builtins.http_client  # noqa: F401 - populate registry
        with (
            patch("agenthatch.agent.runtime.LLMClient"),
            patch("agenthatch.agent.runtime.Sandbox"),
            patch("agenthatch.agent.runtime.ContextManager"),
            patch("agenthatch.agent.runtime.ConversationLoop"),
            patch("agenthatch.agent.runtime.is_builtin", return_value=True),
            patch(
                "agenthatch.agent.runtime.get_default_provider",
                return_value="openai",
            ),
            patch(
                "agenthatch.agent.runtime.get_provider",
                return_value=_make_mock_provider_info(),
            ),
        ):
            agent = SkillAgent(spec_with_capabilities, skill_dir=Path("/tmp"))
            caps = agent.capbus.capabilities
            assert "process_data" in caps
            assert "generate_report" in caps
            assert "http_client" in agent.capbus.builtins

    def test_scripts_registered_as_tool(self, spec_with_scripts, tmp_path):
        with (
            patch("agenthatch.agent.runtime.LLMClient"),
            patch("agenthatch.agent.runtime.Sandbox"),
            patch("agenthatch.agent.runtime.ContextManager"),
            patch("agenthatch.agent.runtime.ConversationLoop"),
            patch("agenthatch.agent.runtime.is_builtin", return_value=False),
            patch(
                "agenthatch.agent.runtime.get_default_provider",
                return_value="openai",
            ),
            patch(
                "agenthatch.agent.runtime.get_provider",
                return_value=_make_mock_provider_info(),
            ),
        ):
            agent = SkillAgent(spec_with_scripts, skill_dir=tmp_path)
            assert "run_skill_script" in agent.capbus.capabilities

    def test_unavailable_caps_marked(self, spec_with_capabilities):
        with (
            patch("agenthatch.agent.runtime.LLMClient"),
            patch("agenthatch.agent.runtime.Sandbox"),
            patch("agenthatch.agent.runtime.ContextManager"),
            patch("agenthatch.agent.runtime.ConversationLoop"),
            patch("agenthatch.agent.runtime.is_builtin", return_value=False),
            patch(
                "agenthatch.agent.runtime.get_default_provider",
                return_value="openai",
            ),
            patch(
                "agenthatch.agent.runtime.get_provider",
                return_value=_make_mock_provider_info(),
            ),
        ):
            agent = SkillAgent(spec_with_capabilities, skill_dir=Path("/tmp"))
            assert "http_client" in agent.capbus.unavailable


class TestSkillBrick:
    def test_creates_from_spec(self, minimal_spec):
        sandbox = Sandbox()
        brick = SkillBrick(minimal_spec, Path("/tmp"), sandbox)
        assert brick.id == "test"
        assert brick.spec is minimal_spec

    def test_build_workflow_for_prompt(self, spec_with_scripts, tmp_path):
        sandbox = Sandbox()
        brick = SkillBrick(spec_with_scripts, tmp_path, sandbox)
        text = brick.build_workflow_for_prompt()
        assert "Run hello" in text
        assert "run_skill_script" in text

    def test_execute_script_not_found(self, minimal_spec):
        sandbox = Sandbox()
        brick = SkillBrick(minimal_spec, Path("/tmp"), sandbox)
        result = brick.execute_script("nonexistent.sh")
        assert "not found" in result
