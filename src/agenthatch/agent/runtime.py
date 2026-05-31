"""SkillAgent — the base brick and entry point for v0.4 Agent runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agenthatch.agent.context import ContextManager
from agenthatch.agent.loop import ConversationLoop
from agenthatch.base.sandbox import Sandbox
from agenthatch.cap.bus import CapBus
from agenthatch.house.resolver import is_builtin
from agenthatch.providers import ProviderFeatures, get_default_provider, get_provider
from agenthatch.skill.llm_client import LLMClient
from agenthatch.skill.spec import AHSSpec

logger = logging.getLogger(__name__)


class SkillBrick:
    """A Skill brick — capability encapsulation loaded from AHSSPEC."""

    def __init__(self, spec: AHSSpec, skill_dir: Path, sandbox: Sandbox):
        self.spec = spec
        self.id = spec.identity.id
        self.skill_dir = skill_dir
        self.sandbox = sandbox

        self.provides_map = {
            cap.capability: cap
            for cap in spec.interface.provides
        }

        self._scripts: dict[str, Path] = {}
        self._script_steps: dict[str, str] = {}
        for step in spec.instructions.workflow:
            if step.script:
                p = skill_dir / step.script
                if p.exists():
                    self._scripts[step.script] = p
                    self._script_steps[step.script] = step.description

    def execute_script(self, script_name: str = "", **kwargs: Any) -> str:
        """Run the script specified by LLM via run_skill_script tool."""
        script = self._scripts.get(script_name)
        if not script:
            available = list(self._scripts.keys())
            return (
                f"Error: script '{script_name}' not found. "
                f"Available scripts: {available}"
            )
        return self.sandbox.run(str(script), env=kwargs)

    def build_workflow_for_prompt(self) -> str:
        """Generate workflow text for system prompt injection."""
        lines: list[str] = []
        for step in self.spec.instructions.workflow:
            line = f"{step.step}. {step.description}"
            if step.script:
                line += (
                    f"\n   -> Use tool: run_skill_script("
                    f'script_name="{step.script}", ...)'
                )
            lines.append(line)
        return "\n".join(lines)


MIN_GENERATION_TOKENS = 1024


class SkillAgent:
    """Base Brick — the entry point for v0.4 Agent runtime."""

    @classmethod
    def from_ahspec(cls, ahs_path: Path, **overrides: Any) -> SkillAgent:
        """Load SkillAgent from agenthatch.yaml."""
        spec = AHSSpec.model_validate(yaml.safe_load(ahs_path.read_text()))
        return cls(spec, skill_dir=ahs_path.parent, **overrides)

    def __init__(
        self,
        ahs_spec: AHSSpec,
        skill_dir: Path,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.spec = ahs_spec
        self.skill_dir = skill_dir
        self.ctx = ContextManager(ahs_spec)

        runtime_config = self._resolve_runtime_config(provider, api_key, model)

        self.capbus = CapBus()
        self.sandbox = Sandbox()

        self.llm = LLMClient(
            provider_name=runtime_config["provider"],
            model=runtime_config["model"],
        )

        self.loop = ConversationLoop(
            llm=self.llm,
            capbus=self.capbus,
            sandbox=self.sandbox,
            ctx=self.ctx,
        )

        self._assemble()

    def _resolve_runtime_config(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Resolve runtime config with 3-level priority: CLI > agenthatch.yaml > config.toml."""
        agent_cfg = self.spec.agent
        resolved: dict[str, Any] = {
            "provider": provider or (
                agent_cfg.runtime.provider if agent_cfg else None
            ) or get_default_provider(),
            "model": model or (
                agent_cfg.runtime.model if agent_cfg else None
            ) or "",
            "api_key": api_key or "",
            "temperature": (
                agent_cfg.runtime.temperature if agent_cfg else 0.7
            ),
            "max_tokens": (
                agent_cfg.runtime.max_tokens if agent_cfg else 4096
            ),
        }

        agent_features = agent_cfg.runtime.features if agent_cfg else {}
        provider_features = getattr(
            get_provider(resolved["provider"]), "features", ProviderFeatures()
        )
        def _getf(key: str, default: bool) -> bool:
            val = agent_features.get(key, default)
            return val if isinstance(val, bool) else default

        merged_features = ProviderFeatures(
            supports_tools=_getf("supports_tools", provider_features.supports_tools),
            supports_stream_tools=_getf(
                "supports_stream_tools", provider_features.supports_stream_tools
            ),
            supports_json_mode=_getf(
                "supports_json_mode", provider_features.supports_json_mode
            ),
            supports_parallel_tool_calls=_getf(
                "supports_parallel_tool_calls",
                provider_features.supports_parallel_tool_calls,
            ),
            supports_reasoning_content=_getf(
                "supports_reasoning_content",
                provider_features.supports_reasoning_content,
            ),
            requires_anthropic_adapter=_getf(
                "requires_anthropic_adapter",
                provider_features.requires_anthropic_adapter,
            ),
            available_models=provider_features.available_models,
        )
        resolved["features"] = merged_features

        # BUG-04-02: Dynamic max_tokens based on estimated input tokens
        requested_max = resolved["max_tokens"]
        estimated_input = self.ctx.estimate_input_tokens()
        if not isinstance(estimated_input, int):
            estimated_input = 0
        safe_max = max(MIN_GENERATION_TOKENS, requested_max - estimated_input)
        if safe_max < requested_max:
            logger.warning(
                "Adjusted max_tokens: %d -> %d (input estimate: %d tokens)",
                requested_max, safe_max, estimated_input,
            )
        resolved["max_tokens"] = safe_max
        return resolved

    def _assemble(self) -> None:
        """Assemble capabilities from AHSSPEC."""
        skill_brick = SkillBrick(self.spec, self.skill_dir, self.sandbox)

        for cap in self.spec.interface.provides:
            self.capbus.register(
                name=cap.capability,
                cap_type=cap.type,
                schema=cap.input_schema,
                source_skill=self.spec.identity.id,
                executor=skill_brick,
            )

        if skill_brick._scripts:
            self.capbus.register(
                name="run_skill_script",
                cap_type="runtime",
                schema={
                    "type": "object",
                    "properties": {
                        "script_name": {
                            "type": "string",
                            "description": "Script to execute",
                            "enum": list(skill_brick._scripts.keys()),
                        },
                    },
                    "required": ["script_name"],
                },
                source_skill=self.spec.identity.id,
                executor=skill_brick,
            )

        for req in self.spec.interface.requires:
            if is_builtin(req.capability):
                self.capbus.inject_builtin(req.capability)
            else:
                self.capbus.mark_unavailable(req.capability)

        self.sandbox.configure(
            runtime=self.spec.base.runtime,
            isolated=self.spec.base.sandbox,
            timeout=self.spec.base.timeout,
            env={e.name: e.description for e in self.spec.base.env},
        )

        if self.spec.instructions.output_template:
            logger.info(
                "Skill '%s': output_template injected (%d chars)",
                self.spec.identity.id,
                len(self.spec.instructions.output_template),
            )

    def chat(self, user_input: str) -> str:
        """Single-turn synchronous chat."""
        return self.loop.run(user_input)

    def chat_stream(self, user_input: str) -> Any:
        """Streaming chat for TUI consumption."""
        yield from self.loop.stream(user_input)
