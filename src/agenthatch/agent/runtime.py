"""SkillAgent — the base brick and entry point for v0.5 Agent runtime.

v0.5 additions:
- HooksManager + StateManager wired into ContextManager
- Per-skill compact config from agenthatch.yaml
- Session summary restore from .agenthatch/state/<skill-id>/
- chat_stream() PEP 380 return value fix
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import yaml
from agenthatch_core.bricks.manifest import (
    BrickManifest,
    LoopKind,
)
from agenthatch_core.bricks.stubs import (
    _NullCapBus,
    _NullHooks,
)
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.hooks import HooksManager
from agenthatch_core.llm.client import LLMClient
from agenthatch_core.loop.agent_loop import ConversationLoop
from agenthatch_core.mcp.client import MCPClient
from agenthatch_core.mcp.config import MCPServerConfig
from agenthatch_core.sandbox.executor import Sandbox, SandboxResult
from pydantic import ValidationError

from agenthatch.agent.compact import CompactSummary
from agenthatch.agent.offload import CheckpointManager, StateManager
from agenthatch.cap.bus import APITemplateExecutor, CapBus
from agenthatch.house.resolver import is_builtin
from agenthatch.providers import (
    ProviderFeatures,
    get_default_provider,
    get_provider,
    resolve_api_key,
)
from agenthatch.skill.spec import AHSSpec

logger = logging.getLogger(__name__)


_SAFE_ENV_PREFIXES = ("",)  # Allow all by default, but block known dangerous keys
_DANGEROUS_ENV_KEYS = {
    "PATH", "HOME", "USER", "SHELL", "PWD",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
}


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
                p = (skill_dir / step.script).resolve()
                if p.exists():
                    self._scripts[step.script] = p
                    self._script_steps[step.script] = step.description

        for script_entry in spec.resources.scripts:
            script_name = script_entry.get("name", "")
            if not script_name:
                continue
            script_path = skill_dir / "scripts" / script_name
            if not script_path.exists():
                script_path = skill_dir / script_name
            if script_path.exists():
                self._scripts[script_name] = script_path

    def execute_script(self, script_name: str = "", **kwargs: Any) -> str:
        """Run the script specified by LLM via run_skill_script tool."""
        script = self._scripts.get(script_name)
        if not script:
            available = list(self._scripts.keys())
            return (
                f"Error: script '{script_name}' not found. "
                f"Available scripts: {available}"
            )
        safe_env = {k: str(v) for k, v in kwargs.items() if k.upper() not in _DANGEROUS_ENV_KEYS}
        result: SandboxResult = self.sandbox.run(str(script), env=safe_env)
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {result.stderr or result.stdout}"
        return result.stdout

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

    # Reasoning models share a single max_tokens budget between
    # reasoning and content tokens. Multiply default budget by this factor
    # to ensure adequate content token allocation (≈40% of total).
    REASONING_MAX_TOKENS_MULTIPLIER: float = 2.5

    @classmethod
    def from_ahspec(cls, ahs_path: Path, **overrides: Any) -> SkillAgent:
        """Load SkillAgent from agenthatch.yaml."""
        try:
            raw = ahs_path.read_text(encoding="utf-8")
            spec = AHSSpec.model_validate(yaml.safe_load(raw))
        except (OSError, yaml.YAMLError, ValidationError) as e:
            from agenthatch.exceptions import AgentHatchError
            raise AgentHatchError(f"Failed to load AHSSPEC from {ahs_path}: {e}") from e

        # v0.8.2: agent.runtime removed from AgentConfig; runtime config
        # now lives in runtime.toml (generated by hatch). The from_ahspec
        # path receives provider/model via SkillAgent.__init__ overrides.

        # Build BrickManifest from skill classification
        manifest = cls._build_manifest(spec)

        return cls(spec, skill_dir=ahs_path.parent, brick_manifest=manifest, **overrides)

    @staticmethod
    def _build_manifest(spec: AHSSpec) -> BrickManifest:
        """Build a BrickManifest from skill classification."""
        from agenthatch_core.bricks.archetypes import (
            SkillArchetype,
            classify_skill,
        )
        from agenthatch_core.bricks.guards import OutputGuard

        # Dump spec to dict for classify_skill
        spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else {}
        classification = classify_skill(spec_dict)
        archetype = classification.archetype

        # Map archetype → loop engine
        if archetype == SkillArchetype.PROMPT_ONLY:
            loop_engine = LoopKind.DIRECT
        else:
            loop_engine = LoopKind.CONVERSATION

        # v0.8: All archetypes use direct subprocess execution.
        # No sandbox tier selection — Sandbox is always enabled.

        capbus = archetype != SkillArchetype.PROMPT_ONLY
        hooks = archetype not in (
            SkillArchetype.PROMPT_ONLY, SkillArchetype.EXTERNAL_TOOL
        )

        api_templates = (
            spec.interface.api_templates
            if hasattr(spec.interface, "api_templates")
            else []
        )
        credential_vault = bool(api_templates)

        file_processor = archetype in (
            SkillArchetype.TOOL_WRAPPER, SkillArchetype.MULTI_STEP
        )

        rules = list(spec.instructions.rules) if hasattr(spec.instructions, "rules") else []
        guard_active = bool(rules) and archetype != SkillArchetype.PROMPT_ONLY
        guard = OutputGuard.from_rules(rules) if guard_active else None  # type: ignore[arg-type]

        return BrickManifest(
            loop_engine=loop_engine,
            capbus=capbus,
            hooks=hooks,
            guard=guard,
            guard_active=guard_active,
            credential_vault=credential_vault,
            file_processor=file_processor,
            archetype=archetype.value,
            archetype_confidence=classification.confidence,
        )

    def __init__(
        self,
        ahs_spec: AHSSpec,
        skill_dir: Path,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        brick_manifest: BrickManifest | None = None,
    ):
        self.spec = ahs_spec
        self.skill_dir = skill_dir
        self.ctx = ContextManager(ahs_spec)
        self.ctx._skill_dir = skill_dir
        self.ctx._rich_prompt = False
        self._rich_prompt: bool = False

        # ── v0.7: BrickManifest-driven chassis assembly ──
        self._manifest = brick_manifest or BrickManifest()

        # ── v0.5: Apply per-skill compact config ──
        agent_cfg = self.spec.agent
        agent_runtime = getattr(agent_cfg, "runtime", None) if agent_cfg else None
        if agent_runtime and agent_runtime.compact:
            self.ctx.compact_config = {
                "enabled": agent_runtime.compact.enabled,
                "ratio": agent_runtime.compact.ratio,
                "min_recent_turns": agent_runtime.compact.min_recent_turns,
            }
            self.ctx._apply_compact_config()

        runtime_config = self._resolve_runtime_config(provider, api_key, model)

        # v0.8: Sandbox always enabled — direct subprocess execution
        self.capbus: Any = CapBus() if self._manifest.capbus else _NullCapBus()
        self.sandbox: Any = Sandbox()

        # v0.8.1: Whitelist removed — agent has full CLI capability

        # ── v0.5: Wire hooks + state management ──
        self.hooks: Any = HooksManager() if self._manifest.hooks else _NullHooks()
        self.state = StateManager(
            Path(".agenthatch") / "state" / self.spec.identity.id
        )

        # ── v0.7: OutputGuard ──
        self.guard: Any = self._manifest.guard

        # ── v0.7: CredentialVault ──
        self.vault: Any = None
        if self._manifest.credential_vault:
            from agenthatch_core.bricks.credential_vault import (
                CredentialVault,
            )
            self.vault = CredentialVault()

        # ── v0.7: FileProcessor ──
        self.file_processor: Any = None
        if self._manifest.file_processor:
            from agenthatch_core.bricks.file_processor import (
                FileProcessor,
            )
            self.file_processor = FileProcessor()

        # ── v0.7: TokenCounter (unconditional) ──
        from agenthatch_core.loop.token_counter import TokenCounter
        self.token_counter = TokenCounter()

        self.llm = LLMClient(
            provider=runtime_config["provider"],
            model=runtime_config["model"],
            api_key=runtime_config.get("api_key") or None,
            base_url=runtime_config.get("base_url") or "",
            features=runtime_config.get("features"),
            context_window=runtime_config.get("context_window"),
        )

        self.loop = ConversationLoop(
            llm=self.llm,
            capbus=self.capbus,  # type: ignore[arg-type]
            sandbox=self.sandbox,
            ctx=self.ctx,
            hooks=self.hooks,  # type: ignore[arg-type]
            token_counter=self.token_counter,
        )

        self.ctx._hooks = self.hooks
        self.ctx._state_manager = self.state
        self.ctx._llm = self.llm

        prior = self.state.load_summary()
        if prior:
            self.ctx.summary = prior  # type: ignore[assignment]
            logger.info(
                "Restored prior session summary: %d turns from %s",
                prior.conversation_turns, prior.compressed_at
            )

        self._assemble()

    def _resolve_runtime_config(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Resolve runtime config with 3-level priority: CLI > environment > config.toml.
        
        v0.8.2: agent.runtime removed from AgentConfig; provider/model resolution
        now uses CLI args > environment > config.toml defaults.
        """
        agent_cfg = self.spec.agent
        rt = getattr(agent_cfg, "runtime", None) if agent_cfg is not None else None

        resolved_provider = provider or (
            rt.provider if rt is not None else None
        ) or get_default_provider()

        # 4-level API key resolution: explicit arg > env > config > prompt
        if api_key:
            resolved_api_key = api_key
        else:
            resolved_api_key = resolve_api_key(resolved_provider, prompt=True)  # type: ignore[assignment]

        provider_info = get_provider(resolved_provider)

        resolved: dict[str, Any] = {
            "provider": resolved_provider,
            "model": model or (
                rt.model if rt is not None else None
            ) or provider_info.default_model,
            "api_key": resolved_api_key or "",
            "temperature": (
                rt.temperature if rt is not None else 0.7
            ),
            "max_tokens": (
                rt.max_tokens if rt is not None else 4096
            ),
        }

        resolved["base_url"] = provider_info.base_url
        resolved["context_window"] = provider_info.context_window

        agent_features = rt.features if rt is not None else {}
        provider_features = getattr(
            provider_info, "features", ProviderFeatures()
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

        # NOTE: Dynamic max_tokens based on estimated input tokens
        # https://github.com/agenthatch/agenthatch/issues
        requested_max = resolved["max_tokens"]
        estimated_input = self.ctx.estimate_input_tokens()
        if not isinstance(estimated_input, int):
            estimated_input = 0
        safe_max = max(MIN_GENERATION_TOKENS, requested_max - estimated_input)

        # Reasoning models share a single max_tokens budget
        # between reasoning and content. The old 0.7x reduction was wrong —
        # it starved content tokens. Instead, INFLATE the budget.
        agent_rt = getattr(self.spec.agent, "runtime", None) if self.spec.agent is not None else None
        user_set_max_tokens = (
            agent_rt is not None
            and agent_rt.max_tokens != 4096  # 4096 is the default
        )
        if (merged_features.supports_reasoning_content
                and not user_set_max_tokens
                and safe_max < 8192):
            inflated = int(requested_max * self.REASONING_MAX_TOKENS_MULTIPLIER)
            inflated = min(inflated, 16384)  # Cap at 16K (API limit)
            safe_max = max(MIN_GENERATION_TOKENS, inflated - estimated_input)
            logger.debug("Inflated max_tokens for reasoning model: %d", safe_max)

        if safe_max < requested_max:
            logger.info(
                "Adjusted max_tokens: %d -> %d (input estimate: %d tokens)",
                requested_max, safe_max, estimated_input,
            )
        resolved["max_tokens"] = safe_max
        return resolved

    def _ensure_token_counted(self) -> None:
        """Safety net: record usage from llm.last_usage if the loop didn't."""
        snap = self.token_counter.snapshot()
        if snap["total_tokens"] > 0:
            return
        usage = getattr(self.llm, "last_usage", None)
        if usage is None:
            return
        self.token_counter.add_usage({
            "total_tokens": getattr(usage, "total_tokens", 0),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        })
        logger.debug(
            "TokenCounter safety-net: recorded %s tokens",
            getattr(usage, "total_tokens", 0),
        )

    def _assemble(self) -> None:
        """Assemble capabilities from AHSSPEC."""
        # Extract anchor rules from instructions.rules
        self.ctx.ANCHOR_RULES = list(self.spec.instructions.rules)

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
            timeout=self.spec.base.timeout,
            env={e.name: e.description for e in self.spec.base.env},
        )

        # MCP integration
        self._mcp_client = MCPClient()
        for server_ref in self.spec.interface.mcp_servers:
            raw_command = getattr(server_ref, "command", "") or ""
            is_mcporter = raw_command == "mcporter" or raw_command.startswith("mcporter ")

            config = MCPServerConfig(
                command=raw_command,
                args=getattr(server_ref, "args", []) or [],
                env=getattr(server_ref, "env", {}) or {},
                transport="stdio" if is_mcporter else (
                    getattr(server_ref, "transport", "stdio") or "stdio"
                ),
                url=getattr(server_ref, "url", "") or "",
            )
            self._mcp_client.add_server(server_ref.name, config)
        self._mcp_client.connect_all()
        self._mcp_client.register_with_capbus(self.capbus)

        # Inject MCP server status into system prompt
        if self.spec.interface.mcp_servers:
            status_lines = ["## MCP Server Status"]
            for srv in self.spec.interface.mcp_servers:
                transport = getattr(self._mcp_client, '_transports', {}).get(srv.name)
                is_connected = transport.is_connected() if transport else False
                # Also check unavailable set
                if srv.name in getattr(self._mcp_client, '_unavailable', set()):
                    is_connected = False
                status = "AVAILABLE" if is_connected else "UNAVAILABLE"
                status_lines.append(f"- {srv.name}: {status}")
            self.ctx.mcp_status_note = "\n".join(status_lines)

        # API template registration
        http_client = self.capbus.builtins.get("http_client")
        for tpl in self.spec.interface.api_templates:
            if http_client is not None:
                executor = APITemplateExecutor(tpl, http_client)
                self.capbus.register_external_tool(
                    f"api__{tpl.name}",
                    {"type": "object", "properties": {
                        p.name: {"type": p.type} for p in tpl.params
                    }},
                    executor.execute,
                )

        # Checkpoint restore
        # Use project-local directory to avoid sandbox permission denial
        self._checkpoint_mgr = CheckpointManager(
            Path.cwd() / ".agenthatch" / "sessions" / self.spec.identity.id
        )
        # Migration: if old path has data, copy it once
        old_dir = Path.home() / ".agenthatch" / "sessions" / self.spec.identity.id
        new_dir = Path.cwd() / ".agenthatch" / "sessions" / self.spec.identity.id
        if old_dir.exists() and not new_dir.exists():
            import shutil
            try:
                shutil.copytree(old_dir, new_dir)
                logger.info("Migrated checkpoint from %s to %s", old_dir, new_dir)
            except OSError as e:
                logger.warning("Checkpoint migration failed: %s", e)
        if self._checkpoint_mgr.exists():
            cp = self._checkpoint_mgr.load()
            if cp:
                self.ctx.history = cp.history
                if cp.summary:
                    summary_dict = dict(cp.summary)
                    # Remap key_findings → key_decisions for checkpoint compat.
                    # v0.5.7/v0.5.8 early versions serialized @property key_findings
                    # into checkpoint, but CompactSummary.__init__() only accepts
                    # key_decisions. Without this remap, TypeError crashes the agent.
                    if 'key_findings' in summary_dict:
                        if 'key_decisions' not in summary_dict:
                            summary_dict['key_decisions'] = summary_dict.pop('key_findings')
                        else:
                            summary_dict.pop('key_findings')
                    # Also strip any other unknown properties that may have been serialized
                    known_fields = set(CompactSummary.__dataclass_fields__.keys())
                    unknown = set(summary_dict.keys()) - known_fields
                    for k in unknown:
                        summary_dict.pop(k, None)
                    self.ctx.summary = CompactSummary(**summary_dict)  # type: ignore[assignment]
                self.ctx._consecutive_compact_failures = cp.compact_failures
                self.ctx._turn_count = cp.turn_count
                self.loop._cb_state = cp.cb_state
                self.loop._cb_failures = cp.cb_failures
                logger.info(
                    "Restored checkpoint: %d turns, session %s",
                    cp.turn_count, cp.session_id,
                )

        self.loop._checkpoint_mgr = self._checkpoint_mgr

        # CapBus wiring to context
        self.ctx._capbus = self.capbus

        if self.spec.instructions.output_template:
            logger.info(
                "Skill '%s': output_template injected (%d chars)",
                self.spec.identity.id,
                len(self.spec.instructions.output_template),
            )

    def chat(self, user_input: str) -> str:
        """Single-turn synchronous chat."""
        if self._manifest.loop_engine == LoopKind.DIRECT:
            from agenthatch_core.bricks.loops import DirectLoop
            result = DirectLoop(
                self.llm, self.ctx, token_counter=self.token_counter
            ).run(user_input)
        else:
            result = self.loop.run(user_input)

        # v0.7.2: Ensure token was counted even if loop path missed it
        self._ensure_token_counted()

        # v0.7: OutputGuard validation
        if self._manifest.guard_active and self.guard is not None:
            cleaned, violations = self.guard.validate(result)
            if violations:
                for v in violations:
                    logger.warning("OutputGuard: %s", v)
            if cleaned:
                result = cleaned

        return result

    def chat_stream(self, user_input: str) -> Any:
        """Streaming chat for TUI consumption."""
        if self._manifest.loop_engine == LoopKind.DIRECT:
            from agenthatch_core.bricks.loops import DirectLoop
            result = yield from DirectLoop(
                self.llm, self.ctx, token_counter=self.token_counter
            ).stream(user_input)
        else:
            result = yield from self.loop.stream(user_input)

        # v0.7.2: Ensure token was counted even if stream path missed it
        self._ensure_token_counted()

        # v0.7: OutputGuard validation
        if self._manifest.guard_active and self.guard is not None:
            cleaned, violations = self.guard.validate(result)
            if violations:
                for v in violations:
                    logger.warning("OutputGuard: %s", v)
            if cleaned:
                result = cleaned

        return result
