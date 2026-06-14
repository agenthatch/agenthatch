"""AHCoreAgent — base class for agenthatch-generated independent Agents.

This is the universal chassis that all generated Agents inherit from.
It wires together LLMClient, CapBus, ConversationLoop, and
ContextManager into a ready-to-run agent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from agenthatch_core.bricks.manifest import BrickManifest, LoopKind
from agenthatch_core.bricks.stubs import _NullCapBus, _NullHooks
from agenthatch_core.config import resolve_runtime_config
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.hooks import HooksManager
from agenthatch_core.llm.client import LLMClient, ProviderFeatures
from agenthatch_core.loop.agent_loop import ConversationLoop, RichToolCallEvent
from agenthatch_core.sandbox.executor import Sandbox
from agenthatch_core.tools.bus import CapBus
from agenthatch_core.types import AgentIdentity

logger = logging.getLogger(__name__)


class AHCoreAgent:
    """Agent base class for agenthatch-generated independent Agents.

    Generated agent.py templates inherit from this class.  It provides:

    * Identity management (id, display_name, version)
    * LLM client wiring via runtime.toml or programmatic config
    * Tool bus (CapBus) with provides/requires/MCP/API template registration
    * Script execution via direct subprocess (no Docker sandbox)
    * Context manager (system prompt, history, compaction)
    * Conversation loop (chat, chat_stream)
    * Lifecycle hooks (pre/post turn, compact, tool call)
    """

    def __init__(
        self,
        identity: AgentIdentity,
        runtime_config: dict | None = None,
        spec_path: Path | None = None,
        tools: list[Callable] | None = None,
        knowledge: Any | None = None,
        brick_manifest: BrickManifest | None = None,
    ):
        self.identity = identity
        self.llm: LLMClient | None = None
        self._agent_root: Path | None = spec_path.parent if spec_path else None
        self._knowledge = knowledge

        # v0.7: BrickManifest drives chassis decomposition
        self._manifest = brick_manifest or BrickManifest()

        # Assemble bricks — use null stubs for disabled features
        self.capbus: Any = CapBus() if self._manifest.capbus else _NullCapBus()
        # v0.8: Sandbox always enabled — direct subprocess execution (no Docker)
        self.sandbox: Any = Sandbox()
        self.hooks: Any = HooksManager() if self._manifest.hooks else _NullHooks()

        # v0.8.1: Whitelist removed — agent has full CLI capability

        # OutputGuard (v0.7) — compiled regex validators from ANCHOR_RULES
        self.guard: Any = self._manifest.guard

        # v0.7.6: Wire guard to CapBus for pre-tool validation
        if self._manifest.guard_active and self.guard is not None:
            self.capbus._guard = self.guard

        # CredentialVault + APITemplateExecutor (v0.7)
        self.vault: Any = None
        if self._manifest.credential_vault:
            from agenthatch_core.bricks.credential_vault import CredentialVault
            self.vault = CredentialVault()

        # FileProcessor (v0.7)
        self.file_processor: Any = None
        if self._manifest.file_processor:
            from agenthatch_core.bricks.file_processor import FileProcessor
            self.file_processor = FileProcessor()

        # TokenCounter (v0.7)
        from agenthatch_core.loop.token_counter import TokenCounter
        self.token_counter = TokenCounter()

        # MemoryBrick (v0.7.6) — persistent memory, default-on
        self._memory: Any = None
        if self._manifest.memory:
            try:
                from agenthatch_core.bricks.memory import MemoryBrick
                self._memory = MemoryBrick(identity.id)
                # Inject recall tool into CapBus
                from agenthatch_core.bricks.memory.tools import recall_tool
                recall = recall_tool(self._memory)
                self.capbus.inject_builtin("recall", recall)
            except ImportError:
                logger.warning("MemoryBrick not available; memory disabled for this session.")

        # Build raw spec from yaml or fallback constants
        self._raw_spec = self._build_raw_spec(identity, spec_path)

        # Context manager (needs spec before runtime config)
        self.ctx = ContextManager(spec=self._raw_spec)

        # v0.7.12: Wire hooks into ContextManager for compaction events
        self.ctx._hooks = self.hooks

        # v0.7.12: Create CheckpointManager for conversation persistence
        self._checkpoint_mgr: Any = None
        try:
            from agenthatch_core.offload.checkpoint import CheckpointManager
            self._checkpoint_mgr = CheckpointManager(
                Path(".agenthatch") / "checkpoints" / identity.id
            )
        except ImportError:
            logger.debug("CheckpointManager not available.")

        # v0.7.12: Wire StateManager into ContextManager for history offload
        try:
            from agenthatch_core.offload.state_manager import StateManager
            self.ctx._state_manager = StateManager(
                Path(".agenthatch") / "history" / identity.id
            )
        except ImportError:
            logger.debug("StateManager not available.")

        # v0.7.6: Wire memory brick into context manager for system prompt injection
        if self._memory is not None:
            self.ctx._memory = self._memory

        # Apply runtime config (creates LLM client)
        if runtime_config:
            self._apply_runtime_config(runtime_config)

        # Register tools from spec
        self._build_runtime_tools()

        # Register user-provided tools
        if tools:
            for tool in tools:
                self._register_python_tool(tool)

    # ── spec loading ──────────────────────────────────────────────────

    def _build_raw_spec(
        self, identity: AgentIdentity, spec_path: Path | None
    ) -> dict:
        """Build complete spec dict for ContextManager + tool registration.

        Priority: agenthatch.yaml is the canonical source.  Python constants
        (WORKFLOW etc.) are fallback only when yaml is unavailable.
        """
        import yaml

        if spec_path and spec_path.exists():
            spec = yaml.safe_load(spec_path.read_text()) or {}
            spec.setdefault("identity", {}).update({
                "id": identity.id,
                "display_name": identity.display_name,
                "version": identity.version,
            })
            return spec

        # Fallback: build minimal spec from class-level constants
        return {
            "identity": {
                "id": identity.id,
                "display_name": identity.display_name,
                "version": identity.version,
            },
            "intent": {"summary": self.__doc__ or ""},
            "instructions": {
                "workflow": getattr(self.__class__, "WORKFLOW", ""),
                "rules": getattr(self.__class__, "ANCHOR_RULES", []),
                "output_template": getattr(
                    self.__class__, "OUTPUT_TEMPLATE", ""
                ),
            },
            "interface": {
                "provides": [],
                "requires": [],
                "mcp_servers": [],
                "api_templates": [],
            },
            "resources": {"scripts": [], "references": []},
        }

    # ── runtime config ────────────────────────────────────────────────

    def _apply_runtime_config(self, config: dict) -> None:
        """Consume runtime.toml fields to wire up LLM client."""
        llm_cfg = config.get("llm", {})

        # Pass context_window from config if available, else from BrickManifest
        context_window = llm_cfg.get("context_window")
        if context_window is None and "context_window" in config:
            context_window = config.get("context_window")

        self.llm = LLMClient(
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("model", "gpt-4o"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            temperature=llm_cfg.get("temperature"),
            max_tokens=llm_cfg.get("max_tokens"),
            context_window=context_window,
        )

        # Merge provider features from config
        features_cfg = config.get("features", {})
        if features_cfg and self.llm:
            self.llm._features = ProviderFeatures(**features_cfg)

        # Compact config
        compact_cfg = config.get("compact", {})
        if compact_cfg:
            self.ctx.compact_config = compact_cfg
            self.ctx._apply_compact_config()

        # Give ContextManager an LLM reference for compaction
        if self.llm is not None:
            self.ctx._llm = self.llm

    # ── tool registration ─────────────────────────────────────────────

    def _build_runtime_tools(self) -> None:
        """Register tools from spec: provides, requires, MCP, API templates.

        Called automatically by __init__ after spec + runtime are ready.
        """
        spec = self._raw_spec
        interface = spec.get("interface", {})
        provides = interface.get("provides", [])
        requires = interface.get("requires", [])
        mcp_servers = interface.get("mcp_servers", [])
        api_templates = interface.get("api_templates", [])
        resources = spec.get("resources", {})
        instructions = spec.get("instructions", {})

        # Build capability → script mapping from resources + workflow
        scripts_dir: Path | None = None
        if self._agent_root:
            scripts_dir = self._agent_root / "skills" / "scripts"
        cap_to_script = _build_cap_to_script(
            provides, resources, instructions, scripts_dir,
        )

        # 1. provides → tool executors
        # v0.7.15: Priority order when multiple runtime backends are available:
        #   MCP servers (MCPProxyExecutor) > direct subprocess scripts > CLI > description-only
        sandbox_usable = isinstance(self.sandbox, Sandbox)
        has_mcp = bool(mcp_servers)

        for cap in provides:
            cap_name = cap.get("capability", cap.get("name", ""))
            if not cap_name:
                continue
            schema = cap.get("input_schema", cap.get("schema", {}))
            output_schema = cap.get("output_schema")

            if has_mcp:
                # v0.7.15: MCP configured → use MCPProxyExecutor (top priority).
                # Sandbox (if also usable) is kept for warmup scripts but not
                # used for tool execution — mcporter handles the transport.
                server_name = mcp_servers[0].get("name", "")
                # v0.8.13: Extract MCP config directly from server entry.
                # Harness stores url/transport/command at top level, not
                # inside a nested "config" key. Build config dict from entry.
                mcp_cfg = {
                    "url": mcp_servers[0].get("url", ""),
                    "transport": mcp_servers[0].get("transport", "stdio"),
                    "command": mcp_servers[0].get("command", ""),
                    "headers": mcp_servers[0].get("headers", {}),
                    "auth_token": mcp_servers[0].get("auth_token", ""),
                    "timeout": mcp_servers[0].get("timeout", 30.0),
                }
                # Merge any nested config if present
                nested = mcp_servers[0].get("config", {})
                if isinstance(nested, dict) and nested:
                    mcp_cfg.update(nested)
                executor = MCPProxyExecutor(
                    cap_name=cap_name,
                    server_name=server_name,
                    mcp_config=mcp_cfg,
                    script_name=cap_to_script.get(cap_name),
                )
                self.capbus.register(
                    name=cap_name,
                    executor=executor.execute,
                    schema={
                        "name": cap_name,
                        "description": cap.get("description", cap_name),
                        "parameters": schema.get("parameters", schema),
                    },
                    source="spec",
                )
            elif sandbox_usable:
                script_name = cap_to_script.get(cap_name)
                # v0.8.21: Verify script exists before registering sandbox executor.
                # If the script doesn't exist, register as description-only so that
                # _register_python_tool() can provide the real Python implementation.
                # This fixes the bug where non-existent sandbox scripts (e.g.,
                # "python create_docx.py") shadow real Python tool implementations.
                script_exists = False
                if script_name and scripts_dir and (scripts_dir / script_name).exists():
                    script_exists = True
                elif scripts_dir and scripts_dir.is_dir():
                    # Legacy fallback: check if {cap_name}.py exists
                    script_exists = (scripts_dir / f"{cap_name}.py").exists()
                if script_exists:
                    executor = _provide_script_executor(
                        cap_name, self.sandbox, self._agent_root,
                        script_name=script_name,
                    )
                    self.capbus.register(
                        name=cap_name,
                        executor=executor,
                        schema={
                            "name": cap_name,
                            "description": cap.get("description", cap_name),
                            "parameters": schema.get("parameters", schema),
                        },
                        source="spec",
                    )
                else:
                    # No valid script — register as description-only.
                    # Python tools (registered later via _register_python_tool)
                    # will provide the real executor.
                    self.capbus.register(
                        name=cap_name,
                        schema={
                            "name": cap_name,
                            "description": cap.get("description", cap_name),
                            "parameters": schema.get("parameters", schema),
                        },
                        source="spec",
                    )
            else:
                # v0.8: External skill agent (direct subprocess, no MCP)
                # Use CLIExecutor for CLI-based capabilities
                script_name = cap_to_script.get(cap_name)
                if script_name:
                    executor = CLIExecutor(
                        cap_name=cap_name,
                        cli_command=script_name,
                    )
                    self.capbus.register(
                        name=cap_name,
                        executor=executor.execute,
                        schema={
                            "name": cap_name,
                            "description": cap.get("description", cap_name),
                            "parameters": schema.get("parameters", schema),
                        },
                        source="spec",
                    )
                else:
                    # Last resort: register as description-only
                    self.capbus.register(
                        name=cap_name,
                        schema={
                            "name": cap_name,
                            "description": cap.get("description", cap_name),
                            "parameters": schema.get("parameters", schema),
                        },
                        source="spec",
                    )

            # v0.7.6: Register output_schema for tool output validation
            if output_schema:
                self.capbus._output_schemas[cap_name] = output_schema

        # 2. requires → builtin injection or mark unavailable
        for req in requires:
            req_name = req.get("capability", req.get("name", ""))
            if not req_name:
                continue
            # Try builtin registry first
            builtin = _lookup_builtin(req_name)
            if builtin is not None:
                self.capbus.inject_builtin(req_name, builtin)
            else:
                self.capbus.mark_unavailable(req_name)

        # 3. MCP servers → connect and register tools
        # v0.7.15: Skip when provides were already registered via MCPProxyExecutor
        # (Step 1, has_mcp guard).  Only run when provides list is empty (server-side
        # tool discovery) to avoid double registration.
        if not provides:
            for mcp_cfg in mcp_servers:
                _register_mcp_tools(self.capbus, mcp_cfg)

        # 4. API templates → api__<name> tools
        for tmpl in api_templates:
            name = tmpl.get("name", "")
            if not name:
                continue
            from agenthatch_core.bricks.api_executor import APITemplateExecutor
            executor = APITemplateExecutor.from_template(tmpl, vault=self.vault)
            self.capbus.register_external_tool(
                f"api__{name}",
                tmpl.get("schema", {}),
                executor.execute,
            )

    def _register_python_tool(self, tool: Callable) -> None:
        """Register a plain Python function as a tool on the CapBus.

        v0.8.13: Never overwrite an executor already registered from spec
        (e.g. MCPProxyExecutor). Python tools are fallback only.
        """
        import inspect

        # v0.8.13: If a spec executor already exists (e.g. MCPProxyExecutor),
        # skip — Python stub functions are fallback only.
        existing = self.capbus.capabilities.get(tool.__name__)
        if existing is not None and existing.executor is not None:
            logger.debug(
                "Tool '%s' already has executor from spec, skipping Python fallback",
                tool.__name__,
            )
            return

        sig = inspect.signature(tool)
        params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for pname, param in sig.parameters.items():
            ptype = "string"
            if param.annotation is not inspect.Parameter.empty:
                anno = param.annotation
                if anno is int:
                    ptype = "integer"
                elif anno is float:
                    ptype = "number"
                elif anno is bool:
                    ptype = "boolean"
            params["properties"][pname] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                params["required"].append(pname)

        self.capbus.register(
            name=tool.__name__,
            executor=lambda args, _t=tool: str(_t(**args)),
            schema={
                "name": tool.__name__,
                "description": (tool.__doc__ or "").strip().split("\n")[0],
                "parameters": params,
                "output_schema": {"type": "string"},
            },
            source="user",
        )
        # v0.8.21: Also register output_schema in _output_schemas dict
        # so _validate_output() knows this tool returns plain text, not JSON.
        self.capbus._output_schemas[tool.__name__] = {"type": "string"}

    # ── conversation API ──────────────────────────────────────────────

    def attach_file(self, filepath: str) -> str:
        """v0.7.12: Process a file and inject its content into context.

        Used by the /attach command in the TUI.
        """
        if self.file_processor is None:
            return "[warn]File processor not available.[/warn]"
        try:
            result = self.file_processor.process(Path(filepath).expanduser())
        except Exception as e:
            return f"[error]Failed to process file: {e}[/error]"
        if result.error:
            return f"[warn]{result.error}[/warn]"
        self.ctx.add_to_history(
            "system",
            f"[Attached: {result.path.name}]\n{result.chunks[0].content}"
        )
        return f"[ok]Attached {result.path.name} ({result.total_chars:,} chars)[/ok]"

    def chat(self, user_input: str) -> str:
        """Single-turn synchronous chat."""
        if self.llm is None:
            raise RuntimeError(
                "LLM client not initialized. Provide runtime_config."
            )

        # v0.7: Loop dispatch from BrickManifest
        if self._manifest.loop_engine == LoopKind.DIRECT:
            from agenthatch_core.bricks.loops import DirectLoop
            result = DirectLoop(
                self.llm, self.ctx,
                token_counter=self.token_counter,
                memory_brick=self._memory,
                hooks=self.hooks,
            ).run(user_input)
        elif self._manifest.loop_engine == LoopKind.PLAN_GUIDED:
            # v0.7.12: PLAN_GUIDED reserved for v0.8 PlanLayer.
            # Falls through to standard ConversationLoop for now.
            loop = ConversationLoop(
                self.llm, self.capbus, self.sandbox, self.ctx,
                hooks=self.hooks,
                token_counter=self.token_counter,
                memory_brick=self._memory,
                checkpoint_mgr=self._checkpoint_mgr,
            )
            result = loop.run(user_input)
        else:
            loop = ConversationLoop(
                self.llm, self.capbus, self.sandbox, self.ctx,
                hooks=self.hooks,
                token_counter=self.token_counter,
                memory_brick=self._memory,  # v0.7.12: wire memory brick
                checkpoint_mgr=self._checkpoint_mgr,  # v0.7.12: wire checkpointing
            )
            result = loop.run(user_input)

        # v0.7: OutputGuard validation
        if self._manifest.guard_active and self.guard is not None:
            cleaned, violations = self.guard.validate(result)
            if violations:
                for v in violations:
                    logger.warning("OutputGuard: %s", v)
            if cleaned:
                result = cleaned

        return result

    def chat_stream(
        self, user_input: str
    ) -> Generator[RichToolCallEvent | str, None, str]:
        """Streaming chat for TUI Live rendering."""
        if self.llm is None:
            raise RuntimeError(
                "LLM client not initialized. Provide runtime_config."
            )

        # v0.7: Loop dispatch from BrickManifest
        if self._manifest.loop_engine == LoopKind.DIRECT:
            from agenthatch_core.bricks.loops import DirectLoop
            result = yield from DirectLoop(
                self.llm, self.ctx,
                token_counter=self.token_counter,
                memory_brick=self._memory,
                hooks=self.hooks,
            ).stream(user_input)
        elif self._manifest.loop_engine == LoopKind.PLAN_GUIDED:
            # v0.7.12: PLAN_GUIDED reserved for v0.8 PlanLayer.
            # Falls through to standard ConversationLoop for now.
            loop = ConversationLoop(
                self.llm, self.capbus, self.sandbox, self.ctx,
                hooks=self.hooks,
                token_counter=self.token_counter,
                memory_brick=self._memory,
                checkpoint_mgr=self._checkpoint_mgr,
            )
            result = yield from loop.stream(user_input)
        else:
            loop = ConversationLoop(
                self.llm, self.capbus, self.sandbox, self.ctx,
                hooks=self.hooks,
                token_counter=self.token_counter,
                memory_brick=self._memory,  # v0.7.12: wire memory brick
                checkpoint_mgr=self._checkpoint_mgr,  # v0.7.12: wire checkpointing
            )
            result = yield from loop.stream(user_input)

        # v0.7: OutputGuard validation (on final result only)
        if self._manifest.guard_active and self.guard is not None:
            cleaned, violations = self.guard.validate(result)
            if violations:
                for v in violations:
                    logger.warning("OutputGuard: %s", v)
            if cleaned:
                result = cleaned

        return result

    # ── classmethod constructors ──────────────────────────────────────

    @classmethod
    def from_spec(
        cls, ahspec: dict, runtime_config: dict | None = None
    ) -> AHCoreAgent:
        """Programmatic entry point — build Agent from a spec dict.

        ``ahspec`` can be an agenthatch.yaml dict or an AHSSpec object.
        Identity is extracted from ``ahspec["identity"]``.
        """
        ident = ahspec.get("identity", {})
        if isinstance(ident, dict):
            agent = cls(
                identity=AgentIdentity(
                    id=ident.get("id", "unknown"),
                    display_name=ident.get(
                        "display_name", ident.get("id", "Agent")
                    ),
                    version=ident.get("version", ""),
                ),
                runtime_config=runtime_config,
            )
        else:
            # Object with attributes (AHSSpec)
            agent = cls(
                identity=AgentIdentity(
                    id=getattr(ident, "id", "unknown"),
                    display_name=getattr(
                        ident, "display_name", getattr(ident, "id", "Agent")
                    ),
                    version=getattr(ident, "version", ""),
                ),
                runtime_config=runtime_config,
            )
        if isinstance(ahspec, dict):
            agent._raw_spec.update(ahspec)
            # H4 fix: rebuild runtime tools AFTER spec is populated.
            # _build_runtime_tools() in __init__ runs against the empty
            # fallback spec; we must re-run it with the real spec so
            # provides, requires, and MCP servers are registered.
            agent._build_runtime_tools()
        return agent


# ── internal helpers ──────────────────────────────────────────────────

def _provide_script_executor(
    tool_name: str, sandbox: Sandbox, agent_root: Path | None,
    script_name: str | None = None,
) -> Callable[[dict], str]:
    """Create a Sandbox executor for a 'provides' capability.

    If script_name is provided, uses that as the command (looked up from
    resources.scripts or workflow steps). Otherwise falls back to the
    legacy behaviour of running 'python {tool_name}.py'.
    """
    def execute(arguments: dict) -> str:
        env = {f"AH_ARG_{k.upper()}": str(v) for k, v in arguments.items()}
        script_dir = (
            agent_root / "skills" / "scripts" if agent_root else None
        )

        if script_name and script_dir and script_dir.is_dir():
            script_file = script_dir / script_name
            if script_file.exists():
                cmd = str(script_file)
                cwd = str(script_dir)
            else:
                return (
                    f"Error: script '{script_name}' not found in {script_dir}. "
                    f"Available: {sorted(p.name for p in script_dir.iterdir())}"
                )
        elif script_name:
            return f"Error: script directory not found for capability '{tool_name}'"
        else:
            # Legacy fallback: use tool_name as filename
            cwd = str(script_dir) if script_dir and script_dir.is_dir() else None
            cmd = f"python {tool_name}.py"

        result = sandbox.run(cmd, cwd=cwd, env=env)
        return result.stdout
    return execute


def _build_cap_to_script(
    provides: list[dict],
    resources: dict,
    instructions: dict,
    scripts_dir: Path | None,
) -> dict[str, str]:
    """Build a mapping from capability name to script filename.

    Examines workflow steps, resources.scripts, and the scripts directory
    to find which script file implements each provided capability.
    """
    cap_to_script: dict[str, str] = {}
    cap_names = {c.get("capability", c.get("name", "")) for c in provides}
    cap_names.discard("")

    # Approach 1: workflow steps that mention a capability name + have a script
    workflow = instructions.get("workflow", [])
    if isinstance(workflow, list):
        for step in workflow:
            if not isinstance(step, dict):
                continue
            script = step.get("script")
            if not script:
                continue
            desc = step.get("description", "").lower()
            for cap_name in sorted(cap_names):
                if cap_name in cap_to_script:
                    continue
                if cap_name.replace("_", " ") in desc or cap_name in desc:
                    cap_to_script[cap_name] = script
                    break

    # Approach 2: resources.scripts entries
    res_scripts = resources.get("scripts", [])
    if isinstance(res_scripts, list):
        for entry in res_scripts:
            if not isinstance(entry, dict):
                continue
            script_name = entry.get("name", "")
            if not script_name:
                continue
            script_stem = Path(script_name).stem
            for cap_name in cap_names:
                if cap_name in cap_to_script:
                    continue
                cap_flat = cap_name.replace("_", "")
                stem_flat = script_stem.replace("_", "").replace("-", "")
                if cap_flat in stem_flat or stem_flat in cap_flat:
                    cap_to_script[cap_name] = script_name
                    break

    # Approach 3: direct filename matching from scripts directory
    if scripts_dir and scripts_dir.is_dir():
        for cap_name in cap_names:
            if cap_name in cap_to_script:
                continue
            for ext in (".py", ".sh", ".js", ".rb"):
                candidate = f"{cap_name}{ext}"
                if (scripts_dir / candidate).exists():
                    cap_to_script[cap_name] = candidate
                    break
            if cap_name in cap_to_script:
                continue
            # Fuzzy: script stem contains or is contained by capability name
            cap_flat = cap_name.replace("_", "")
            for sf in scripts_dir.iterdir():
                if not sf.is_file():
                    continue
                stem_flat = sf.stem.replace("_", "").replace("-", "")
                if len(stem_flat) >= 4 and (
                    cap_flat in stem_flat or stem_flat in cap_flat
                ):
                    cap_to_script[cap_name] = sf.name
                    break

    return cap_to_script


def _lookup_builtin(name: str) -> Any | None:
    """Look up a builtin tool by name.  Returns instance or None.

    Checks agenthatch-core registry (extensibility point)."""
    try:
        from agenthatch_core.tools.builtins import BUILTIN_REGISTRY
        cls = BUILTIN_REGISTRY.get(name)
        if cls is not None:
            return cls()
    except ImportError:
        pass

    return None


def _register_mcp_tools(capbus: CapBus, mcp_cfg: dict) -> None:
    """Connect to an MCP server and register its tools on the CapBus."""
    try:
        from agenthatch_core.tools.mcp_loader import load_mcp_tools
        load_mcp_tools(capbus, mcp_cfg)
    except ImportError:
        logger.warning(
            "MCP loader not available; skipping MCP server: %s",
            mcp_cfg.get("name", "unknown"),
        )


def _api_template_executor(tmpl: dict) -> Callable[..., str]:
    """Create an executor for an API template."""
    import urllib.request
    import json as _json

    url = tmpl.get("url", "")
    method = tmpl.get("method", "GET").upper()
    auth_env = tmpl.get("auth_env_var")

    def execute(**kwargs: Any) -> str:
        import os

        headers = dict(tmpl.get("headers", {}))
        if auth_env:
            token = os.environ.get(auth_env, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        req_url = url
        if method == "GET" and kwargs:
            import urllib.parse
            qs = urllib.parse.urlencode(kwargs)
            req_url = f"{url}?{qs}"

        req = urllib.request.Request(req_url, method=method, headers=headers)
        if method in ("POST", "PUT", "PATCH") and kwargs:
            data = _json.dumps(kwargs).encode("utf-8")
            req = urllib.request.Request(
                req_url, data=data, method=method, headers=headers
            )
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            return f"API call failed: {e}"

    return execute


# ── v0.7.11: Proxy executors for MCP-only and external-skill agents ────

# v0.8.12: LRU connection cache — claude-code pattern (memoizeWithLRU).
# Avoids repeated mcporter probes on every tool call.
# Key: (server_name, cap_name) → value: bool (True = mcporter is available)
_mcp_avail_cache: dict[tuple[str, str], bool] = {}
_MCP_CACHE_MAX = 64  # max cache entries


class MCPProxyExecutor:
    """Proxy executor for MCP-only agents.

    v0.8.12: Adopts claude-code MCP patterns:
      - LRU connection caching (_mcp_avail_cache) — avoids repeated probes
      - Retry with exponential backoff for transient failures
      - Structured error categorization (permanent vs transient)
      - Graceful degradation: always returns string, never throws
      - Falls back to CLI script execution if MCP is unavailable

    Routes tool calls through mcporter CLI with graceful degradation.
    Tries MCP server connection first; falls back to CLI script execution
    if MCP is unavailable.
    """

    _RETRY_MAX = 2          # max retries for transient failures
    _RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry

    def __init__(
        self,
        cap_name: str,
        server_name: str = "",
        mcp_config: dict | None = None,
        script_name: str | None = None,
    ):
        self.cap_name = cap_name
        self.server_name = server_name
        self.mcp_config = mcp_config or {}
        self.script_name = script_name

    def execute(self, arguments: dict) -> str:
        """Execute capability, trying MCP first then CLI fallback."""
        # Try MCP server connection
        if self.mcp_config:
            try:
                return self._execute_via_mcp(arguments)
            except Exception as e:
                logger.debug(
                    "MCPProxyExecutor: MCP call failed for %s: %s",
                    self.cap_name, e,
                )

        # Fall back to CLI script execution
        if self.script_name:
            try:
                return self._execute_via_cli(arguments)
            except Exception as e:
                return (
                    f"Error: capability '{self.cap_name}' failed: {e}. "
                    f"MCP server not connected and CLI fallback failed."
                )

        return (
            f"Error: capability '{self.cap_name}' is not available. "
            f"MCP server '{self.server_name}' not connected and "
            f"no CLI fallback found."
        )

    def _execute_via_mcp(self, arguments: dict) -> str:
        """Execute through mcporter MCP client.

        v0.8.12: Full claude-code MCP patterns:
          - LRU cache for mcporter availability (avoids repeated probes)
          - Retry with exponential backoff for transient failures
          - Structured error categorization (permanent vs transient)
          - Graceful degradation: returns error string, never throws
          - Dot notation: mcporter call KnowledgeBase.listDocuments
        """
        import subprocess
        import shutil
        import time

        # v0.8.12: LRU cache probe — claude-code ensureConnectedClient pattern
        cache_key = (self.server_name, self.cap_name)
        if cache_key in _mcp_avail_cache:
            if not _mcp_avail_cache[cache_key]:
                return (
                    f"MCP server '{self.server_name}' is unavailable "
                    f"(cached from previous attempt). Check mcporter configuration."
                )

        # Quick probe — is mcporter even installed?
        if not shutil.which("mcporter"):
            _mcp_avail_cache[cache_key] = False
            self._evict_lru_if_needed()
            return (
                f"mcporter CLI not installed. "
                f"Install with: npm install -g mcporter\n"
                f"Then configure MCP server '{self.server_name}' "
                f"in ~/config/mcporter.json"
            )

        # Determine the mcporter server.tool selector
        mcp_tool = f"{self.server_name}.{self.cap_name}"
        if self.script_name:
            parts = self.script_name.split()
            for i, part in enumerate(parts):
                if part == "call" and i + 1 < len(parts):
                    mcp_tool = parts[i + 1]
                    break

        args_list = ["mcporter", "call", mcp_tool]
        for k, v in arguments.items():
            args_list.append(f"{k}={v}")

        # v0.8.12: Retry loop with exponential backoff (claude-code pattern)
        last_error: str | None = None
        for attempt in range(self._RETRY_MAX + 1):
            try:
                result = subprocess.run(
                    args_list,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                last_error = (
                    f"MCP call to '{mcp_tool}' timed out after 30s. "
                    f"Check mcporter is running and the MCP server is responsive."
                )
                if self._should_retry("timeout", attempt):
                    time.sleep(self._RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                return last_error
            except FileNotFoundError:
                _mcp_avail_cache[cache_key] = False
                return (
                    f"mcporter CLI not found. "
                    f"Install with: npm install -g mcporter"
                )
            except Exception as e:
                last_error = f"MCP call failed: {e}"
                if self._should_retry("exception", attempt):
                    time.sleep(self._RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                return last_error

            if result.returncode != 0:
                stderr = result.stderr.strip()
                error_type = self._classify_mcp_error(stderr)

                if error_type == "auth":
                    # Permanent — clear cache so next call re-authenticates
                    _mcp_avail_cache.pop(cache_key, None)
                    return (
                        f"MCP server '{self.server_name}' returned 401 Unauthorized. "
                        f"Re-authenticate at mcphub and update your token."
                    )
                if error_type == "transient":
                    last_error = (
                        f"MCP server '{self.server_name}' is unreachable. "
                        f"Check network/VPN and server URL."
                    )
                    if self._should_retry("transient", attempt):
                        time.sleep(self._RETRY_BASE_DELAY * (2 ** attempt))
                        continue
                    return last_error
                if error_type == "not_found":
                    return (
                        f"MCP tool '{mcp_tool}' not recognized by server. "
                        f"Mcporter output: {stderr[:200]}"
                    )
                # Unknown error
                return f"mcporter exited with {result.returncode}: {stderr[:200]}"

            # Success — cache the availability
            _mcp_avail_cache[cache_key] = True
            self._evict_lru_if_needed()
            return result.stdout or "(empty response)"

        return last_error or f"MCP call to '{mcp_tool}' failed after {self._RETRY_MAX + 1} attempts"

    @staticmethod
    def _classify_mcp_error(stderr: str) -> str:
        """Classify MCP error into category for retry decision.

        v0.8.12: claude-code pattern — distinguishes permanent vs transient errors.
        Returns: "auth" | "transient" | "not_found" | "unknown"
        """
        stderr_lower = stderr.lower()
        if "401" in stderr or "unauthorized" in stderr_lower:
            return "auth"
        if "econnrefused" in stderr_lower or "connection refused" in stderr_lower:
            return "transient"
        if "timeout" in stderr_lower or "timed out" in stderr_lower:
            return "transient"
        if "econnreset" in stderr_lower or "connection reset" in stderr_lower:
            return "transient"
        if "not found" in stderr_lower or "unknown" in stderr_lower:
            return "not_found"
        return "unknown"

    def _should_retry(self, error_type: str, attempt: int) -> bool:
        """Decide whether to retry based on error type and attempt number.

        v0.8.12: claude-code pattern — only retry transient errors.
        """
        if attempt >= self._RETRY_MAX:
            return False
        return error_type in ("timeout", "transient", "exception")

    @staticmethod
    def _evict_lru_if_needed() -> None:
        """Evict oldest cache entry if cache exceeds max size."""
        if len(_mcp_avail_cache) > _MCP_CACHE_MAX:
            # Pop first item (oldest insertion in Python 3.7+)
            oldest = next(iter(_mcp_avail_cache))
            _mcp_avail_cache.pop(oldest, None)

    def _execute_via_cli(self, arguments: dict) -> str:
        """Execute as a CLI script directly.

        v0.7.15: Splits script_name into command + args (was: treated entire
        string as a single command name, causing 'No such file or directory').
        """
        import subprocess

        cmd = self.script_name.split() if self.script_name else []
        for k, v in arguments.items():
            cmd.extend([f"--{k}", str(v)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout or result.stderr or "(no output)"


class CLIExecutor:
    """Execute capabilities as CLI commands for direct-execution agents.

    Used by external skill agents (e.g., agent-browser) that provide
    capabilities via CLI tools rather than direct subprocess scripts or MCP servers.
    """

    def __init__(self, cap_name: str, cli_command: str):
        self.cap_name = cap_name
        self.cli_command = cli_command

    def execute(self, arguments: dict) -> str:
        """Run the CLI command with tool arguments."""
        import subprocess

        cmd = self.cli_command.split()
        for k, v in arguments.items():
            cmd.extend([f"--{k}", str(v)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout or result.stderr or "(no output)"
        except FileNotFoundError:
            return (
                f"Error: CLI tool '{self.cli_command}' not found. "
                f"Capability '{self.cap_name}' requires this tool to be installed."
            )
        except Exception as e:
            return f"Error executing '{self.cap_name}': {e}"