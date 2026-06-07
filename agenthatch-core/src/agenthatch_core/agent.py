"""AHCoreAgent — base class for agenthatch-generated independent Agents.

This is the universal chassis that all generated Agents inherit from.
It wires together LLMClient, CapBus, Sandbox, ConversationLoop, and
ContextManager into a ready-to-run agent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from agenthatch_core.bricks.manifest import BrickManifest, LoopKind, SandboxTier
from agenthatch_core.bricks.stubs import _NullCapBus, _NullSandbox, _NullHooks
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
    * Sandbox for script execution
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
        self.sandbox: Any = Sandbox() if self._manifest.sandbox != SandboxTier.NONE else _NullSandbox()
        self.hooks: Any = HooksManager() if self._manifest.hooks else _NullHooks()

        # Apply sandbox tier
        if self._manifest.sandbox != SandboxTier.NONE and hasattr(self.sandbox, 'configure'):
            from agenthatch_core.bricks.sandboxes import SandboxWhitelist
            whitelist = SandboxWhitelist.from_tier(self._manifest.sandbox)
            self.sandbox._ALLOWED_COMMANDS = whitelist.commands

        # OutputGuard (v0.7) — compiled regex validators from ANCHOR_RULES
        self.guard: Any = self._manifest.guard

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

        # Build raw spec from yaml or fallback constants
        self._raw_spec = self._build_raw_spec(identity, spec_path)

        # Context manager (needs spec before runtime config)
        self.ctx = ContextManager(spec=self._raw_spec)

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

        # 1. provides → Sandbox script executors
        for cap in provides:
            cap_name = cap.get("capability", cap.get("name", ""))
            if not cap_name:
                continue
            schema = cap.get("input_schema", cap.get("schema", {}))
            executor = _provide_script_executor(
                cap_name, self.sandbox, self._agent_root,
                script_name=cap_to_script.get(cap_name),
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
        """Register a plain Python function as a tool on the CapBus."""
        import inspect

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
            },
            source="user",
        )

    # ── conversation API ──────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """Single-turn synchronous chat."""
        if self.llm is None:
            raise RuntimeError(
                "LLM client not initialized. Provide runtime_config."
            )

        # v0.7: Loop dispatch from BrickManifest
        if self._manifest.loop_engine == LoopKind.DIRECT:
            from agenthatch_core.bricks.loops import DirectLoop
            result = DirectLoop(self.llm, self.ctx).run(user_input)
        else:
            loop = ConversationLoop(self.llm, self.capbus, self.sandbox, self.ctx)
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
            result = yield from DirectLoop(self.llm, self.ctx).stream(user_input)
        else:
            loop = ConversationLoop(self.llm, self.capbus, self.sandbox, self.ctx)
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
                    version=ident.get("version", "0.1.0"),
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
                    version=getattr(ident, "version", "0.1.0"),
                ),
                runtime_config=runtime_config,
            )
        if isinstance(ahspec, dict):
            agent._raw_spec.update(ahspec)
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
            for cap_name in cap_names:
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

    Checks agenthatch-core registry first (extensibility point), then
    falls back to the canonical agenthatch.agent.builtins registry which
    contains the actual implementations (http_client, bash_runtime, etc).
    """
    try:
        from agenthatch_core.tools.builtins import BUILTIN_REGISTRY
        cls = BUILTIN_REGISTRY.get(name)
        if cls is not None:
            return cls()
    except ImportError:
        pass

    # Fall back to agenthatch builtins (canonical registry)
    try:
        from agenthatch.agent.builtins import BUILTIN_REGISTRY as AH_BUILTINS
        cls = AH_BUILTINS.get(name)
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