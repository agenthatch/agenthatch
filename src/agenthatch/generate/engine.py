"""GenerateEngine — Phase 3: Agent generation from AHSSPEC via Jinja2 templates.

Extracts variables from AHSSPEC and renders Jinja2 templates to produce
a self-contained, independently-runnable Agent directory.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jinja2

logger = logging.getLogger(__name__)

# Template file → output file mapping (relative to agent output root)
TEMPLATE_MAP: dict[str, str] = {
    "pyproject.toml.j2": "pyproject.toml",
    "agent.py.j2": "src/{package_name}/agent.py",
    "tools.py.j2": "src/{package_name}/tools.py",
    "references.py.j2": "src/{package_name}/references.py",
    "runtime.toml.j2": "runtime.toml",
    "README.md.j2": "README.md",
}


def _json_type_to_python(json_type: str) -> str:
    """Map JSON Schema type to Python type annotation."""
    mapping = {
        "string": "str",
        "number": "int",
        "float": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return mapping.get(json_type, "Any")


class GenerateEngine:
    """Renders Jinja2 templates from AHSSPEC variables to produce an Agent directory."""

    def __init__(self, template_dir: Path | None = None):
        """
        Args:
            template_dir: Path to the templates directory.
                          Defaults to the bundled templates/ next to this file.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self._template_dir = template_dir
        self._env = self._build_env()

    def _build_env(self) -> jinja2.Environment:
        """Create Jinja2 environment with custom filters."""
        loader = jinja2.FileSystemLoader(str(self._template_dir))
        env = jinja2.Environment(
            loader=loader,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Custom filters for safe Python string embedding
        def python_escape(value: str) -> str:
            """Escape for safe triple-quoted string literal."""
            return value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

        def python_repr(value: Any) -> str:
            """Generate Python-compatible literal via json.dumps.

            Handles None (→ None), bools (→ True/False), strings, numbers,
            and other JSON-serializable types.
            """
            if value is None:
                return "None"
            if isinstance(value, bool):
                return "True" if value else "False"
            return json.dumps(value, ensure_ascii=False)

        env.filters["python_escape"] = python_escape
        env.filters["python_repr"] = python_repr
        env.filters["pybool"] = lambda v: "True" if v else "False"
        return env

    # ── variable extraction ───────────────────────────────────────────

    def extract_variables(
        self, ahspec: dict[str, Any], *, skill_dir: Path | None = None
    ) -> dict[str, Any]:
        """Extract template variables from an AHSSPEC dict.

        Handles both raw YAML dicts and Pydantic model dumps.
        """
        identity = ahspec.get("identity", {})
        intent = ahspec.get("intent", {})
        interface = ahspec.get("interface", {})
        base = ahspec.get("base", {})
        instructions = ahspec.get("instructions", {})

        agent_name = identity.get("id", "unknown-agent")
        display_name = self._humanize_display_name(
            identity.get("display_name", "Unknown Agent"), agent_name
        )
        version = identity.get("version", "0.1.0")

        # Derive package_name: kebab-case → snake_case
        package_name = agent_name.replace("-", "_")

        # Derive agent_class: valid Python identifier from display_name
        agent_class = self._to_class_name(display_name)

        # Description from intent summary
        description = intent.get("summary", "")

        # Workflow: can be a list of step dicts or a string
        workflow = instructions.get("workflow", "")
        workflow_steps: list[dict[str, Any]] = []
        if isinstance(workflow, list):
            workflow_steps = workflow
            workflow = self._format_workflow(workflow)

        output_tpl = instructions.get("output_template", "")

        # Rules: list of strings
        rules = instructions.get("rules", [])

        # Requires: list of capability names (strings) or dicts
        requires = self._extract_requires(interface.get("requires", []))

        # Base runtime
        base_runtime = base.get("runtime", "python3.11") if base else "python3.11"

        # LLM provider/model: read from global config if available
        llm_provider, model, base_url = self._read_default_provider()

        # Tools: list of provide capability names (legacy) + full metadata
        tools = self._extract_tool_names(interface.get("provides", []))
        mcp_servers = interface.get("mcp_servers", [])
        api_templates = interface.get("api_templates", [])
        instructions = ahspec.get("instructions", {})
        resources = ahspec.get("resources", {})
        script_map = self._resolve_script_map(
            interface.get("provides", []),
            instructions=instructions,
            resources=resources,
        )
        tool_metadata = self._extract_tool_metadata(
            interface.get("provides", []),
            mcp_servers=mcp_servers,
            script_map=script_map,
            api_templates=api_templates,
        )

        # v0.7: Brick manifest from skill classification
        brick_manifest = self._build_brick_manifest(ahspec, skill_dir=skill_dir)

        return {
            "agent_name": agent_name,
            "agent_class": agent_class,
            "display_name": display_name,
            "version": version,
            "package_name": package_name,
            "description": description,
            "workflow": workflow,
            "workflow_steps": workflow_steps,  # v0.7.6: structured for CompiledWorkflow
            "output_tpl": output_tpl,
            "rules": rules,
            "base_runtime": base_runtime,
            "llm_provider": llm_provider,
            "model": model,
            "base_url": base_url,
            "tools": tools,
            "tool_metadata": tool_metadata,
            "mcp_servers": mcp_servers,
            "api_templates": api_templates,
            "script_map": script_map,
            "requires": requires,
            "brick_manifest": brick_manifest,
            "ai_tool_impls": {},  # populated by AI generation step
            "ai_references": {},  # populated by AI reference extraction
        }

    @staticmethod
    def _humanize_display_name(display_name: str, agent_id: str) -> str:
        """Convert kebab-case or snake_case display_name to human-readable form.

        "agent-browser" → "Agent Browser"
        "pdf_tool" → "PDF Tool"
        Preserves already-human names like "Weather Reporter".
        """
        # If the display_name is identical to the kebab-case ID, humanize it
        if display_name == agent_id:
            parts = re.split(r"[-_]", display_name)
            return " ".join(p.capitalize() for p in parts if p)

        # If it already has spaces or mixed case, it's likely fine
        if " " in display_name or any(c.isupper() for c in display_name[1:]):
            return display_name

        # Looks like a machine name: kebab/snake_case with no spaces
        if "-" in display_name or "_" in display_name:
            parts = re.split(r"[-_]", display_name)
            return " ".join(p.capitalize() for p in parts if p)

        return display_name

    @staticmethod
    def _to_class_name(display_name: str) -> str:
        """Convert a display name to a valid Python class name.

        "Discover Search" → "DiscoverSearch"
        "HTTP Client Tool" → "HTTPClientTool"
        "3D Printer" → "ThreeDPrinter"
        """
        # Split on whitespace/hyphens/underscores, strip non-alphanumeric
        parts = re.split(r"[\s\-_]+", display_name.strip())
        clean: list[str] = []
        for p in parts:
            p = re.sub(r"[^a-zA-Z0-9]", "", p)
            if p:
                # Uppercase first alpha char, preserve rest; strip leading digits
                clean.append(p[0].upper() + p[1:])

        result = "".join(clean)
        if not result:
            return "UnknownAgent"

        # Python class name must not start with a digit
        if result[0].isdigit():
            num_words = {
                "0": "Zero", "1": "One", "2": "Two", "3": "Three",
                "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                "8": "Eight", "9": "Nine",
            }
            prefix = num_words.get(result[0], "Num")
            result = prefix + result[1:]

        return result

    @staticmethod
    def _read_default_provider() -> tuple[str, str, str]:
        """Read default provider, model, and base_url from global config.

        Returns ("openai", "gpt-4o", "https://api.openai.com/v1") if no config found.
        """
        import tomllib as _tomllib

        config_path = Path.home() / ".agenthatch" / "config.toml"
        if not config_path.exists():
            return ("openai", "gpt-4o", "https://api.openai.com/v1")

        try:
            cfg = _tomllib.loads(config_path.read_text())
        except Exception:
            return ("openai", "gpt-4o", "https://api.openai.com/v1")

        provider = cfg.get("providers", {}).get("default", "openai")
        provider_cfg = cfg.get("providers", {}).get(provider, {})
        model = provider_cfg.get("default_model", "gpt-4o")
        base_url = provider_cfg.get("base_url", "https://api.openai.com/v1")
        return (provider, model, base_url)

    @staticmethod
    def _build_brick_manifest(
        ahspec: dict[str, Any], *, skill_dir: Path | None = None
    ) -> dict[str, Any] | None:
        """v0.7.15: Build BrickManifest dict from skill classification.

        Returns None if classification fails (backward-compatible fallback).

        v0.7.15 fixes:
          - Accepts skill_dir to check physical scripts/ directory, upgrading
            PROMPT_ONLY → TOOL_WRAPPER when scripts exist on disk.
          - Respects YAML base.sandbox for MCP_CONNECTOR (was: forced NONE).
        """
        try:
            from agenthatch_core.bricks.archetypes import (
                ClassificationResult,
                SkillArchetype,
                classify_skill,
            )
            from agenthatch_core.bricks.manifest import LoopKind
        except ImportError:
            return None

        try:
            result = classify_skill(ahspec)
        except Exception:
            return None

        archetype = result.archetype

        # v0.7.15: Upgrade PROMPT_ONLY if scripts/ directory exists on disk
        if archetype == SkillArchetype.PROMPT_ONLY and skill_dir is not None:
            scripts_path = skill_dir / "skills" / "scripts"
            if scripts_path.is_dir():
                script_files = [f for f in scripts_path.iterdir() if f.is_file()]
                if script_files:
                    archetype = SkillArchetype.TOOL_WRAPPER
                    result = ClassificationResult(
                        archetype=SkillArchetype.TOOL_WRAPPER,
                        confidence=0.70,
                        reasons=[
                            f"Found {len(script_files)} script(s) in skills/scripts/"
                        ],
                    )

        # Map archetype → loop engine
        if archetype == SkillArchetype.PROMPT_ONLY:
            loop_engine = LoopKind.DIRECT
        else:
            loop_engine = LoopKind.CONVERSATION

        # Map archetype → capbus
        capbus = archetype != SkillArchetype.PROMPT_ONLY

        # Map archetype → hooks (only for multi-step)
        hooks = archetype not in (
            SkillArchetype.PROMPT_ONLY, SkillArchetype.EXTERNAL_TOOL
        )

        # CredentialVault only if api_templates are present
        api_templates = ahspec.get("interface", {}).get("api_templates", [])
        credential_vault = bool(api_templates)

        # FileProcessor only for tool-wrapper and multi-step
        file_processor = archetype in (
            SkillArchetype.TOOL_WRAPPER, SkillArchetype.MULTI_STEP
        )

        # Guard active only if ANCHOR_RULES exist and not prompt-only
        rules = ahspec.get("instructions", {}).get("rules", [])
        guard_active = bool(rules) and archetype != SkillArchetype.PROMPT_ONLY

        return {
            "loop_engine": loop_engine.value,
            "capbus": capbus,
            "hooks": hooks,
            "guard_active": guard_active,
            "credential_vault": credential_vault,
            "file_processor": file_processor,
            "memory": True,  # v0.7.6: default-on, opt-out via memory: false
            "archetype": archetype.value,
            "archetype_confidence": result.confidence,
        }

    @staticmethod
    def _format_workflow(workflow: list[dict[str, Any]]) -> str:
        """Format a list of workflow step dicts into a string."""
        lines: list[str] = []
        for step in workflow:
            if isinstance(step, dict):
                num = step.get("step", "")
                desc = step.get("description", "")
                line = f"{num}. {desc}" if num else desc
                if step.get("script"):
                    line += f" (Use tool: {step['script']})"
                lines.append(line)
            else:
                lines.append(str(step))
        return "\n".join(lines)

    @staticmethod
    def _extract_requires(requires: list[dict[str, Any]]) -> list[str]:
        """Extract requirement names from interface.requires."""
        result: list[str] = []
        for req in requires:
            if isinstance(req, dict):
                name = req.get("capability", req.get("name", ""))
                if name:
                    result.append(name)
            elif isinstance(req, str):
                result.append(req)
        return result

    @staticmethod
    def _extract_tool_names(provides: list[dict[str, Any]]) -> list[str]:
        """Extract tool names from interface.provides."""
        result: list[str] = []
        for cap in provides:
            if isinstance(cap, dict):
                name = cap.get("capability", cap.get("name", ""))
                if name:
                    result.append(name)
            elif isinstance(cap, str):
                result.append(cap)
        return result

    @staticmethod
    def _resolve_script_map(
        provides: list[dict[str, Any]],
        instructions: dict[str, Any],
        resources: dict[str, Any],
    ) -> dict[str, str]:
        """Map capability names to script filenames.

        Uses the same matching logic as agent.py's _build_cap_to_script:
        1. Workflow steps that mention a capability + have a script
        2. Resources.scripts entries with fuzzy name match
        3. Direct filename match from scripts directory (runtime only, skipped here)
        """
        cap_to_script: dict[str, str] = {}
        cap_names: set[str] = set()
        for c in provides:
            if isinstance(c, dict):
                name = str(c.get("capability", c.get("name", "")))
                if name:
                    cap_names.add(name)

        # Approach 1: workflow steps
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

        # Approach 2: resources.scripts
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

        # Strip "skills/scripts/" prefix from script paths.
        # The generated tools.py uses SKILLS_SCRIPTS_DIR (already skills/scripts/)
        # so we need just the filename, not the full resource path.
        for cap_name, script_path in list(cap_to_script.items()):
            if script_path.startswith("skills/scripts/"):
                cap_to_script[cap_name] = script_path[len("skills/scripts/"):]

        return cap_to_script

    @staticmethod
    def _extract_tool_metadata(
        provides: list[dict[str, Any]],
        mcp_servers: list[dict[str, Any]] | None = None,
        script_map: dict[str, str] | None = None,
        api_templates: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract full tool metadata from interface.provides.

        Returns list of dicts with:
          - name: capability name
          - func_name: Python-safe function name (snake_case)
          - description: from capability description
          - input_schema: dict of param_name → type string
          - params: list of (name, type, default) tuples for signature
          - is_mcp: whether this tool is backed by an MCP server
          - mcp_server: MCP server name if applicable
          - has_inputs: whether the tool accepts parameters
          - script_name: mapped script filename (or "")
          - has_backend: whether tool has any runtime backend
          - backend_kind: "mcp" | "script" | "api_template" | "none"
        """
        result: list[dict[str, Any]] = []
        # Build set of MCP TOOL names (not server names)
        mcp_tool_names: set[str] = set()
        mcp_tool_to_server: dict[str, str] = {}
        for s in (mcp_servers or []):
            server_name = s.get("name", "")
            for t in s.get("tools", []):
                if isinstance(t, dict):
                    tn = t.get("name", "")
                    if tn:
                        mcp_tool_names.add(tn)
                        mcp_tool_to_server[tn] = server_name
        script_map = script_map or {}
        api_map: dict[str, dict[str, Any]] = {}
        for tmpl in (api_templates or []):
            if isinstance(tmpl, dict) and tmpl.get("name"):
                api_map[tmpl["name"]] = tmpl

        for cap in provides:
            if not isinstance(cap, dict):
                continue
            name = cap.get("capability", cap.get("name", ""))
            if not name:
                continue

            desc = cap.get("description", "")
            input_schema = cap.get("input_schema", {})

            # Normalize input_schema
            if isinstance(input_schema, dict):
                params: list[tuple[str, str, str]] = []
                for param_name, param_type in input_schema.items():
                    if param_name in ("type", "properties", "required"):
                        continue
                    if isinstance(param_type, str):
                        py_type = _json_type_to_python(param_type)
                        params.append((param_name, py_type, "None"))
                    elif isinstance(param_type, dict) and "type" in param_type:
                        py_type = _json_type_to_python(param_type["type"])
                        default = param_type.get("default", "None")
                        params.append((param_name, py_type, str(default)))
                has_inputs = len(params) > 0
            else:
                params = []
                has_inputs = False

            # Determine if MCP-backed
            is_mcp = name in mcp_tool_names or cap.get("type") == "mcp"
            mcp_server = mcp_tool_to_server.get(name, "")

            # Determine backend kind
            script_name = script_map.get(name, "")
            api_tmpl = api_map.get(name)
            if is_mcp and mcp_server:
                backend_kind = "mcp"
            elif script_name:
                backend_kind = "script"
            elif api_tmpl:
                backend_kind = "api_template"
            else:
                backend_kind = "none"

            result.append({
                "name": name,
                "func_name": name.replace("-", "_"),
                "description": desc or f"Handle the '{name}' capability.",
                "input_schema": input_schema,
                "params": params,
                "is_mcp": is_mcp,
                "mcp_server": mcp_server,
                "has_inputs": has_inputs,
                "script_name": script_name,
                "has_backend": backend_kind != "none",
                "backend_kind": backend_kind,
            })

        return result

    # ── AI-driven tool implementation generation ──────────────────────

    @staticmethod
    def _ai_generate_tool_impls(
        ahspec: dict[str, Any],
        skill_dir: Path,
        tool_metadata: list[dict[str, Any]],
        chat_fn: Any,
    ) -> dict[str, str]:
        """Generate real Python tool implementations using AI.

        Reads the FULL skill directory context (not just SKILL.md):
          - SKILL.md — main skill description + code examples
          - All reference files — detailed specifications
          - All script files — existing working code as reference
          - agenthatch.yaml — interface definitions

        The AI cross-references these files to produce meaningful
        implementations for each tool.

        Returns dict mapping func_name → implementation body (Python code).
        """
        # ── Step 1: Collect full skill context ──────────────────────
        context_files: list[dict[str, str]] = []

        # SKILL.md is always first
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            context_files.append({
                "path": "SKILL.md",
                "content": skill_md.read_text(encoding="utf-8"),
            })

        # All reference files
        refs_dir = skill_dir / "skills" / "references"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.glob("*")):
                if ref_file.is_file() and ref_file.suffix in (".md", ".txt"):
                    try:
                        content = ref_file.read_text(encoding="utf-8")
                        if len(content) > 0:
                            context_files.append({
                                "path": f"skills/references/{ref_file.name}",
                                "content": content,
                            })
                    except Exception:
                        pass

        # All script files (as reference for implementation patterns)
        scripts_dir = skill_dir / "skills" / "scripts"
        if scripts_dir.is_dir():
            for script_file in sorted(scripts_dir.glob("*")):
                if script_file.is_file():
                    try:
                        content = script_file.read_text(encoding="utf-8")
                        if len(content) > 0:
                            context_files.append({
                                "path": f"skills/scripts/{script_file.name}",
                                "content": content,
                            })
                    except Exception:
                        pass

        # Build the file context block
        file_context = ""
        for f in context_files:
            file_context += f"\n--- {f['path']} ---\n{f['content']}\n"

        if not file_context:
            logger.warning("No skill files found for AI tool generation")
            return {}

        # ── Step 2: Build tool metadata block ───────────────────────
        tools_desc = ""
        for t in tool_metadata:
            params_str = ", ".join(
                f"{n}: {ty}" for n, ty, _ in t.get("params", [])
            )
            tools_desc += (
                f"\nTool: {t['name']} (func_name: {t['func_name']})\n"
                f"  Description: {t['description']}\n"
                f"  Backend: {t['backend_kind']}\n"
                f"  Params: {params_str or 'none'}\n"
            )
            if t.get("script_name"):
                tools_desc += f"  Script: {t['script_name']}\n"
            if t.get("mcp_server"):
                tools_desc += f"  MCP Server: {t['mcp_server']}\n"

        # ── Step 3: System prompt ───────────────────────────────────
        system_prompt = (
            "You are an expert Python code generator for agent tool implementations. "
            "You will receive:\n"
            "1. A full skill directory context (SKILL.md, reference files, scripts)\n"
            "2. A list of tool definitions with their metadata\n\n"
            "Generate a complete Python function body for EACH tool. "
            "Follow these rules:\n"
            "- For script-backed tools: use the SKILLS_SCRIPTS_DIR constant (already defined) "
            "to locate scripts, e.g. SKILLS_SCRIPTS_DIR / 'script_name'\n"
            "- For MCP-backed tools: return a placeholder (MCP handles execution)\n"
            "- For API template tools: generate HTTP requests based on the skill context\n"
            "- For tools without backend: read the SKILL.md code examples and generate a real implementation\n"
            "- Do NOT use **kwargs — use the exact parameter names from the tool definition\n"
            "- Include proper error handling and return meaningful results\n"
            "- Import only from stdlib or packages mentioned in the skill context\n"
            "- Use SKILLS_SCRIPTS_DIR (a pathlib.Path) for all script paths, never hardcode paths\n\n"
            "Output format: Return a JSON object mapping func_name → implementation body.\n"
            "The function body should be the code INSIDE the function (after the signature and docstring).\n"
            'Example: {"fetch_url": "import subprocess\\n    ..."}'
        )

        # ── Step 4: User prompt ─────────────────────────────────────
        archetype = ahspec.get("base", {}).get("archetype", "generic")
        identity = ahspec.get("identity", {})
        agent_name = identity.get("display_name", "Unknown")

        user_prompt = (
            f"Generate Python implementations for the {agent_name} agent "
            f"(archetype: {archetype}).\n\n"
            f"=== SKILL FILES ===\n{file_context}\n\n"
            f"=== TOOL DEFINITIONS ===\n{tools_desc}\n\n"
            "Return a JSON object with TWO keys:\n"
            '1. "tools": object mapping func_name → implementation body code\n'
            '2. "references": object mapping dataclass/constant name → Python code\n\n'
            "For references: extract structured data (enums, constants, "
            "field definitions, configuration values) from the skill context "
            "files. If the reference files contain form field types, API "
            "endpoints, status codes, or other structured data, extract them "
            "as Python constants or dataclasses.\n\n"
            "Each implementation body should be the code INSIDE the function "
            "(after the signature and docstring). Use the skill context to "
            "understand what each tool should do."
        )

        # ── Step 5: Call LLM ────────────────────────────────────────
        try:
            response = chat_fn(system_prompt, user_prompt)
        except Exception as e:
            logger.error("AI tool generation LLM call failed: %s", e)
            return {}

        # ── Step 6: Parse response ──────────────────────────────────
        try:
            # Extract JSON from the response (may be wrapped in ```json blocks)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0]

            impls = json.loads(json_text.strip())
            if not isinstance(impls, dict):
                logger.warning("AI returned non-dict response: %s", type(impls))
                return {}

            # Validate all keys are valid func_names
            valid_tools = {}
            tool_names = {t["func_name"] for t in tool_metadata}
            tools_data = impls.get("tools", impls)  # backward compat
            if isinstance(tools_data, dict):
                for func_name, body in tools_data.items():
                    if func_name in tool_names and isinstance(body, str) and len(body) > 10:
                        # Normalize indentation: ensure 4-space indent for template insertion
                        body_lines = body.strip().split("\n")
                        indented = "\n".join(
                            " " * 4 + line if line.strip() else ""
                            for line in body_lines
                        )
                        valid_tools[func_name] = indented

            # Extract reference structures
            references = {}
            refs_data = impls.get("references", {})
            if isinstance(refs_data, dict):
                for ref_name, ref_code in refs_data.items():
                    if isinstance(ref_code, str) and len(ref_code) > 10:
                        references[ref_name] = ref_code

            return {"tools": valid_tools, "references": references}
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning("Failed to parse AI tool generation response: %s", e)
            return {}

    # ── generation ────────────────────────────────────────────────────

    def generate(
        self,
        ahspec: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        copy_skills: bool = True,
        skill_dir: Path | None = None,
        ai_chat_fn: Any | None = None,
    ) -> list[Path]:
        """Generate a complete Agent directory from an AHSSPEC dict.

        Args:
            ahspec: AHSSPEC dict (from agenthatch.yaml).
            output_dir: Target directory for the generated Agent.
            dry_run: If True, print files without writing.
            force: If True, overwrite existing output directory.
            copy_skills: If True, copy SKILL.md and resources.
            skill_dir: Source skill directory (for copying resources).
            ai_chat_fn: Optional callback for AI-driven tool generation.
                Signature: (system_prompt: str, user_prompt: str) -> str

        Returns:
            List of Paths that were (or would be) written.
        """
        variables = self.extract_variables(ahspec, skill_dir=skill_dir)

        # v0.9: AI-driven tool implementation generation
        # Reads the full skill directory context and generates real Python
        # implementations for each tool (not just stubs).
        if ai_chat_fn and skill_dir and variables.get("tool_metadata"):
            try:
                ai_result = self._ai_generate_tool_impls(
                    ahspec=ahspec,
                    skill_dir=skill_dir,
                    tool_metadata=variables["tool_metadata"],
                    chat_fn=ai_chat_fn,
                )
                if ai_result:
                    ai_tools = ai_result.get("tools", {})
                    ai_refs = ai_result.get("references", {})
                    if ai_tools:
                        variables["ai_tool_impls"] = ai_tools
                        logger.info(
                            "AI generated %d tool implementations", len(ai_tools)
                        )
                    if ai_refs:
                        variables["ai_references"] = ai_refs
                        logger.info(
                            "AI extracted %d reference structures", len(ai_refs)
                        )
            except Exception as e:
                logger.warning("AI tool generation failed, using template defaults: %s", e)

        written: list[Path] = []

        if dry_run:
            logger.info("Dry-run mode — no files will be written.")
        else:
            self._prepare_output_dir(output_dir, force)

        for template_name, output_rel in TEMPLATE_MAP.items():
            output_path_str = output_rel.format(
                package_name=variables["package_name"]
            )
            output_path = output_dir / output_path_str

            try:
                template = self._env.get_template(template_name)
                rendered = template.render(**variables)
            except jinja2.TemplateNotFound:
                logger.warning("Template not found: %s — skipping", template_name)
                continue
            except Exception as e:
                logger.error("Failed to render %s: %s", template_name, e)
                raise

            if dry_run:
                logger.info("Would write: %s (%d chars)", output_path, len(rendered))
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered, encoding="utf-8")
                logger.info("Written: %s", output_path)

            written.append(output_path)

        # Copy agenthatch.yaml to output root
        if not dry_run:
            self._write_ahspec_copy(ahspec, output_dir, variables)

        # Copy skills resources
        if copy_skills and skill_dir and not dry_run:
            self._copy_skills(skill_dir, output_dir, variables["package_name"])

        # Create __init__.py in package
        if not dry_run:
            pkg_init = output_dir / "src" / variables["package_name"] / "__init__.py"
            pkg_init.parent.mkdir(parents=True, exist_ok=True)
            if not pkg_init.exists():
                pkg_init.write_text(
                    f"# {variables['agent_class']} — generated by agenthatch\n",
                    encoding="utf-8",
                )
            written.append(pkg_init)

        # v0.7.15: Validate generated Python files compile correctly
        if not dry_run:
            validation_errors = self._validate_generated_python(output_dir)
            if validation_errors:
                for err in validation_errors:
                    logger.error("Validation error: %s", err)
                raise RuntimeError(
                    f"Generated agent contains {len(validation_errors)} validation "
                    f"error(s).  This is a template bug — the agent may crash at "
                    f"runtime.  Aborting generation.\n"
                    + "\n".join(f"  • {e}" for e in validation_errors)
                )

        return written

    # ── Generation validation ──────────────────────────────────────────

    @staticmethod
    def _validate_generated_python(output_dir: Path) -> list[str]:
        """Validate all generated Python files compile and contain no JS artifacts.

        v0.7.15: Catches template bugs like ``null`` instead of ``None``
        and truncated output before the user discovers them at runtime.

        Returns a list of error messages (empty list = all clear).
        """
        errors: list[str] = []

        for py_file in output_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")

            # 1. Check for JavaScript/JSON artifacts
            for js_kw in ("null", "undefined", "true", "false"):
                # Use word-boundary-ish check: keyword not inside a string or comment
                if re.search(rf"\b{js_kw}\b", content) and f'"{js_kw}"' not in content:
                    # Heuristic: if it appears as a bare keyword (not in quotes)
                    # Check each line independently
                    for lineno, line in enumerate(content.splitlines(), 1):
                        stripped = line.strip()
                        if (
                            stripped == js_kw
                            or stripped.endswith(f"={js_kw}")
                            or stripped.endswith(f"= {js_kw}")
                        ):
                            if js_kw == "true":
                                errors.append(
                                    f"{py_file.relative_to(output_dir)}:{lineno}: "
                                    f"'{js_kw}' found (use 'True' in Python)"
                                )
                            elif js_kw == "false":
                                errors.append(
                                    f"{py_file.relative_to(output_dir)}:{lineno}: "
                                    f"'{js_kw}' found (use 'False' in Python)"
                                )
                            elif js_kw == "null":
                                errors.append(
                                    f"{py_file.relative_to(output_dir)}:{lineno}: "
                                    f"'{js_kw}' found (use 'None' in Python)"
                                )
                            elif js_kw == "undefined":
                                errors.append(
                                    f"{py_file.relative_to(output_dir)}:{lineno}: "
                                    f"'{js_kw}' found (not a Python keyword)"
                                )

            # 2. Check Python syntax compiles
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(
                    f"{py_file.relative_to(output_dir)}:{e.lineno}: "
                    f"SyntaxError: {e.msg}"
                )

        return errors

    def _prepare_output_dir(self, output_dir: Path, force: bool) -> None:
        """Prepare the output directory."""
        if output_dir.exists():
            if force:
                logger.info("Removing existing output directory: %s", output_dir)
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    f"Use --force to overwrite."
                )
        output_dir.mkdir(parents=True, exist_ok=True)

    def _write_ahspec_copy(
        self, ahspec: dict[str, Any], output_dir: Path, variables: dict[str, Any]
    ) -> None:
        """Write a copy of agenthatch.yaml to the output root."""
        import yaml

        # Update agent status
        ahspec_copy = dict(ahspec)
        if "agent" not in ahspec_copy:
            ahspec_copy["agent"] = {}
        agent_cfg = ahspec_copy["agent"]
        if isinstance(agent_cfg, dict):
            agent_cfg["status"] = "hatched"
            agent_cfg["generated_at"] = datetime.now(UTC).isoformat()
        else:
            ahspec_copy["agent"] = {
                "status": "hatched",
                "generated_at": datetime.now(UTC).isoformat(),
            }

        yaml_path = output_dir / "agenthatch.yaml"
        yaml_str = yaml.dump(
            json.loads(json.dumps(ahspec_copy, default=str)),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        yaml_path.write_text(yaml_str, encoding="utf-8")

    def _copy_skills(self, skill_dir: Path, output_dir: Path, package_name: str) -> None:
        """Copy the entire skill directory as a fallback bundle.

        The agent carries a complete copy of its source skill so that
        it can self-reference during runtime — reading its own SKILL.md
        for guidance, executing scripts, and self-healing when necessary.

        The source skill_dir typically contains:
          - SKILL.md
          - skills/scripts/... (executable scripts)
          - skills/references/... (reference docs)

        We copy to two locations:
        1. output_dir/ (top-level, for human reference)
        2. output_dir/src/<package_name>/skills/ (for tools.py subprocess access)

        Excludes VCS and build artifacts via ignore patterns.
        """
        import fnmatch

        def ignore(src: str, names: list[str]) -> list[str]:
            patterns = (
                ".git", "__pycache__", "*.pyc", ".DS_Store",
                "node_modules", ".venv", "venv", ".env",
            )
            ignored = []
            for name in names:
                for pat in patterns:
                    if fnmatch.fnmatch(name, pat):
                        ignored.append(name)
                        break
            return ignored

        # Destination 1: top-level copy (full structure, for human reference)
        dest_top = output_dir / "skills"
        if dest_top.exists():
            shutil.rmtree(dest_top)
        dest_top.mkdir(parents=True, exist_ok=True)
        for item in skill_dir.iterdir():
            if ignore(str(skill_dir), [item.name]):
                continue
            dest = dest_top / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True, ignore=ignore)
            else:
                shutil.copy2(item, dest)

        # Destination 2: for tools.py — scripts/ at the right level.
        # tools.py has: SKILLS_SCRIPTS_DIR = Path(__file__).parent / "skills" / "scripts"
        # So we need: src/<pkg>/skills/scripts/
        dest_pkg = output_dir / "src" / package_name / "skills"
        skill_subdir = skill_dir / "skills"
        if skill_subdir.is_dir():
            if dest_pkg.exists():
                shutil.rmtree(dest_pkg)
            shutil.copytree(skill_subdir, dest_pkg, dirs_exist_ok=True, ignore=ignore)
        else:
            # No skills/ subdirectory — copy whole skill_dir
            if dest_pkg.exists():
                shutil.rmtree(dest_pkg)
            shutil.copytree(skill_dir, dest_pkg, dirs_exist_ok=True, ignore=ignore)


def generate_agent(
    ahspec: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    copy_skills: bool = True,
    skill_dir: Path | None = None,
    template_dir: Path | None = None,
) -> list[Path]:
    """Convenience function: generate an Agent directory from AHSSPEC.

    Args:
        ahspec: AHSSPEC dict.
        output_dir: Target directory path.
        dry_run: Print without writing.
        force: Overwrite existing directory.
        copy_skills: Copy SKILL.md and resources.
        skill_dir: Source skill directory.
        template_dir: Custom template directory.

    Returns:
        List of written file paths.
    """
    engine = GenerateEngine(template_dir=template_dir)
    return engine.generate(
        ahspec,
        output_dir,
        dry_run=dry_run,
        force=force,
        copy_skills=copy_skills,
        skill_dir=skill_dir,
    )
