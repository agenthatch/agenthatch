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

# v1.0.0: Templates that only render when knowledge_base is enabled.
# These produce the runtime KnowledgeBaseBrick shim + retrieve tool.
KB_TEMPLATE_MAP: dict[str, str] = {
    "knowledge_base.py.j2": "src/{package_name}/knowledge_base.py",
}


def _json_type_to_python(json_type: str) -> str:
    """Map JSON Schema type to Python type annotation.

    Per JSON Schema spec: ``number`` is any numeric value (including
    floats) and ``integer`` is a subset of number. Mapping ``number``
    to ``int`` would lose float precision in generated tool signatures.
    """
    mapping = {
        "string": "str",
        "number": "float",
        "float": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return mapping.get(json_type, "Any")


def _matches_pattern(path: str, pattern: str) -> bool:
    """Match ``path`` against a glob-style ``pattern`` (v1.0.1 R2-H2).

    Supports ``draft/*``, ``*.tmp``, ``**/secret/**`` style patterns.
    Uses :func:`fnmatch.fnmatch` on both the full relative path and the
    basename so users can write either ``draft/*`` (path match) or
    ``*.tmp`` (basename match).
    """
    import fnmatch
    from os.path import basename
    return (
        fnmatch.fnmatch(path, pattern)
        or fnmatch.fnmatch(basename(path), pattern)
    )


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
            """Escape for safe triple-quoted string literal.

            v1.0.1 (R2-H4 regression): Previously only escaped backslashes
            and triple-quotes.  But LLM-generated text may contain:
              - **null bytes** (``\\x00``): Python source files cannot
                contain null bytes — ``compile()`` raises ``SyntaxError:
                source code string cannot contain null bytes``.
              - **other control chars** (``\\x01``–``\\x1f`` except
                ``\\t``, ``\\n``, ``\\r``): not fatal but produce
                invisible garbage in docstrings / system prompts.
            Now we escape all control chars to their ``\\xNN`` form
            (after backslash and triple-quote escaping) so the rendered
            source always compiles cleanly.
            """
            # Step 1: Escape backslashes first so we don't double-escape
            # the backslashes we're about to add.
            v = value.replace("\\", "\\\\")
            # Step 2: Escape triple-quotes (would terminate the
            # triple-quoted docstring / constant literal).  We use the
            # char-by-char form to avoid embedding a literal triple-quote
            # in this very docstring (which would terminate it).
            v = v.replace('"""', "\\" + "\\" + "\\" + '"' + '"' + '"')
            # Step 3: Escape null bytes and control chars (except
            # common whitespace \t, \n, \r which are valid in source).
            # Use a regex sub for compactness.
            v = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
                lambda m: f"\\x{ord(m.group()):02x}",
                v,
            )
            return v

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
        version = identity.get("version", "")

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
            dependencies=base.get("dependencies", []) if base else [],
        )

        # v0.7: Brick manifest from skill classification
        brick_manifest = self._build_brick_manifest(ahspec, skill_dir=skill_dir)

        # v1.0.0: Knowledge base variables (only present when user passed <kb>)
        kb_config = ahspec.get("knowledge_base")
        kb_vars = self._extract_kb_variables(kb_config) if kb_config else None

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
            # v0.9.8: loop_workflow extracted as top-level template variable
            "loop_workflow": brick_manifest.get("loop_workflow") if brick_manifest else None,
            "ai_tool_impls": {},  # populated by AI generation step
            "ai_references": {},  # populated by AI reference extraction
            # v0.8.19: Pass dependencies for CLI tool fallback in template
            "dependencies": base.get("dependencies", []) if base else [],
            # v1.0.0: Knowledge base — None when no KB declared
            "kb": kb_vars,
            "kb_enabled": kb_vars is not None,
        }

    @staticmethod
    def _extract_kb_variables(kb_config: dict[str, Any]) -> dict[str, Any]:
        """Flatten KnowledgeBaseConfig into template-friendly variables.

        Returns a dict with keys prefixed ``kb_`` for direct template use.
        """
        usage = kb_config.get("usage_strategy", {}) or {}
        prompt = kb_config.get("prompt_artifact", {}) or {}
        sources = kb_config.get("sources", []) or []
        return {
            "sources": sources,
            "source_paths": [s.get("path", "") for s in sources if s.get("path")],
            "usage_strategy": usage,
            "when_to_retrieve": usage.get("when_to_retrieve", []),
            "query_templates": usage.get("query_templates", []),
            "integration_pattern": usage.get("integration_pattern", "tool_call_then_answer"),
            "max_results_per_query": usage.get("max_results_per_query", 5),
            "citation_required": usage.get("citation_required", True),
            "fallback_when_no_match": usage.get("fallback_when_no_match", "inform_user"),
            "system_prompt_section": prompt.get("system_prompt_section", ""),
            "retrieve_tool_description": prompt.get("retrieve_tool_description", ""),
            "integration_instructions": prompt.get("integration_instructions", ""),
            "chunk_size": kb_config.get("chunk_size", 800),
            "chunk_overlap": kb_config.get("chunk_overlap", 100),
            "embedding_model": kb_config.get("embedding_model", "all-MiniLM-L6-v2"),
            "retrieval_top_k": kb_config.get("retrieval_top_k", 5),
            "retrieval_alpha": kb_config.get("retrieval_alpha", 0.7),
            # v1.0.1 (R2-M1): Default False to match KnowledgeBaseConfig
            # and the runtime KnowledgeStore.  Round 1's C5 fix changed
            # the schema default but missed this extraction site —
            # older KB configs without the field would have rendered
            # ``ENABLE_LLM_RERANK = True`` despite no rerank_fn being
            # injected at runtime, misleading users.
            "enable_llm_rerank": kb_config.get("enable_llm_rerank", False),
            "total_documents": kb_config.get("total_documents", 0),
            "total_chunks": kb_config.get("total_chunks", 0),
            "index_size_bytes": kb_config.get("index_size_bytes", 0),
        }

    @staticmethod
    def _build_knowledge_index(
        *, output_dir: Path, kb_vars: dict[str, Any]
    ) -> dict[str, int]:
        """v1.0.0 Phase 3.5: Build the KB SQLite index into ``output_dir/knowledge/``.

        Walks each ``source_paths`` entry, chunks files via KBChunker
        (800-char chunks at paragraph boundaries), and writes a single
        ``kb_index.db`` that the runtime KnowledgeBaseBrick opens via
        ``KnowledgeStore.load()``.

        Returns ``{total_chunks, index_size_bytes}``.  On any failure,
        returns zeros — never blocks generation (the agent runtime will
        fall back to a no-KB experience).
        """
        try:
            from agenthatch_core.bricks.knowledge.chunker import KBChunker
            from agenthatch_core.bricks.knowledge.store import (
                KBDocument,
                KnowledgeStore,
            )
        except ImportError as e:
            logger.warning("KB index build skipped (import failed): %s", e)
            return {"total_chunks": 0, "index_size_bytes": 0}

        knowledge_dir = output_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = int(kb_vars.get("chunk_size", 800))
        chunk_overlap = int(kb_vars.get("chunk_overlap", 100))
        chunker = KBChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Collect chunks from all sources
        # v1.0.1 (R2-C1): Pass a path-unique source_label (relative path
        # from the user-provided source root) so two files with the same
        # basename in different subdirectories don't collide on doc_id
        # and silently overwrite each other via INSERT OR REPLACE.
        # v1.0.1 (R2-H2): Respect each source's ``include_patterns``
        # and ``exclude_patterns`` from frontmatter.  Previously these
        # were hardcoded to ``("*.md", "*.txt", "*.rst", "*.markdown")``
        # and frontmatter overrides were silently ignored — user-set
        # ``include_patterns: ["*.json"]`` or ``exclude_patterns: ["draft/*"]``
        # had no effect.
        _DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
            "*.md", "*.txt", "*.rst", "*.markdown",
        )
        all_chunks: list[Any] = []
        for source in kb_vars.get("sources", []):
            if not isinstance(source, dict):
                continue
            source_path_str = source.get("path", "")
            if not source_path_str:
                continue
            source_path = Path(source_path_str)
            if not source_path.exists():
                logger.warning("KB source not found: %s", source_path)
                continue

            include_patterns = source.get("include_patterns") or _DEFAULT_INCLUDE_PATTERNS
            exclude_patterns = source.get("exclude_patterns") or []

            if source_path.is_file():
                # v1.0.1 (R2-C1 regression): Single-file source — pass
                # an explicit, path-unique ``source_label`` (absolute
                # path) so two same-basename files in different
                # directories don't collide on doc_id and silently
                # overwrite each other via ``INSERT OR REPLACE``.
                # The full path is stored as ``source`` in metadata;
                # the chunker also keeps ``file_path`` (absolute) for
                # debugging.
                all_chunks.extend(
                    chunker.chunk_file(source_path, source_label=str(source_path))
                )
                continue

            # v1.0.1 (R2b-M10): Collect files into a set first to
            # deduplicate across patterns.  Previously a file named
            # ``notes.markdown`` matched both ``*.md`` (no — only
            # ``.md`` suffix) and ``*.markdown`` patterns... actually
            # fnmatch ``*.md`` won't match ``notes.markdown``, but a
            # file like ``README.md`` could match if user-supplied
            # patterns overlap (e.g. ``["*.md", "*.MD"]`` on a
            # case-insensitive FS, or ``["*", "*.md"]``).  Without
            # dedup the file would be chunked twice, producing duplicate
            # chunks (deduped at DB layer via INSERT OR REPLACE, but
            # wasting CPU and inflating the chunk count log).
            seen_files: set[Path] = set()
            matched_files: list[Path] = []
            for pattern in include_patterns:
                for f in source_path.rglob(pattern):
                    if f in seen_files:
                        continue
                    seen_files.add(f)
                    matched_files.append(f)
            for f in sorted(matched_files):
                # Apply exclude_patterns (matched against the path
                # relative to the source root).
                try:
                    rel = f.relative_to(source_path)
                    rel_str = str(rel)
                except ValueError:
                    rel_str = f.name
                if any(_matches_pattern(rel_str, p) for p in exclude_patterns):
                    continue
                all_chunks.extend(
                    chunker.chunk_file(f, source_label=rel_str)
                )

        if not all_chunks:
            logger.warning("KB index build: no chunks produced (empty sources?)")
            return {"total_chunks": 0, "index_size_bytes": 0}

        # Build the SQLite index
        # v1.0.1 (L6): Removed redundant `store._get_db(); store._init_schema()`
        # calls — `_get_db()` already initializes schema for the main
        # thread via `_init_schema_for_conn`, and `add_documents()` calls
        # `_get_db()` internally, so the explicit calls were no-ops.
        # v1.0.1 (C5): Default `enable_llm_rerank` to False to match
        # `KnowledgeBaseConfig` — rerank infra exists but no rerank_fn
        # is injected at runtime yet.
        store = KnowledgeStore(
            knowledge_dir,
            embedding_model=kb_vars.get("embedding_model", "all-MiniLM-L6-v2"),
            enable_llm_rerank=kb_vars.get("enable_llm_rerank", False),
        )
        try:
            # v1.0.1 (R3-H3): Clear stale data from previous builds so
            # removed sources don't linger.  Previously ``add_documents``
            # used ``INSERT OR REPLACE`` keyed on ``doc_id`` — if the
            # user removed a source file between builds, its chunks
            # (and FTS5 entries) stayed in the DB, causing the runtime
            # ``retrieve()`` to return stale content from a deleted file.
            #
            # The index FILE is preserved (per the project constraint
            # ``--force must not erase KB index during build``); only
            # table CONTENTS are cleared because we're rebuilding from
            # current sources.  The ``kb_ad`` AFTER DELETE trigger
            # cleans up the corresponding ``kb_fts`` rows automatically.
            db = store._get_db()
            db.execute("DELETE FROM kb_documents")
            db.commit()

            store.add_documents([
                KBDocument(
                    doc_id=c.doc_id,
                    content=c.content,
                    metadata=c.metadata,
                )
                for c in all_chunks
            ])
            store.build_index()
            stats = store.get_stats()
            logger.info(
                "KB index built: %d documents → %d chunks, %.1f KB index",
                len(all_chunks),
                stats["total_documents"],
                stats["index_size_bytes"] / 1024,
            )
            return {
                "total_chunks": int(stats["total_documents"]),
                "index_size_bytes": int(stats["index_size_bytes"]),
            }
        except Exception as e:
            logger.warning("KB index build failed: %s", e)
            return {"total_chunks": 0, "index_size_bytes": 0}
        finally:
            store.close()

    @staticmethod
    def _humanize_display_name(display_name: str, agent_id: str) -> str:
        """Convert kebab-case or snake_case display_name to human-readable form.

        "interactive-tool" → "Interactive Tool"
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

        provider = cfg.get("agenthatch", {}).get("default", "openai")
        # Resolve custom.xxx nested key
        if provider.startswith("custom."):
            custom_key = provider.removeprefix("custom.")
            provider_cfg = cfg.get("providers", {}).get("custom", {}).get(custom_key, {})
        else:
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

        # Map archetype → brick configuration (single source of truth)
        from agenthatch_core.bricks.archetypes import archetype_to_brick_config

        cfg = archetype_to_brick_config(
            archetype=archetype,
            api_templates=ahspec.get("interface", {}).get("api_templates", []),
            rules=ahspec.get("instructions", {}).get("rules", []),
        )

        # v0.9.8: Allow spec to override task_complete_enabled and loop_workflow.
        # The base section of agenthatch.yaml can set these for interactive agents.
        base = ahspec.get("base", {}) or {}
        if "task_complete_enabled" in base:
            cfg["task_complete_enabled"] = bool(base["task_complete_enabled"])
        if "loop_workflow" in base:
            cfg["loop_workflow"] = base["loop_workflow"]
        if "loop_engine" in base:
            cfg["loop_engine"] = base["loop_engine"]

        return {
            **cfg,
            "memory": True,
            "archetype": archetype.value,
            "archetype_confidence": result.confidence,
            "loop_engine": cfg["loop_engine"].value,  # engine.py uses string value
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

        # Approach 3: Fallback — any remaining unmatched capabilities
        # that share a workflow step with a script also get that script.
        # This handles multi-tool skills backed by a single script
        # (e.g. calc_add/calc_subtract/calc_multiply/calc_divide → calc.py).
        unmatched = cap_names - set(cap_to_script.keys())
        if unmatched and len(cap_to_script) > 0:
            # Use the most common script from already-matched caps
            script_counts: dict[str, int] = {}
            for _cap_name, script_name in cap_to_script.items():
                script_counts[script_name] = script_counts.get(script_name, 0) + 1
            main_script = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
            for cap_name in unmatched:
                cap_to_script[cap_name] = main_script

        # Strip "skills/scripts/" or "scripts/" prefix from script paths.
        # The generated tools.py uses SKILLS_SCRIPTS_DIR (already .../skills/scripts/)
        # so we need just the filename, not the full resource path.
        for cap_name, script_path in list(cap_to_script.items()):
            if script_path.startswith("skills/scripts/"):
                cap_to_script[cap_name] = script_path[len("skills/scripts/"):]
            elif script_path.startswith("scripts/"):
                cap_to_script[cap_name] = script_path[len("scripts/"):]

        return cap_to_script

    @staticmethod
    def _extract_tool_metadata(
        provides: list[dict[str, Any]],
        mcp_servers: list[dict[str, Any]] | None = None,
        script_map: dict[str, str] | None = None,
        api_templates: list[dict[str, Any]] | None = None,
        dependencies: list[str] | None = None,
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
                        default = param_type.get("default")
                        # repr() quotes strings so they render as valid Python
                        # literals in the generated function signature; other
                        # types (int, float, bool, None) already str() correctly.
                        default_str = repr(default) if isinstance(default, str) else str(default)
                        params.append((param_name, py_type, default_str))
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

            # v0.8.19: If no backend was detected but MCP servers are
            # configured, treat the tool as MCP-backed.  Harness F
            # detects MCP servers but the MCPServerEntry schema has no
            # "tools" field, so the tool→server mapping is empty.
            # Without this fallback all MCP tools become stubs.
            if backend_kind == "none" and mcp_servers:
                first_mcp = mcp_servers[0]
                if isinstance(first_mcp, dict) and first_mcp.get("name"):
                    backend_kind = "mcp"
                    mcp_server = first_mcp["name"]
                    is_mcp = True

            # v0.9: If no backend was detected but base.dependencies declares
            # CLI tools, treat the tool as a CLI wrapper.  This handles skills
            # where all capabilities are backed by one CLI binary.
            if backend_kind == "none" and dependencies:
                backend_kind = "cli_tool"

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
                # v0.8.13: repr() to safely embed script_name in Python strings
                # (handles internal quotes like type="all")
                "script_name_repr": repr(script_name),
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
    ) -> dict[str, Any]:
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

        # All reference files (check both skills/references/ and root-level)
        for refs_dir in (
            skill_dir / "skills" / "references",
            skill_dir,
        ):
            if refs_dir.is_dir():
                for ref_file in sorted(refs_dir.glob("*")):
                    if ref_file.is_file() and ref_file.suffix in (".md", ".txt"):
                        # Skip SKILL.md (already added) and agenthatch.yaml
                        if ref_file.name in ("SKILL.md", "agenthatch.yaml"):
                            continue
                        try:
                            content = ref_file.read_text(encoding="utf-8")
                            if len(content) > 0:
                                rel = ref_file.relative_to(skill_dir)
                                context_files.append({
                                    "path": str(rel),
                                    "content": content,
                                })
                        except Exception:
                            pass

        # All script files (check both skills/scripts/ and root-level scripts/)
        for scripts_dir in (
            skill_dir / "skills" / "scripts",
            skill_dir / "scripts",
        ):
            if scripts_dir.is_dir():
                for script_file in sorted(scripts_dir.glob("*")):
                    if script_file.is_file():
                        try:
                            content = script_file.read_text(encoding="utf-8")
                            if len(content) > 0:
                                rel = script_file.relative_to(skill_dir)
                                context_files.append({
                                    "path": str(rel),
                                    "content": content,
                                })
                        except Exception:
                            pass

        # Template files (both skills/templates/ and root-level templates/)
        for tmpl_dir in (
            skill_dir / "skills" / "templates",
            skill_dir / "templates",
        ):
            if tmpl_dir.is_dir():
                for tmpl_file in sorted(tmpl_dir.glob("*")):
                    if tmpl_file.is_file() and tmpl_file.suffix in (
                        ".js", ".html", ".css", ".py", ".json", ".yaml", ".txt",
                    ):
                        try:
                            content = tmpl_file.read_text(encoding="utf-8")
                            if len(content) > 0:
                                # Truncate large template files to 8KB
                                if len(content) > 8192:
                                    content = (
                                        content[:4096]
                                        + "\n... [truncated] ...\n"
                                        + content[-4096:]
                                    )
                                rel = tmpl_file.relative_to(skill_dir)
                                context_files.append({
                                    "path": str(rel),
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
            "Follow these rules:\n\n"
            "INDENTATION RULES (CRITICAL — incorrect indentation causes SyntaxErrors):\n"
            "- The code you generate will be inserted into a function body at indent 4.\n"
            "- Start your code at indent 0 (no leading spaces).\n"
            "- Each nested block (if/else/try/except/for/while/with) adds 4 spaces.\n"
            "- Example of correct structure:\n"
            "  if condition:\n"
            "      do_something()\n"
            "      try:\n"
            "          result = inner_call()\n"
            "          if result.ok:\n"
            "              return result.data\n"
            "          else:\n"
            "              return f'Error: {result.msg}'\n"
            "      except Exception as e:\n"
            "          return str(e)\n"
            "  else:\n"
            "      return 'condition not met'\n\n"
            "BACKEND RULES:\n"
            "- For script-backed tools: use SKILLS_SCRIPTS_DIR / 'script_name'\n"
            "- For MCP-backed tools: return a placeholder (MCP handles execution)\n"
            "- For API template tools: generate HTTP requests from skill context\n"
            "- For CLI-backed tools (backend=none, skill has CLI dependencies):\n"
            "  read CLI commands from SKILL.md and generate subprocess.run() calls.\n"
            "  Example: subprocess.run(['tool-name', 'action', arg],\n"
            "      capture_output=True, text=True, timeout=120)\n"
            "- For other tools without backend: generate real implementation from context\n"
            "- Do NOT use **kwargs — use exact parameter names from tool definition\n"
            "- Include proper error handling and return meaningful results\n"
            "- Import only stdlib or packages from the skill context\n\n"
            "Output format: JSON object mapping func_name → implementation body.\n"
            "The body is the code INSIDE the function (after signature and docstring).\n"
            'Example: {"fetch_url": "import subprocess\\n'
            'try:\\n'
            '    result = subprocess.run([url], capture_output=True, text=True)\\n'
            '    return result.stdout.strip()\\n'
            'except Exception as e:\\n'
            '    return str(e)"}'
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
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = chat_fn(system_prompt, user_prompt)
                if not response or not response.strip():
                    logger.warning(
                        "AI tool generation returned empty response "
                        "(attempt %d/%d)",
                        attempt + 1,
                        max_retries + 1,
                    )
                    if attempt < max_retries:
                        continue
                    return {}
                break
            except Exception as e:
                logger.error(
                    "AI tool generation LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                if attempt >= max_retries:
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
                        # Normalize indentation: strip AI's own indentation, then add
                        # consistent 4-space indent for template insertion.
                        # Use the AI's original indentation (which is usually
                        # mostly correct) and add base_indent to
                        # every line and validate the result.
                        body_lines = body.strip().split("\n")
                        indented_lines = [" " * 4 + line for line in body_lines]
                        indented = "\n".join(indented_lines)

                        # Validate: try to compile the generated code.
                        # Wrap in a dummy function since the code is at
                        # indent 4 (it will be inserted into a function body).
                        wrapper = (
                            "def _validate():\n"
                            + indented
                            + "\n"
                        )
                        valid = True
                        error_lines: list[int] = []
                        try:
                            compile(wrapper, f"<tool:{func_name}>", "exec")
                        except SyntaxError as se:
                            valid = False
                            if se.lineno:
                                # Adjust: wrapper adds 1 line (def _validate),
                                # so error line in indented code is lineno - 1
                                code_lineno = se.lineno - 1
                                error_lines.append(code_lineno)
                                # v0.8.19: Also collect subsequent lines
                                error_lines.append(code_lineno + 1)
                                error_lines.append(code_lineno + 2)
                            # Attempt normalization with error context
                            try:
                                fixed = GenerateEngine._normalize_indentation(
                                    indented_lines, error_lines
                                )
                                fixed_str = "\n".join(fixed)
                                fixed_wrapper = (
                                    "def _validate():\n"
                                    + fixed_str
                                    + "\n"
                                )
                                compile(fixed_wrapper, f"<tool:{func_name}>", "exec")
                                indented = fixed_str
                                valid = True
                            except SyntaxError:
                                pass

                        if valid:
                            valid_tools[func_name] = indented
                        else:
                            logger.warning(
                                "AI-generated code for tool '%s' has syntax errors, "
                                "skipping. Tool will use template fallback.",
                                func_name,
                            )

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
                    ai_tools: dict[str, str] = ai_result.get("tools", {})
                    ai_refs: dict[str, str] = ai_result.get("references", {})
                    if ai_tools:
                        variables["ai_tool_impls"] = ai_tools
                        logger.info(
                            "AI generated %d tool implementations", len(ai_tools)
                        )
                    else:
                        # v0.8.19: warn when AI generation produced no tools
                        tool_count = len(variables["tool_metadata"])
                        logger.warning(
                            "AI tool generation returned no implementations "
                            "for %d tools. Tools will be stubs. "
                            "Check LLM provider configuration.",
                            tool_count,
                        )
                    if ai_refs:
                        variables["ai_references"] = ai_refs
                        logger.info(
                            "AI extracted %d reference structures", len(ai_refs)
                        )
                else:
                    # v0.8.19: warn when AI generation returned empty
                    tool_count = len(variables["tool_metadata"])
                    logger.warning(
                        "AI tool generation failed for %d tools. "
                        "Tools will be stubs. "
                        "Check that the LLM provider supports the "
                        "Anthropic Messages API format if using a custom provider.",
                        tool_count,
                    )
            except Exception as e:
                logger.warning(
                    "AI tool generation failed, using template defaults: %s", e
                )

        written: list[Path] = []

        if dry_run:
            logger.info("Dry-run mode — no files will be written.")
        else:
            self._prepare_output_dir(output_dir, force)

        # v1.0.0 Phase 3.5: Build the KB vector index *after* output_dir is
        # prepared (so --force doesn't wipe it) but *before* template
        # rendering (so knowledge_base.py.j2 sees correct stats).
        if (
            not dry_run
            and variables.get("kb_enabled")
            and variables.get("kb", {}).get("source_paths")
        ):
            # v1.0.1 (R4-V2): Surface KB index build activity via a
            # logger.info call — the call site is library code (no
            # console access), so we use logger which the CLI's
            # RichHandler picks up.  Without this the user sees
            # "▸ Phase 3/3 Agent Generation" followed by a 5-90s silent
            # gap (sentence-transformers downloads ~80MB on first run).
            import logging as _logging
            _log = _logging.getLogger("agenthatch")
            _log.info(
                "Phase 3.5: building KB index (chunking + embedding + FTS5)..."
            )
            kb_stats = self._build_knowledge_index(
                output_dir=output_dir,
                kb_vars=variables["kb"],
            )
            _log.info(
                "Phase 3.5: KB index built — %d chunks, %d bytes",
                kb_stats["total_chunks"],
                kb_stats["index_size_bytes"],
            )
            # Mutate both the variables dict (for template rendering) and
            # the AHSSPEC dict (for agenthatch.yaml persistence).
            variables["kb"]["total_chunks"] = kb_stats["total_chunks"]
            variables["kb"]["index_size_bytes"] = kb_stats["index_size_bytes"]
            kb_cfg = ahspec.get("knowledge_base")
            if isinstance(kb_cfg, dict):
                kb_cfg["total_chunks"] = kb_stats["total_chunks"]
                kb_cfg["index_size_bytes"] = kb_stats["index_size_bytes"]

            # v1.0.1 (R3-M11): If KB build produced zero chunks (empty
            # sources, all-binary files, all-too-large files, etc.),
            # DISABLE KB at the template/manifest level instead of
            # generating a ``knowledge_base.py`` that points at an empty
            # index.  Previously the empty-DB ``knowledge_base.py`` was
            # still rendered and the manifest still advertised
            # ``knowledge_base: True`` — at runtime the agent would
            # import the module, register a ``retrieve`` tool that
            # always returned "no matching chunks", and inject a KB
            # system-prompt section that promised retrieval would work.
            #
            # Disabling here ensures:
            #   - ``KB_TEMPLATE_MAP`` rendering is skipped (no
            #     ``knowledge_base.py`` written).
            #   - ``brick_manifest.knowledge_base`` reflects reality (False)
            #     so ``agent.py.j2`` doesn't emit ``knowledge_base=True``
            #     to the runtime BrickManifest.
            #   - The runtime AHCoreAgent doesn't try to import a
            #     non-existent module (which would log a warning).
            if kb_stats["total_chunks"] == 0:
                logger.warning(
                    "KB build produced 0 chunks — disabling KB at "
                    "manifest/template level (no knowledge_base.py "
                    "will be generated). Check your source paths "
                    "and include/exclude patterns."
                )
                variables["kb_enabled"] = False
                # Clear AHSSPEC's knowledge_base config so the
                # generated ``agenthatch.yaml`` doesn't advertise KB.
                # ``variables["kb"]`` is left intact for debugging
                # (template rendering can still surface the build stats)
                # but ``kb_enabled=False`` gates all rendering branches.
                if isinstance(kb_cfg, dict):
                    kb_cfg["enabled"] = False
                # Flip ``brick_manifest.knowledge_base`` to False so
                # ``agent.py.j2`` skips emitting ``knowledge_base=True``
                # to the runtime BrickManifest.  Without this, the
                # generated agent would instantiate with KB enabled,
                # try to import ``knowledge_base.py`` (which we did NOT
                # render), and surface the ImportError as a startup
                # warning.
                bm = variables.get("brick_manifest")
                if isinstance(bm, dict):
                    bm["knowledge_base"] = False

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

        # v1.0.0: Render KB-specific templates (only when kb_enabled)
        if variables.get("kb_enabled"):
            for template_name, output_rel in KB_TEMPLATE_MAP.items():
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

        # Copy agenthatch.yaml to output root (picks up any KB stats mutations)
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
                # v0.8.11: Auto-fix syntax errors instead of aborting
                syntax_errors = [e for e in validation_errors if "SyntaxError" in e]
                if syntax_errors:
                    logger.warning(
                        "Auto-fixing %d syntax errors in generated code",
                        len(syntax_errors),
                    )
                    self._auto_fix_syntax_errors(output_dir, syntax_errors)
                    # Re-validate after auto-fix
                    validation_errors = self._validate_generated_python(output_dir)

                # v0.8.12: Never abort generation. Log errors and proceed.
                # The agent runtime will self-heal at startup.
                if validation_errors:
                    for err in validation_errors:
                        logger.warning("Validation issue (agent will self-heal): %s", err)

            # v0.9: Tool stub detection — warn if any tools are non-functional stubs
            stub_tools = self._check_tool_stubs(output_dir)
            if stub_tools:
                logger.warning(
                    "CRITICAL: %d/%d tools are STUBS (non-functional): %s. "
                    "AI tool generation failed for these tools. "
                    "The agent will not work correctly. "
                    "Re-run 'agenthatch hatch' or implement tools manually.",
                    len(stub_tools),
                    len(variables.get("tool_metadata", [])),
                    ", ".join(stub_tools),
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
        # v0.8.11: Exclude skills/ directory (skill original code, not generated)
        skills_dir = output_dir / "skills"

        for py_file in output_dir.rglob("*.py"):
            if skills_dir in py_file.parents or py_file.parent == skills_dir:
                continue  # Skip skill's original code

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

    # ── Tool stub detection (v0.9) ──────────────────────────────────────

    @staticmethod
    def _check_tool_stubs(output_dir: Path) -> list[str]:
        """Detect non-functional STUB tools in generated tools.py.

        A stub tool is a function body that only contains the error message:
          "AI tool generation did not produce a valid implementation"

        Returns list of tool function names that are stubs.
        """
        stub_tools: list[str] = []
        stub_signature = "AI tool generation did not produce a valid implementation"

        # Find tools.py in the generated output
        for tools_py in output_dir.rglob("tools.py"):
            content = tools_py.read_text(encoding="utf-8")
            if stub_signature not in content:
                continue

            # Parse with AST to find which functions are stubs
            try:
                import ast as _ast
                tree = _ast.parse(content)
            except SyntaxError:
                continue

            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    # Check if the function body contains the stub message
                    body_text = _ast.get_source_segment(content, node)
                    if body_text and stub_signature in body_text:
                        stub_tools.append(node.name)

        return stub_tools

    # ── Syntax auto-fix (v0.8.11) ────────────────────────────────────

    @staticmethod
    def _auto_fix_syntax_errors(
        output_dir: Path, syntax_errors: list[str],
    ) -> None:
        """Auto-fix common syntax errors in generated Python files.

        Handles:
        - unexpected indent / expected an indented block
        - inconsistent indentation (mixed tabs/spaces)
        - extra blank lines causing indentation resets
        """
        import re

        # Group errors by file
        files_to_fix: dict[str, list[int]] = {}
        for err in syntax_errors:
            match = re.match(r"(.+?):(\d+):", err)
            if match:
                fname = match.group(1)
                lineno = int(match.group(2))
                full_path = output_dir / fname
                if full_path.exists():
                    files_to_fix.setdefault(str(full_path), []).append(lineno)

        for filepath, error_lines in files_to_fix.items():
            path = Path(filepath)
            try:
                content = path.read_text(encoding="utf-8")
                lines = content.split("\n")
                fixed = GenerateEngine._normalize_indentation(lines, error_lines)
                path.write_text("\n".join(fixed), encoding="utf-8")
                logger.info("Auto-fixed indentation in %s", path.name)
            except Exception as e:
                logger.warning("Auto-fix failed for %s: %s", path.name, e)

    @staticmethod
    def _normalize_indentation(
        lines: list[str], error_lines: list[int],
    ) -> list[str]:
        """Normalize indentation in Python source lines.

        Strategy:
        1. Replace tabs with 4 spaces
        2. For lines with "unexpected indent" errors, reduce indent by 4 spaces
        3. For lines needing "expected an indented block", add 4-space indent
        4. Detect and fix inconsistent indent in adjacent lines
        """
        # Replace all tabs with 4 spaces
        fixed = [line.replace("\t", "    ") for line in lines]

        # For each error line, check context and fix
        for lineno in error_lines:
            if lineno < 1 or lineno > len(fixed):
                continue
            idx = lineno - 1  # 0-based

            # v0.8.17: Handle lines that should be indented after a colon
            # (e.g. try:/if:/for: with non-indented body).
            # Previous code only fixed lines with 0 or <4 spaces of indent,
            # missing the common case where AI-generated code has exactly
            # the base indent (4 spaces) for every line.
            if idx > 0 and fixed[idx - 1].rstrip().endswith(":"):
                prev_line = fixed[idx - 1]
                prev_indent = len(prev_line) - len(prev_line.lstrip(" "))
                curr_indent = len(fixed[idx]) - len(fixed[idx].lstrip(" "))
                # Fix: indent to prev_indent + 4 when current line is at
                # same or lower indent than the colon line.
                if curr_indent <= prev_indent and fixed[idx].strip():
                    fixed[idx] = " " * (prev_indent + 4) + fixed[idx].lstrip(" ")
                continue

            # Check if this line has more indent than context expects
            if idx > 0:
                prev_line = fixed[idx - 1]
                prev_indent = len(prev_line) - len(prev_line.lstrip(" "))
                curr_indent = len(fixed[idx]) - len(fixed[idx].lstrip(" "))
                # v0.8.16: If previous line doesn't end with ':' and current
                # line has deeper indent, reduce to prev line's indent level.
                # (was: curr_indent > prev_indent + 4 — missed exact +4 jumps)
                deeper = curr_indent > prev_indent
                colon_ended = not prev_line.rstrip().endswith(":")
                if colon_ended and deeper and fixed[idx].strip():
                    fixed[idx] = " " * prev_indent + fixed[idx].lstrip(" ")

            # v0.8.19: Fix block-breaker keywords (else/elif/except/finally)
            # that are at the wrong indent level.  The AI sometimes places
            # them at the parent block's body indent instead of the correct
            # (deeper) indent.
            #
            # Heuristic: try increasing the indent of the breaker and any
            # immediately following lines that belong to its body (indent
            # >= curr_indent, but stop at another block keyword at the
            # same indent level).
            stripped_line = fixed[idx].lstrip()
            if stripped_line.split():
                first_tok = stripped_line.split()[0].rstrip(":")
                if first_tok in ("else", "elif", "except", "finally"):
                    curr_indent = len(fixed[idx]) - len(fixed[idx].lstrip(" "))
                    bumped = list(fixed)
                    # Bump this line
                    bumped[idx] = " " * (curr_indent + 4) + stripped_line
                    # Bump following lines that are at >= curr_indent,
                    # but stop at another block keyword at curr_indent
                    BLOCK_KWS = frozenset([
                        "if", "elif", "else", "try", "except", "finally",
                        "for", "while", "with", "def", "class",
                    ])
                    for bi in range(idx + 1, len(bumped)):
                        bi_line = bumped[bi]
                        bi_stripped = bi_line.lstrip()
                        if not bi_stripped:
                            bumped[bi] = " " * (curr_indent + 4) + bi_stripped
                            continue
                        bi_indent = len(bi_line) - len(bi_stripped)
                        if bi_indent >= curr_indent:
                            # Stop at another block keyword at same indent
                            bi_first = bi_stripped.split()[0] if bi_stripped.split() else ""
                            bi_tok = bi_first.rstrip(":")
                            if bi_indent == curr_indent and bi_tok in BLOCK_KWS:
                                break
                            bumped[bi] = " " * (bi_indent + 4) + bi_stripped
                        else:
                            break  # indent went below — end of this block
                    # Test if the bump fixes the issue
                    try:
                        bumped_wrapper = (
                            "def _v():\n"
                            + "\n".join(bumped)
                            + "\n"
                        )
                        compile(bumped_wrapper, "<fix>", "exec")
                        fixed = bumped
                    except SyntaxError:
                        pass

        return fixed

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
