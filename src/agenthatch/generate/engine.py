"""GenerateEngine — Phase 3: Agent generation from AHSSPEC via Jinja2 templates.

Extracts variables from AHSSPEC and renders Jinja2 templates to produce
a self-contained, independently-runnable Agent directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2

logger = logging.getLogger(__name__)

# Template file → output file mapping (relative to agent output root)
TEMPLATE_MAP: dict[str, str] = {
    "pyproject.toml.j2": "pyproject.toml",
    "agent.py.j2": "src/{package_name}/agent.py",
    "cli.py.j2": "src/{package_name}/cli.py",
    "tools.py.j2": "src/{package_name}/tools.py",
    "runtime.toml.j2": "runtime.toml",
    "README.md.j2": "README.md",
}


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

        def python_repr(value: str) -> str:
            """Generate Python-compatible string literal via json.dumps."""
            return json.dumps(value, ensure_ascii=False)

        env.filters["python_escape"] = python_escape
        env.filters["python_repr"] = python_repr
        return env

    # ── variable extraction ───────────────────────────────────────────

    def extract_variables(self, ahspec: dict) -> dict[str, Any]:
        """Extract template variables from an AHSSPEC dict.

        Handles both raw YAML dicts and Pydantic model dumps.
        """
        identity = ahspec.get("identity", {})
        intent = ahspec.get("intent", {})
        interface = ahspec.get("interface", {})
        base = ahspec.get("base", {})
        instructions = ahspec.get("instructions", {})

        agent_name = identity.get("id", "unknown-agent")
        agent_class = identity.get("display_name", "UnknownAgent")
        version = identity.get("version", "0.1.0")

        # Derive package_name: kebab-case → snake_case
        package_name = agent_name.replace("-", "_")

        # Description from intent summary
        description = intent.get("summary", "")

        # Workflow: can be a list of step dicts or a string
        workflow = instructions.get("workflow", "")
        if isinstance(workflow, list):
            workflow = self._format_workflow(workflow)

        output_tpl = instructions.get("output_template", "")

        # Rules: list of strings
        rules = instructions.get("rules", [])

        # Requires: list of capability names (strings) or dicts
        requires = self._extract_requires(interface.get("requires", []))

        # Base runtime
        base_runtime = base.get("runtime", "python3.11") if base else "python3.11"

        # Model: derived from base runtime or default
        model = "gpt-4o"

        # Tools: list of provide capability names
        tools = self._extract_tool_names(interface.get("provides", []))

        return {
            "agent_name": agent_name,
            "agent_class": agent_class,
            "version": version,
            "package_name": package_name,
            "description": description,
            "workflow": workflow,
            "output_tpl": output_tpl,
            "rules": rules,
            "requires": requires,
            "base_runtime": base_runtime,
            "llm_provider": "openai",  # Default LLM provider (user overrides in runtime.toml)
            "model": model,
            "tools": tools,
        }

    @staticmethod
    def _format_workflow(workflow: list) -> str:
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
    def _extract_requires(requires: list) -> list[str]:
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
    def _extract_tool_names(provides: list) -> list[str]:
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

    # ── generation ────────────────────────────────────────────────────

    def generate(
        self,
        ahspec: dict,
        output_dir: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        copy_skills: bool = True,
        skill_dir: Path | None = None,
    ) -> list[Path]:
        """Generate a complete Agent directory from an AHSSPEC dict.

        Args:
            ahspec: AHSSPEC dict (from agenthatch.yaml).
            output_dir: Target directory for the generated Agent.
            dry_run: If True, print files without writing.
            force: If True, overwrite existing output directory.
            copy_skills: If True, copy SKILL.md and resources.
            skill_dir: Source skill directory (for copying resources).

        Returns:
            List of Paths that were (or would be) written.
        """
        variables = self.extract_variables(ahspec)
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
            self._copy_skills(skill_dir, output_dir)

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

        return written

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
        self, ahspec: dict, output_dir: Path, variables: dict
    ) -> None:
        """Write a copy of agenthatch.yaml to the output root."""
        import yaml

        # Update agent status
        ahspec_copy = dict(ahspec)
        if "agent" not in ahspec_copy:
            ahspec_copy["agent"] = {}
        agent_cfg = ahspec_copy["agent"]
        if isinstance(agent_cfg, dict):
            agent_cfg["status"] = "generated"
            agent_cfg["generated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            ahspec_copy["agent"] = {
                "status": "generated",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        yaml_path = output_dir / "agenthatch.yaml"
        yaml_str = yaml.dump(
            json.loads(json.dumps(ahspec_copy, default=str)),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        yaml_path.write_text(yaml_str, encoding="utf-8")

    def _copy_skills(self, skill_dir: Path, output_dir: Path) -> None:
        """Copy SKILL.md and resource files from source skill directory."""
        dest_skills = output_dir / "skills"
        dest_skills.mkdir(parents=True, exist_ok=True)

        # Copy SKILL.md
        for md_name in ("SKILL.md", "skill.md", "Skill.md"):
            src_md = skill_dir / md_name
            if src_md.exists():
                shutil.copy2(src_md, dest_skills / md_name)
                break

        # Copy scripts/
        src_scripts = skill_dir / "scripts"
        if src_scripts.is_dir():
            dest_scripts = dest_skills / "scripts"
            if dest_scripts.exists():
                shutil.rmtree(dest_scripts)
            shutil.copytree(src_scripts, dest_scripts)

        # Copy references/
        src_refs = skill_dir / "references"
        if src_refs.is_dir():
            dest_refs = dest_skills / "references"
            if dest_refs.exists():
                shutil.rmtree(dest_refs)
            shutil.copytree(src_refs, dest_refs)


def generate_agent(
    ahspec: dict,
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