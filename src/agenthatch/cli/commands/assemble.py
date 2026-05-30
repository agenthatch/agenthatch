"""agenthatch assemble — multi-skill assembly (v0.4 experimental)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.panel import Panel

from agenthatch.cli import console
from agenthatch.house.resolver import resolve_dependencies


def assemble_command(
    skills: list[str] = typer.Argument(..., help="Skill IDs to assemble"),  # noqa: B008
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print assembly plan without running"
    ),
) -> None:
    """Assemble multiple skills into a composite Agent (experimental)."""
    from agenthatch.config import Config
    from agenthatch.house.index import SkillhouseIndex

    config = Config.load()
    skillhouse_cfg = config.get("skillhouse", {})
    skillhouse_path = skillhouse_cfg.get("path", ".agenthatch/skillhouse.json")
    idx = SkillhouseIndex(skillhouse_path)

    skill_specs: list[dict[str, Any]] = []
    for skill_name in skills:
        entry = idx.find_by_name(skill_name)
        if not entry:
            console.print(f"[warn]Skill '{skill_name}' not found in index.[/warn]")
            raise typer.Exit(1)
        ahs_path = Path(entry["ahs_path"])
        if not ahs_path.exists():
            console.print(f"[warn]agenthatch.yaml not found for '{skill_name}'.[/warn]")
            raise typer.Exit(1)
        spec = yaml.safe_load(ahs_path.read_text())
        skill_specs.append({"id": skill_name, "spec": spec, "path": ahs_path})

    if not skill_specs:
        console.print("[warn]No valid skills to assemble.[/warn]")
        raise typer.Exit(1)

    if dry_run:
        _print_assembly_plan(skill_specs, idx)
    else:
        console.print("[dim]Assembly execution not yet implemented (v1.5.0).[/dim]")
        console.print("[dim]Use --dry-run to preview the plan.[/dim]")


def _print_assembly_plan(skill_specs: list[dict[str, Any]], idx: Any) -> None:
    """Print the assembly plan for --dry-run."""
    from agenthatch.house.resolver import is_builtin

    providers: dict[str, list[str]] = idx._data.get("providers", {})
    dep_graph: dict[str, list[str]] = idx._data.get("topology", {}).get(
        "dependency_graph", {}
    )

    all_caps: list[str] = []
    for s in skill_specs:
        interface = s["spec"].get("interface", {})
        for prov in interface.get("provides", []):
            all_caps.append(prov.get("capability", ""))

    order = resolve_dependencies(
        all_caps, providers, dep_graph, idx._data.get("entries", {})
    )

    console.print(
        Panel(
            "\n".join(
                [f"  {i+1}. {sid}" for i, sid in enumerate(order or [s["id"] for s in skill_specs])]
            ),
            title="[bold]Assembly Plan[/bold]",
            border_style="cyan",
        )
    )

    for s in skill_specs:
        interface = s["spec"].get("interface", {})
        requires = interface.get("requires", [])
        builtins = [req["capability"] for req in requires if is_builtin(req.get("capability", ""))]
        if builtins:
            console.print(f"  Builtins for [bold]{s['id']}[/]: {', '.join(builtins)}")
