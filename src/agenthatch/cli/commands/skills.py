"""agenthatch skills — List, add, and delete registered skills."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from agenthatch.house.index import SkillhouseIndex

import typer
from rich.panel import Panel
from rich.table import Table

from agenthatch.cli import console
from agenthatch.config import Config

# ── skill subcommand group ───────────────────────────────────────────────

skill_app = typer.Typer(
    name="skill",
    help="Manage skills in the skillhouse registry.",
    no_args_is_help=True,
)


def _load_index() -> SkillhouseIndex:
    from agenthatch.house.index import SkillhouseIndex

    config = Config.load()
    skillhouse_config = config.get("skillhouse", {})
    skillhouse_path = (
        skillhouse_config.get("path", ".agenthatch/skillhouse.json")
        if isinstance(skillhouse_config, dict)
        else ".agenthatch/skillhouse.json"
    )
    sh_path = Path(skillhouse_path)
    if not sh_path.is_absolute():
        sh_path = Path.cwd() / sh_path
    if not sh_path.exists():
        console.print("[yellow]No skillhouse.json found. Run 'agenthatch hatch' first.[/yellow]")
        raise typer.Exit(code=1)
    return SkillhouseIndex(str(sh_path))


def _format_status(e: dict[str, Any], raw_entry: dict[str, Any] | None) -> str:
    """Format status column with hatch date or unhatched indicator."""
    agent_cfg = (raw_entry or {}).get("agent", {})
    if isinstance(agent_cfg, dict):
        status = agent_cfg.get("status", "unhatched")
        hatched_at = agent_cfg.get("hatched_at", "")
    else:
        status = getattr(agent_cfg, "status", "unhatched") if agent_cfg else "unhatched"
        hatched_at = getattr(agent_cfg, "hatched_at", "") if agent_cfg else ""

    if status == "discovered" or e.get("status") == "discovered":
        return "[yellow]unhatched[/yellow]"

    if status == "hatched" and hatched_at:
        # Show just the date portion
        date_str = str(hatched_at)[:10] if hatched_at else ""
        return f"[green]hatched[/green] [dim]{date_str}[/dim]"

    if status == "hatched":
        return "[green]hatched[/green]"

    return "[yellow]unhatched[/yellow]"


# ── skill list ───────────────────────────────────────────────────────────

@skill_app.command("list")
def list_skills(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    type_filter: Annotated[
        str | None,
        typer.Option("--type", help="Filter by capability type"),
    ] = None,
    discover: Annotated[
        bool,
        typer.Option("--discover", help="Auto-discover skills from filesystem"),
    ] = True,
) -> None:
    """List all registered skills and discover new ones.

    Examples:
        agenthatch skill list
        agenthatch skill list --type data
        agenthatch skill list --json
        agenthatch skill list --no-discover
    """
    idx = _load_index()
    entries = idx.list_all()

    # Auto-discover skills not yet in skillhouse
    if discover:
        from agenthatch.house.discovery import discover_all

        discovery_result = discover_all()
        registered_ids = {e["id"] for e in entries}
        for skill in discovery_result.skills:
            if skill.skill_id not in registered_ids:
                entries.append({
                    "id": skill.skill_id,
                    "display_name": skill.skill_id,
                    "version": "",
                    "summary": f"[discovered] from {skill.source}",
                    "status": "discovered",
                })

    # Apply type filter
    if type_filter:
        filtered: list[dict[str, Any]] = []
        for e in entries:
            sid = e["id"]
            raw_entry = idx.get_entry(sid) or {}
            provides = raw_entry.get("interface", {}).get("provides", [])
            if any(c.get("type") == type_filter for c in provides):
                filtered.append(e)
        entries = filtered

    # Enrich entries with status from agent config
    for e in entries:
        sid = e["id"]
        raw_entry = idx.get_entry(sid) or {}
        agent_cfg = raw_entry.get("agent", {})
        if isinstance(agent_cfg, dict):
            status = agent_cfg.get("status", "unhatched")
            hatched_at = agent_cfg.get("hatched_at", "")
        else:
            status = getattr(agent_cfg, "status", "unhatched") if agent_cfg else "unhatched"
            hatched_at = getattr(agent_cfg, "hatched_at", "") if agent_cfg else ""
        e["status"] = status
        e["hatched_at"] = str(hatched_at)[:10] if hatched_at else ""

    if json_output:
        console.print_json(json.dumps(entries, ensure_ascii=False))
        return

    if not entries:
        console.print("[yellow]No skills registered.[/yellow]")
        return

    table = Table(title="Registered Skills")
    table.add_column("ID", style="accent")
    table.add_column("Display Name")
    table.add_column("Status", justify="center")
    table.add_column("Summary")

    for e in entries:
        sid = e["id"]
        raw_entry = idx.get_entry(sid) or {}
        status_display = _format_status(e, raw_entry)
        summary = e.get("summary", "")
        summary_display = summary[:80] + ("..." if len(summary) > 80 else "")
        table.add_row(
            e["id"],
            e.get("display_name", e["id"]),
            status_display,
            summary_display,
        )

    console.print(table)
    hatched_count = sum(1 for e in entries if e.get("status") == "hatched")
    console.print(
        f"\n[dim]{hatched_count} hatched, "
        f"{len(entries) - hatched_count} unhatched[/dim]"
    )


# ── skill add ────────────────────────────────────────────────────────────

@skill_app.command("add")
def add_skill(
    path: Annotated[
        str,
        typer.Argument(help="Path to skill directory or SKILL.md file"),
    ],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Override skill ID"),
    ] = None,
) -> None:
    """Register a new skill without hatching it.

    Parses the SKILL.md and registers it in skillhouse with
    status "unhatched". Use 'agenthatch hatch' to hatch it later.

    Examples:
        agenthatch skill add ./my-skill/
        agenthatch skill add ~/skills/pdf-editor/SKILL.md
        agenthatch skill add ./custom --name my-renamed-skill
    """
    skill_path = Path(path).expanduser().resolve()

    # Determine SKILL.md location
    if skill_path.is_dir():
        md_path = skill_path / "SKILL.md"
    elif skill_path.name.endswith(".md"):
        md_path = skill_path
        skill_path = skill_path.parent
    else:
        console.print(f"[error]Not a skill directory or markdown file: {path}[/error]")
        raise typer.Exit(code=1)

    if not md_path.exists():
        console.print(f"[error]SKILL.md not found at {md_path}[/error]")
        raise typer.Exit(code=1)

    # Parse SKILL.md frontmatter to extract identity
    try:
        import frontmatter
        post = frontmatter.load(str(md_path))
    except Exception as e:
        console.print(f"[error]Failed to parse SKILL.md: {e}[/error]")
        raise typer.Exit(code=1) from e

    fm = post.metadata or {}
    skill_id: str = name or fm.get("name", "") or skill_path.name  # type: ignore[assignment]
    display_name = fm.get("display_name") or str(skill_id).replace("-", " ").title()

    # Load index and register
    try:
        idx = _load_index()
    except typer.Exit:
        # No skillhouse yet — create a minimal one
        from agenthatch.house.index import SkillhouseIndex
        idx = SkillhouseIndex()

    # Create a minimal AHSSpec for registration

    from agenthatch.skill.spec import (
        AgentConfig,
        AHSSpec,
        BaseSpec,
        Composition,
        Identity,
        Instructions,
        Intent,
        Interface,
        Resources,
    )

    ahs_spec = AHSSpec(
        identity=Identity(
            id=skill_id,
            display_name=display_name,  # type: ignore[arg-type]
            version="",
        ),
        intent=Intent(
            triggers=[],
            satisfies=[],
            summary=fm.get("description", post.content[:200].strip()),  # type: ignore[arg-type]
        ),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
        resources=Resources(),
        composition=Composition(),
        agent=AgentConfig(
            status="unhatched",
            hatched_at=None,
            generated_at=None,
        ),
    )

    yaml_path = skill_path / "agenthatch.yaml"
    idx.add_entry(skill_id, ahs_spec, str(yaml_path))

    console.print(
        Panel(
            f"[bold]{skill_id}[/bold] registered as [yellow]unhatched[/yellow]\n"
            f"[dim]Display name: {display_name}[/dim]\n"
            f"[dim]Source: {skill_path}[/dim]\n\n"
            f"Run [bold]agenthatch hatch {skill_path}[/bold] to hatch this skill.",
            title="[accent]Skill Added[/accent]",
            border_style="green",
        )
    )


# ── skill delete ─────────────────────────────────────────────────────────

@skill_app.command("delete")
def delete_skill(
    skill_id: Annotated[
        str,
        typer.Argument(help="Skill ID to delete (kebab-case)"),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
    remove_agent: Annotated[
        bool,
        typer.Option("--remove-agent", help="Also delete generated agent directory"),
    ] = False,
) -> None:
    """Remove a skill from the skillhouse registry.

    By default, this only removes the skill from the registry index.
    Use --remove-agent to also delete the generated agent directory.

    Examples:
        agenthatch skill delete my-skill
        agenthatch skill delete my-skill --force
        agenthatch skill delete my-skill --remove-agent
    """
    idx = _load_index()

    raw_entry = idx.get_entry(skill_id)
    if raw_entry is None:
        console.print(f"[yellow]Skill '{skill_id}' not found in registry.[/yellow]")
        raise typer.Exit(code=0)

    display_name = raw_entry.get("identity", {}).get("display_name", skill_id)
    hatched = (
        raw_entry.get("agent", {}).get("status") == "hatched"
        if isinstance(raw_entry.get("agent"), dict)
        else False
    )
    status_str = "[green]hatched[/green]" if hatched else "[yellow]unhatched[/yellow]"

    # Confirmation
    if not force:
        agent_output = raw_entry.get("agent_output", "")
        extra = ""
        if agent_output and remove_agent:
            extra = f"\n  [red]Will also delete:[/red] [dim]{agent_output}[/dim]"
        elif agent_output:
            extra = f"\n  [dim]Agent directory (not deleted): {agent_output}[/dim]"

        console.print(
            Panel(
                f"Delete [bold]{skill_id}[/bold] ({display_name})?\n"
                f"Status: {status_str}{extra}",
                title="[error]Confirm Delete[/error]",
                border_style="red",
            )
        )
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Remove agent directory if requested
    if remove_agent:
        agent_output = raw_entry.get("agent_output", "")
        if agent_output:
            agent_dir = Path(agent_output)
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
                console.print(f"[dim]Deleted agent directory: {agent_dir}[/dim]")

    # Remove from index
    idx.remove_entry(skill_id)
    console.print(
        f"[ok]✓[/ok]  [bold]{skill_id}[/bold] removed from skillhouse."
    )


# ── legacy skills command (redirects to list) ────────────────────────────

def skills_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    type_filter: Annotated[
        str | None,
        typer.Option("--type", help="Filter by capability type"),
    ] = None,
    discover: Annotated[
        bool,
        typer.Option("--discover", help="Auto-discover skills from filesystem"),
    ] = True,
) -> None:
    """List all registered skills (alias for 'agenthatch skill list').

    Examples:
        agenthatch skills
        agenthatch skills --type data
        agenthatch skills --json
    """
    list_skills(json_output=json_output, type_filter=type_filter, discover=discover)
