"""agenthatch skills — List registered skills in skillhouse.json."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.table import Table

from agenthatch.cli import console
from agenthatch.config import Config


def skills_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    type_filter: Annotated[
        str | None,
        typer.Option("--type", help="Filter by capability type"),
    ] = None,
) -> None:
    """List all registered skills in skillhouse.json.

    Examples:
        agenthatch skills
        agenthatch skills --type data
        agenthatch skills --json
    """
    from agenthatch.house.index import SkillhouseIndex

    config = Config.load()
    skillhouse_config = config.get("skillhouse", {}) if "skillhouse" in config else {}
    skillhouse_path = skillhouse_config.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_config, dict) else ".agenthatch/skillhouse.json"

    from pathlib import Path

    sh_path = Path(skillhouse_path)
    if not sh_path.is_absolute():
        sh_path = Path.cwd() / sh_path

    if not sh_path.exists():
        console.print("[yellow]No skillhouse.json found. Run 'agenthatch hatch' first.[/yellow]")
        raise typer.Exit(code=0)

    idx = SkillhouseIndex(str(sh_path))
    entries = idx.list_all()

    # Apply type filter if specified
    if type_filter:
        filtered: list[dict[str, Any]] = []
        for e in entries:
            sid = e["id"]
            raw_entry = idx.get_entry(sid) or {}
            provides = raw_entry.get("interface", {}).get("provides", [])
            if any(c.get("type") == type_filter for c in provides):
                filtered.append(e)
        entries = filtered

    if json_output:
        import json
        console.print_json(json.dumps(entries, ensure_ascii=False))
        return

    if not entries:
        console.print("[yellow]No skills registered.[/yellow]")
        return

    table = Table(title="Registered Skills")
    table.add_column("ID", style="accent")
    table.add_column("Display Name")
    table.add_column("Version", justify="right")
    table.add_column("Summary")

    for e in entries:
        sid = e["id"]
        raw_entry = idx.get_entry(sid) or {}
        if raw_entry.get("status") == "discovered":
            version_display = "[dim]not hatched[/dim]"
            summary_display = "[dim](run 'agenthatch hatch' first)[/dim]"
        else:
            version_display = e["version"]
            summary_display = e["summary"][:80] + ("..." if len(e["summary"]) > 80 else "")
        table.add_row(
            e["id"],
            e["display_name"],
            version_display,
            summary_display,
        )

    console.print(table)
    console.print(f"\n[dim]{len(entries)} skill(s) registered[/dim]")
