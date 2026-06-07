"""agenthatch search — Search for skills in skillhouse.json.

Uses hybrid search: keyword (triggers) + embedding (satisfies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from agenthatch.cli import console
from agenthatch.config import Config

if TYPE_CHECKING:
    from agenthatch.house.index import SearchResult


def search_command(
    query: Annotated[
        str,
        typer.Argument(help="Search query (natural language)"),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top", "-k", help="Max results"),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    type_filter: Annotated[
        str | None,
        typer.Option("--type", help="Filter by capability type"),
    ] = None,
) -> None:
    """Search for skills in skillhouse.json.

    Uses hybrid search: keyword (triggers) + embedding (satisfies).

    Examples:
        agenthatch search "weather forecast"
        agenthatch search "code review" --top 10
        agenthatch search "data" --type data --json
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
    results = idx.search(query, top_k=top_k)

    # Apply type filter if specified
    if type_filter and results:
        filtered: list[SearchResult] = []
        for r in results:
            entry = idx.get_entry(r.skill_id) or {}
            provides = entry.get("interface", {}).get("provides", [])
            if any(c.get("type") == type_filter for c in provides):
                filtered.append(r)
        results = filtered

    if json_output:
        import json
        output = [
            {
                "skill_id": r.skill_id,
                "display_name": r.display_name,
                "summary": r.summary,
                "score": r.score,
                "match_source": r.match_source,
            }
            for r in results
        ]
        console.print_json(json.dumps(output, ensure_ascii=False))
        return

    if not results:
        console.print(f"[yellow]No skills found for '{query}'[/yellow]")
        return

    # Rich table output
    table = Table(title=f"Search: {query}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Skill", style="accent")
    table.add_column("Summary")
    table.add_column("Score", justify="right")
    table.add_column("Match", style="dim")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.display_name,
            r.summary[:80] + ("..." if len(r.summary) > 80 else ""),
            f"{r.score:.2f}",
            r.match_source,
        )

    console.print(table)
