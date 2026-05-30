"""agenthatch run — launch a SkillAgent in interactive TUI mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

from agenthatch.agent.loop import RichToolCallEvent as RichTCEvent
from agenthatch.cli import console


def _render_header(spec: Any) -> Panel:
    """Render Agent info header."""
    return Panel(
        f"[bold bright_blue]{spec.identity.display_name}[/bold bright_blue] "
        f"({spec.identity.id} {spec.identity.version})\n"
        f"[dim]Type /help for commands, /quit or Ctrl+D to exit[/dim]",
        title="[bold bright_blue]Agent[/]",
        border_style="bright_blue",
    )


def _handle_command(user_input: str, agent: Any) -> str | None:
    """Handle /commands. Returns None if normal chat, str for inline response."""
    if not user_input.startswith("/"):
        return None

    cmd = user_input.strip().lower()
    if cmd == "/help":
        return _render_help(agent)
    elif cmd == "/clear":
        agent.ctx.history.clear()
        return "[ok]Conversation history cleared.[/ok]"
    elif cmd == "/status":
        return _render_status(agent)
    elif cmd in ("/quit", "/exit"):
        raise SystemExit(0)
    else:
        return f"[warn]Unknown command: {cmd}[/warn]. Type /help for available commands."


def _render_help(agent: Any) -> str:
    lines = [
        f"[bold bright_blue]{agent.spec.identity.display_name}[/]"
        f" — {agent.spec.intent.summary or 'No summary'}",
        "",
        "[bold]Available commands:[/bold]",
        "  [bold cyan]/help[/]    Show this help",
        "  [bold cyan]/clear[/]   Clear conversation history",
        "  [bold cyan]/status[/]  Show provider/model info",
        "  [bold cyan]/quit[/]    Exit (or Ctrl+D)",
    ]
    return "\n".join(lines)


def _render_status(agent: Any) -> str:
    lines = [
        f"[bold]Provider:[/bold] {agent.llm.provider_name}",
        f"[bold]Model:[/bold] {agent.llm.model}",
        f"[bold]Skill:[/bold] {agent.spec.identity.id} v{agent.spec.identity.version}",
    ]
    return "\n".join(lines)


def run_command(
    skill_name: str = typer.Argument(..., help="Skill ID or path to run"),  # noqa: B008
    provider: str = typer.Option(None, "--provider", "-p", help="Override provider"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Override API key"),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
) -> None:
    """Launch a SkillAgent in interactive TUI mode."""
    from agenthatch.agent.runtime import SkillAgent
    from agenthatch.config import Config
    from agenthatch.house.index import SkillhouseIndex

    config = Config.load()

    ahs_path: Path | None = None
    try:
        skill_path = Path(skill_name)
        if skill_path.is_dir():
            ahs_path = skill_path / "agenthatch.yaml"
    except Exception:
        ahs_path = None

    if ahs_path is None or not ahs_path.exists():
        skillhouse_cfg = config.get("skillhouse", {})
        skillhouse_path = skillhouse_cfg.get("path", ".agenthatch/skillhouse.json")
        idx = SkillhouseIndex(skillhouse_path)
        entry = idx.find_by_name(skill_name)
        if entry:
            ahs_path = Path(entry["ahs_path"])
        else:
            console.print(f"[warn]Skill '{skill_name}' not found.[/warn]")
            raise typer.Exit(1)

    if not ahs_path.exists():
        console.print(f"[warn]agenthatch.yaml not found for '{skill_name}'.[/warn]")
        console.print("[dim]Run [bold]agenthatch hatch[/bold] first.[/dim]")
        raise typer.Exit(1)

    agent = SkillAgent.from_ahspec(
        ahs_path,
        provider=provider,
        api_key=api_key,
        model=model,
    )

    console.print(_render_header(agent.spec))
    console.print("")

    try:
        while True:
            user_input = Prompt.ask("[bold green]You[/]")
            if not user_input.strip():
                continue

            cmd_result = _handle_command(user_input, agent)
            if cmd_result is not None:
                console.print(cmd_result)
                continue

            console.print("")
            console.print(f"[bold bright_blue]{agent.spec.identity.display_name}[/]")

            with console.status("[dim]Thinking...[/dim]"):
                response_text = agent.chat(user_input)
            console.print(Markdown(response_text))
            console.print("")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[dim]Goodbye![/dim]")
