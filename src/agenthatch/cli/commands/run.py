"""agenthatch run — launch a SkillAgent in interactive TUI mode.

v0.5: Rich Live streaming with tool call display replaces opaque spinner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agenthatch.agent.loop import RichToolCallEvent
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
    features = agent.llm.features
    caps: list[str] = []
    if features.supports_tools:
        caps.append("tools")
    if features.supports_stream_tools:
        caps.append("stream+tools")
    if features.supports_json_mode:
        caps.append("json_mode")
    if features.supports_reasoning_content:
        caps.append("reasoning")
    lines.append(f"[bold]Capabilities:[/bold] {', '.join(caps) if caps else 'none'}")
    if features.available_models:
        lines.append(f"[bold]Available models:[/bold] {', '.join(features.available_models)}")
    estimated_tokens = agent.ctx.estimate_input_tokens()
    lines.append(f"[bold]Est. input tokens:[/bold] {estimated_tokens}")
    return "\n".join(lines)


def _stream_response(agent: Any, user_input: str) -> str:
    """Stream agent response with live tool call display."""
    full_text: list[str] = []
    tool_status: dict[str, str] = {}

    def render() -> str:
        lines = [f"[bold bright_blue]{agent.spec.identity.display_name}[/]"]
        for name, status in tool_status.items():
            lines.append(f"  [cyan]{name}[/] {status}")
        if full_text:
            lines.append("".join(full_text)[-300:])
        return "\n".join(lines) or "[dim]Thinking...[/dim]"

    with Live(Panel(render(), title="Agent"), refresh_per_second=8) as live:
        gen = agent.chat_stream(user_input)
        final_text = ""
        while True:
            try:
                event = next(gen)
            except StopIteration as e:
                final_text = e.value
                break
            if isinstance(event, RichToolCallEvent):
                if event.phase == "start":
                    tool_status[event.tool_name] = "[bold yellow]running...[/]"
                elif event.phase == "done":
                    elapsed = f"{event.elapsed:.1f}s" if event.elapsed else ""
                    preview = (event.result_preview or "")[:80]
                    tool_status[event.tool_name] = (
                        f"[bold green]done[/] ({elapsed}) "
                        f"[dim]→ {preview}[/]"
                    )
            elif isinstance(event, str):
                full_text.append(event)
            live.update(Panel(render(), title="Agent"))
        live.update(Panel(
            "[bold bright_blue]Response:[/]\n"
            + (final_text or "".join(full_text) or "(no response)"),
            title="Agent"
        ))
    return final_text or "".join(full_text) or "(no response)"


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

            try:
                response_text = _stream_response(agent, user_input)
            except Exception as e:
                response_text = f"Agent error: {e}"
                console.print(f"[error]{response_text}[/error]")
                continue
            console.print(Markdown(response_text))
            console.print("")

    except (KeyboardInterrupt, SystemExit, EOFError):
        console.print()
