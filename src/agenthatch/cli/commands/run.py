"""agenthatch run — launch an independent Agent in interactive TUI mode.

v0.6: Rewritten as Agent-direct launcher.  Finds the hatched Agent
directory, confirms dependencies, and launches in-process with Rich Live TUI.

Old config-driven SkillAgent.from_ahspec() path has been removed.
"""

from __future__ import annotations

import sys
import time
import tomllib
from importlib import util as _importlib_util
from pathlib import Path
from typing import Any

import typer
from agenthatch_core.config import (
    inherit_api_key,
    resolve_runtime_config,
)
from agenthatch_core.loop.agent_loop import RichToolCallEvent
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.styles import Style as PTStyle
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agenthatch.cli import console
from agenthatch.cli.commands._completion import _complete_skill_name

PT_STYLE = PTStyle.from_dict({
    "prompt": "bold green",
})


def run_command(
    skill_name: str = typer.Argument(  # noqa: B008
        ...,
        help="Skill ID or path to run",
        autocompletion=_complete_skill_name,
    ),
    provider: str = typer.Option(None, "--provider", "-p", help="Override provider"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Override API key"),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
) -> None:
    """Launch an independent Agent in interactive TUI mode.

    v0.6: Finds the hatched Agent directory (from agenthatch hatch),
    confirms dependencies, and launches in-process.

    Examples:
        agenthatch run weather-reporter
        agenthatch run ./weather-reporter-agent/
        agenthatch run weather-reporter --provider deepseek --model deepseek-v4-pro
    """
    # 1. Find the hatched Agent directory
    agent_dir = _find_hatched_agent(skill_name)

    # 2. Load runtime config
    runtime_config = _load_runtime_toml(agent_dir)

    # 3. Apply CLI overrides
    if provider:
        runtime_config.setdefault("llm", {})["provider"] = provider
    if api_key:
        runtime_config.setdefault("llm", {})["api_key"] = api_key
    if model:
        runtime_config.setdefault("llm", {})["model"] = model

    # 4. Inherit API key from global config if still missing after CLI overrides
    from agenthatch_core.config import inherit_api_key
    runtime_config = inherit_api_key(runtime_config)

    # 5. Resolve API key source for startup indicator
    key_source = _resolve_key_source(api_key, runtime_config, agent_dir)

    # 6. In-process launch
    _launch(agent_dir, skill_name, runtime_config, key_source)


# ── Agent Discovery ─────────────────────────────────────────────────────────

def _find_hatched_agent(skill_name: str) -> Path:
    """Find the hatched Agent directory for a skill name.

    Lookup priority:
      1. Direct path (./<name>-agent/ or user-provided path)
      2. skillhouse.json index (agent_output field)
      3. ~/.agenthatch/agents/<name>-agent/
      4. Error
    """
    from agenthatch.config import Config
    from agenthatch.house.index import SkillhouseIndex

    # Priority 1: Direct path (only when input looks like a filesystem path)
    input_path = Path(skill_name)
    looks_like_path = (
        "/" in skill_name
        or "\\" in skill_name
        or bool(input_path.suffix)
        or skill_name in (".", "..")
    )
    if looks_like_path and input_path.exists():
        if input_path.is_dir():
            return _validate_agent_dir(input_path)
        return _validate_agent_dir(input_path.parent)

    # Check ./<name>-agent/
    cwd_agent = Path.cwd() / f"{skill_name}-agent"
    if cwd_agent.is_dir():
        return _validate_agent_dir(cwd_agent)

    # Priority 2: skillhouse.json index
    config = Config.load()
    skillhouse_cfg = config.get("skillhouse", {})
    skillhouse_path = skillhouse_cfg.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_cfg, dict) else ".agenthatch/skillhouse.json"

    idx_path = Path(skillhouse_path)
    if not idx_path.is_absolute():
        idx_path = Path.cwd() / idx_path

    if idx_path.exists():
        idx = SkillhouseIndex(str(idx_path))
        entry = idx.find_by_name(skill_name)
        if entry:
            agent_output = entry.get("agent_output", "")
            if agent_output:
                agent_dir = Path(agent_output)
                if agent_dir.is_dir():
                    return _validate_agent_dir(agent_dir)

    # Priority 3: ~/.agenthatch/agents/<name>-agent/
    home_agent = Path.home() / ".agenthatch" / "agents" / f"{skill_name}-agent"
    if home_agent.is_dir():
        return _validate_agent_dir(home_agent)

    # Priority 4: Error
    _fail_agent_not_found(skill_name)
    raise RuntimeError("unreachable")


def _validate_agent_dir(agent_dir: Path) -> Path:
    """Validate that a directory looks like a hatched Agent."""
    # Check for key files
    has_pyproject = (agent_dir / "pyproject.toml").exists()
    has_agent_py = list(agent_dir.glob("src/*/agent.py"))

    if not has_pyproject and not has_agent_py:
        _fail_agent_not_found(str(agent_dir))

    return agent_dir.resolve()


def _fail_agent_not_found(name: str) -> None:
    """Print helpful error and exit."""
    console.print(f"[red]Error:[/red] Agent not found: '{name}'")
    console.print(
        "[dim]Run [bold]agenthatch hatch <name>[/bold] first to generate "
        "the Agent directory.[/dim]"
    )
    raise typer.Exit(code=1)


# ── Runtime Config Loading ──────────────────────────────────────────────────

def _load_runtime_toml(agent_dir: Path) -> dict[str, Any]:
    """Load and resolve runtime.toml from the Agent directory."""
    toml_path = agent_dir / "runtime.toml"
    if not toml_path.exists():
        return {}
    raw = tomllib.loads(toml_path.read_text())
    resolved = resolve_runtime_config(raw)
    return inherit_api_key(resolved)


def _resolve_key_source(
    cli_key: str | None,
    runtime_config: dict[str, Any],
    agent_dir: Path,
) -> str:
    """Determine which source provided the active API key.

    Returns a human-readable string like "CLI flag", "runtime.toml", etc.
    """
    import os

    llm = runtime_config.get("llm", {})
    active_key = llm.get("api_key", "")

    if not active_key:
        return "none (no key configured)"

    # CLI flag has highest priority
    if cli_key:
        return "CLI flag (--api-key)"

    # Check if runtime.toml has an api_key entry
    toml_path = agent_dir / "runtime.toml"
    if toml_path.exists():
        toml_cfg = tomllib.loads(toml_path.read_text())
        toml_key = toml_cfg.get("llm", {}).get("api_key", "")
        if toml_key and toml_key == active_key:
            return "runtime.toml (per-agent override)"

    # Check global config
    global_config = Path.home() / ".agenthatch" / "config.toml"
    if global_config.exists():
        gcfg = tomllib.loads(global_config.read_text())
        provider = gcfg.get("providers", {}).get("default", "")
        provider_cfg = gcfg.get("providers", {}).get(provider, {})
        global_key = provider_cfg.get("api_key", "")
        if global_key and global_key == active_key:
            return "agenthatch global config"

    # Check environment variables
    env_vars = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for var in env_vars:
        if os.environ.get(var) == active_key:
            return f"environment variable ({var})"

    return "runtime.toml"


# ── In-Process Launch ───────────────────────────────────────────────────────

def _launch(
    agent_dir: Path, skill_name: str, runtime_config: dict[str, Any], key_source: str = ""
) -> None:
    """Launch the Agent in-process with Rich Live TUI.

    Uses sys.path injection + importlib to import the Agent class,
    keeping full TUI control (unlike subprocess which loses it).
    """
    agent_pkg = skill_name.replace("-", "_")
    agent_src = agent_dir / "src"

    # Find the actual agent module
    agent_modules = list(agent_src.glob("*/agent.py"))
    if not agent_modules:
        console.print(
            f"[red]Error:[/red] No agent.py found in {agent_src}"
        )
        raise typer.Exit(code=1)

    agent_module_path = agent_modules[0]
    agent_pkg = agent_module_path.parent.name

    # Inject Agent's src directory into sys.path
    path_injected = False
    if str(agent_src) not in sys.path:
        sys.path.insert(0, str(agent_src))
        path_injected = True

    try:
        # Dynamic import using importlib
        spec = _importlib_util.spec_from_file_location(
            agent_pkg, agent_module_path
        )
        if spec is None or spec.loader is None:
            console.print(
                f"[red]Error:[/red] Cannot load agent module from {agent_module_path}"
            )
            raise typer.Exit(code=1)

        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get Agent class from the AGENT_CLASS constant
        agent_class_name = getattr(module, "AGENT_CLASS", None)
        if agent_class_name is None:
            console.print(
                "[red]Error:[/red] Agent module missing AGENT_CLASS constant."
            )
            raise typer.Exit(code=1)

        AgentClass = getattr(module, agent_class_name, None)
        if AgentClass is None:
            console.print(
                f"[red]Error:[/red] Class '{agent_class_name}' not found "
                f"in agent module."
            )
            raise typer.Exit(code=1)

        # Instantiate and run
        agent = AgentClass(runtime_config=runtime_config)
        _run_interactive_tui(agent, key_source)

    finally:
        if path_injected:
            sys.path.remove(str(agent_src))


# ── Interactive TUI ─────────────────────────────────────────────────────────

def _run_interactive_tui(agent: Any, key_source: str = "") -> None:
    """Rich Live TUI for interactive Agent conversation.

    This is the premium TUI experience preserved from v0.5:
    - Streaming text rendering via Rich Live
    - Tool call status display (running/done/elapsed)
    - /commands support (/help, /compact, /clear, /quit, /key-source)
    """
    console.print(
        Panel(
            f"[bold bright_blue]{agent.identity.display_name}[/] "
            f"({agent.identity.id} {agent.identity.version})",
            title="Agent", border_style="bright_blue"
        )
    )
    if key_source:
        console.print(f"[dim]API key: {key_source}[/dim]")
    console.print("[dim]Type /help for commands, /quit or Ctrl+D to exit[/dim]")
    console.print()

    try:
        while True:
            # prompt_toolkit replaces Rich.Prompt for CJK input correctness
            # (macOS libedit bug: backspace drifts on multi-byte characters)
            try:
                user_input = pt_prompt(
                    [("class:prompt", "You: ")],
                    style=PT_STYLE,
                )
            except EOFError:
                raise
            if not user_input.strip():
                continue

            cmd_result = _handle_command(user_input, agent)
            if cmd_result is not None:
                console.print(cmd_result)
                continue

            console.print()
            console.print(f"[bold bright_blue]{agent.identity.display_name}[/]")

            try:
                response_text = _stream_response(agent, user_input)
            except Exception as e:
                response_text = f"Agent error: {e}"
                console.print(f"[error]{response_text}[/error]")
                continue

            console.print(Markdown(response_text))
            # v0.7: Show token usage if available
            if hasattr(agent, "token_counter"):
                snap = agent.token_counter.snapshot()
                if snap["total_tokens"] > 0:
                    parts = [f"Tokens: {snap['total_tokens']:,}"]
                    detail_parts = []
                    if snap["prompt_tokens"]:
                        detail_parts.append(f"in: {snap['prompt_tokens']:,}")
                    if snap["completion_tokens"]:
                        detail_parts.append(f"out: {snap['completion_tokens']:,}")
                    if detail_parts:
                        parts.append(f"({', '.join(detail_parts)})")
                    if snap.get("call_count"):
                        parts.append(f"[{snap['call_count']} calls]")
                    console.print(f"[dim]{' '.join(parts)}[/dim]")
            console.print()

    except (KeyboardInterrupt, SystemExit, EOFError):
        console.print()


def _stream_response(agent: Any, user_input: str) -> str:
    """Stream agent response with live tool call display and reasoning feedback."""
    full_text: list[str] = []
    reasoning_lines: list[str] = []
    tool_status: dict[str, str] = {}
    frame_title = f"[bold bright_blue]{agent.identity.display_name}[/]"
    start_time = time.monotonic()

    def statusbar() -> str:
        """Build the status bar line with live token/time info."""
        parts: list[str] = []
        if hasattr(agent, "token_counter"):
            snap = agent.token_counter.snapshot()
            if snap.get("total_tokens", 0) > 0:
                parts.append(f"[dim]Tokens:[/] {snap['total_tokens']:,}")
                parts.append(f"[dim]Calls:[/] {snap.get('call_count', 0)}")
        elapsed = time.monotonic() - start_time
        parts.append(f"[dim]⏱[/] {elapsed:.1f}s")
        return " │ ".join(parts)

    def render() -> str:
        lines = [frame_title]
        for name, status in tool_status.items():
            lines.append(f"  [cyan]{name}[/] {status}")
        if full_text:
            lines.append("".join(full_text)[-300:])
        elif reasoning_lines:
            body = "[dim]" + "\n".join(reasoning_lines[-3:]) + "[/]"
            lines.append(body)
        if not full_text and not reasoning_lines:
            lines.append("[dim]Thinking...[/dim]")
        lines.append("")  # spacer
        lines.append(statusbar())
        return "\n".join(lines)

    with Live(Panel(render(), title="Agent"), refresh_per_second=10) as live:
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
                # event could be reasoning_content or content delta — both are strings
                full_text.append(event)
            live.update(Panel(render(), title="Agent"))
        live.update(Panel("[dim]✓ Done[/dim]", title="Agent"))
    return final_text or "".join(full_text) or "(no response)"


# ── /commands ───────────────────────────────────────────────────────────────

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
    elif cmd == "/config":
        return _handle_config_command(agent)
    elif cmd == "/key-source":
        return _handle_key_source_command(agent)
    elif cmd == "/compact":
        try:
            agent.ctx.compact(
                agent.llm.model_max_tokens if agent.llm else 4096
            )
            return "[ok]Context compacted.[/ok]"
        except Exception as e:
            return f"[warn]Compact failed: {e}[/warn]"
    elif cmd in ("/quit", "/exit"):
        raise SystemExit(0)
    else:
        return (
            f"[warn]Unknown command: {cmd}[/warn]. "
            "Type /help for available commands."
        )


def _handle_config_command(agent: Any) -> str | None:
    """Interactive API key configuration via /config command."""
    agent_dir = _find_agent_dir_for_config(agent)
    if agent_dir is None:
        return "[warn]Cannot locate agent directory for config persistence.[/warn]"

    runtime_path = agent_dir / "runtime.toml"

    try:
        import tomli_w
    except ImportError:
        return "[warn]tomli_w not installed. Run: pip install tomli_w[/warn]"

    if not runtime_path.exists():
        runtime_path.write_text("[llm]\nprovider = \"deepseek\"\nmodel = \"deepseek-v4-pro\"\n")

    cfg = tomllib.loads(runtime_path.read_text())
    llm = cfg.setdefault("llm", {})
    current_key = llm.get("api_key", "")

    if current_key and not current_key.startswith("$"):
        key_display = "sk..." + current_key[-4:]
    elif not current_key:
        key_display = "auto-inherit from agenthatch config"
    else:
        key_display = "env var reference"

    console.print(Panel(
        f"Provider: {llm.get('provider', 'deepseek')}\n"
        f"Model: {llm.get('model', 'deepseek-v4-pro')}\n"
        f"API Key: {key_display}",
        title="API Key Configuration"
    ))
    console.print("  1. Switch Provider\n  2. Switch Model\n  3. Change API Key\n  4. Back")

    choice = Prompt.ask("Select", choices=["1", "2", "3", "4"])

    if choice == "1":
        new_provider = Prompt.ask("Provider", choices=["deepseek", "openai", "anthropic", "ollama"])
        llm["provider"] = new_provider
    elif choice == "2":
        llm["model"] = Prompt.ask("Model", default=llm.get("model", "deepseek-v4-pro"))
    elif choice == "3":
        llm["api_key"] = Prompt.ask("API Key", password=True)
    else:
        return None

    runtime_path.write_text(tomli_w.dumps(cfg))

    # Instant apply: rebuild LLMClient
    from agenthatch_core.llm.client import LLMClient
    agent.llm = LLMClient(
        provider=llm.get("provider", "deepseek"),
        model=llm.get("model", "deepseek-v4-pro"),
        api_key=llm.get("api_key"),
    )

    # Show key source
    if choice == "3":
        key_src = "runtime.toml (per-agent override)"
    elif llm.get("api_key"):
        key_src = "runtime.toml (per-agent override)"
    else:
        key_src = "agenthatch global config"
    return (
        f"[bold green]✓ Configuration updated and applied[/bold green]\n"
        f"  Source: {key_src}"
    )


def _find_agent_dir_for_config(agent: Any) -> Path | None:
    """Find the agent directory for config persistence."""
    # Try to find from agent's _agent_root attribute
    agent_root: Path | None = getattr(agent, '_agent_root', None)
    if agent_root and agent_root.exists():
        return agent_root

    # Fallback: search in current directory
    cwd = Path.cwd()
    for candidate in cwd.glob("*-agent"):
        if (candidate / "runtime.toml").exists():
            return candidate
    return None


def _handle_key_source_command(agent: Any) -> str:
    """Display the API key resolution chain with active source marked."""
    import os

    lines = ["[bold]API Key Resolution:[/bold]", ""]

    # CLI flag (session-only, checked during launch)
    lines.append("  - --api-key CLI flag (session-only)")

    # runtime.toml
    agent_dir = _find_agent_dir_for_config(agent)
    runtime_has_key = False
    if agent_dir:
        toml_path = agent_dir / "runtime.toml"
        if toml_path.exists():
            toml_cfg = tomllib.loads(toml_path.read_text())
            runtime_key = toml_cfg.get("llm", {}).get("api_key", "")
            if runtime_key:
                runtime_has_key = True

    if runtime_has_key:
        lines.append("  [bold green]✓ runtime.toml api_key[/]  ← active")
    else:
        lines.append("  - runtime.toml api_key (not set)")

    # Global config
    global_config = Path.home() / ".agenthatch" / "config.toml"
    global_has_key = False
    if global_config.exists():
        try:
            gcfg = tomllib.loads(global_config.read_text())
            provider = gcfg.get("providers", {}).get("default", "")
            provider_cfg = gcfg.get("providers", {}).get(provider, {})
            if provider_cfg.get("api_key"):
                global_has_key = True
        except Exception:
            pass

    if global_has_key and not runtime_has_key:
        lines.append("  [bold green]✓ ~/.agenthatch/config.toml[/]  ← active")
    elif global_has_key:
        lines.append("  - ~/.agenthatch/config.toml (shadowed by runtime.toml)")
    else:
        lines.append("  - ~/.agenthatch/config.toml (not configured)")

    # Environment variables
    env_vars = {
        "DEEPSEEK_API_KEY": "deepseek",
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
    }
    env_active = False
    for var in env_vars:
        if os.environ.get(var):
            env_active = True
            break

    if env_active:
        lines.append("  - Environment variable (configured)")
    else:
        lines.append("  - Environment variable (not set)")

    lines.append("")
    lines.append("[dim]Resolution order: CLI flag → runtime.toml → global config → env var[/dim]")

    return "\n".join(lines)


def _render_help(agent: Any) -> str:
    lines = [
        f"[bold bright_blue]{agent.identity.display_name}[/]"
        f" — {agent._raw_spec.get('intent', {}).get('summary', 'No summary')}",
        "",
        "[bold]Available commands:[/bold]",
        "  [bold cyan]/help[/]       Show this help",
        "  [bold cyan]/clear[/]      Clear conversation history",
        "  [bold cyan]/status[/]     Show provider/model info",
        "  [bold cyan]/config[/]     Configure API key, provider, or model",
        "  [bold cyan]/key-source[/] Show API key resolution chain",
        "  [bold cyan]/compact[/]    Trigger context compaction",
        "  [bold cyan]/quit[/]       Exit (or Ctrl+D)",
    ]
    return "\n".join(lines)


def _render_status(agent: Any) -> str:
    lines = [
        f"[bold]Provider:[/bold] {agent.llm.provider_name if agent.llm else 'N/A'}",
        f"[bold]Model:[/bold] {agent.llm.model if agent.llm else 'N/A'}",
        f"[bold]Agent:[/bold] {agent.identity.id} v{agent.identity.version}",
    ]
    if agent.llm:
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
        lines.append(
            f"[bold]Capabilities:[/bold] {', '.join(caps) if caps else 'none'}"
        )
    estimated_tokens = agent.ctx.estimate_input_tokens()
    lines.append(f"[bold]Est. input tokens:[/bold] {estimated_tokens}")
    return "\n".join(lines)
