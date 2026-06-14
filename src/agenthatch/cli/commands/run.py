"""agenthatch run — launch an independent Agent in interactive TUI mode.

v0.6: Rewritten as Agent-direct launcher.  Finds the hatched Agent
directory, confirms dependencies, and launches in-process with Rich Live TUI.

Old config-driven SkillAgent.from_ahspec() path has been removed.
"""

from __future__ import annotations

import sys
import textwrap
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
from agenthatch_core.loop.token_counter import ThinkingDelta
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style as PTStyle
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agenthatch.cli import console
from agenthatch.cli.commands._completion import _complete_skill_name
from agenthatch.cli.interrupt import EarlyInputReader

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
        provider = gcfg.get("agenthatch", {}).get("default", "")
        if provider.startswith("custom."):
            custom_key = provider.removeprefix("custom.")
            provider_cfg = gcfg.get("providers", {}).get("custom", {}).get(custom_key, {})
        else:
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
            f"{agent_pkg}.agent", agent_module_path
        )
        if spec is None or spec.loader is None:
            console.print(
                f"[red]Error:[/red] Cannot load agent module from {agent_module_path}"
            )
            raise typer.Exit(code=1)

        module = _importlib_util.module_from_spec(spec)
        # Set __package__ so relative imports (from .tools import ...) work
        module.__package__ = agent_pkg
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
            f"({agent.identity.id})",
            title="Agent", border_style="bright_blue"
        )
    )
    if key_source:
        console.print(f"[dim]API key: {key_source}[/dim]")
    console.print(
        "[dim]Type /help for commands. "
        "Ctrl+C to interrupt agent, /quit or Ctrl+D to exit[/dim]"
    )
    console.print()

    # v0.9: Shared prompt_toolkit output with CPR pre-disabled.
    # After EarlyInputReader modifies termios during streaming, prompt_toolkit
    # may fail the CPR (Cursor Position Request) probe and print a noisy
    # "WARNING: your terminal doesn't support cursor position requests (CPR)."
    # Pre-marking CPR as not-supported skips the probe entirely.
    _pt_output = create_output()
    _pt_output.ask_for_cpr = lambda: None  # no-op: skip CPR probe + warning

    try:
        early_input: str | None = None
        while True:
            # v0.9: If user typed during previous streaming, use that input
            if early_input:
                user_input = early_input
                early_input = None
            else:
                # prompt_toolkit replaces Rich.Prompt for CJK input correctness
                # (macOS libedit bug: backspace drifts on multi-byte characters)
                try:
                    user_input = pt_prompt(
                        [("class:prompt", "You: ")],
                        style=PT_STYLE,
                        output=_pt_output,
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
                response_text, early_input = _stream_response(agent, user_input)
            except Exception as e:
                response_text = f"Agent error: {e}"
                early_input = None
                console.print(f"[error]{response_text}[/error]")
                continue

            if early_input:
                console.print(
                    f"[dim]Captured:[/dim] [bold green]{early_input}[/bold green]"
                )

            console.print(Markdown(response_text))
            console.print()

    except (KeyboardInterrupt, SystemExit, EOFError):
        console.print()


def _stream_response(agent: Any, user_input: str) -> tuple[str, str]:
    """v0.9: Streaming with early-input capture and interrupt support.

    Returns (response_text, early_input) where early_input is any text
    the user typed during streaming (may be empty).

    Based on Claude Code's earlyInput.ts pattern:
    - Background thread captures stdin while agent is streaming
    - Ctrl+C interrupts the current turn
    - Text typed during execution is buffered and returned
    """
    reader = EarlyInputReader(agent)
    reader.start()
    try:
        response = _stream_response_inner(agent, user_input, reader)
        early = reader.consume()
        return response, early
    finally:
        reader.stop()


def _stream_response_inner(agent: Any, user_input: str, reader: EarlyInputReader) -> str:
    """Core streaming logic with interrupt-awareness."""
    full_text: list[str] = []
    reasoning_chars: list[str] = []
    tool_status: dict[str, str] = {}
    start_time = time.monotonic()
    show_thinking = getattr(agent, "_show_reasoning", True)

    def render_panel() -> str:
        lines = [f"[bold bright_blue]{agent.identity.display_name}[/]"]
        for name, status in tool_status.items():
            lines.append(f"  [cyan]{name}[/] {status}")
        if reader.interrupted:
            lines.append("  [bold yellow]⏸ Interrupted — waiting for your input...[/]")
        if full_text:
            lines.append("".join(full_text)[-500:])
        if not full_text and not tool_status:
            lines.append("[dim]Thinking...[/dim]")
        if show_thinking and reasoning_chars and not full_text:
            joined = "".join(reasoning_chars)
            # v0.7.15: textwrap instead of [:120] hard truncation.
            # DeepSeek reasoning often emits one continuous line —
            # textwrap breaks it into displayable chunks without cutting mid-content.
            real_lines = [L for L in joined.split("\n") if L.strip()]
            display_lines: list[str] = []
            for line in real_lines[-3:]:
                if len(line) > 100:
                    display_lines.extend(
                        textwrap.wrap(line, width=100, break_long_words=False,
                                      break_on_hyphens=False)
                    )
                else:
                    display_lines.append(line)
            for line in display_lines[-6:]:
                stripped = line.strip()
                if stripped:
                    lines.append(f"[dim italic]  {stripped}[/dim italic]")
        elapsed = time.monotonic() - start_time
        stats_parts = [f"[dim]⏱ {elapsed:.1f}s[/dim]"]
        if hasattr(agent, "token_counter"):
            snap = agent.token_counter.snapshot()
            total = snap.get("total_tokens", 0)
            if total > 0:
                prompt_tok = snap.get("prompt_tokens", 0)
                completion_tok = snap.get("completion_tokens", 0)
                call_count = snap.get("call_count", 0)
                stats_parts.append(
                    f"[dim]{total:,} tokens (in: {prompt_tok:,}, "
                    f"out: {completion_tok:,})[/dim]"
                )
                if call_count > 0:
                    stats_parts.append(f"[dim]{call_count} calls[/dim]")
        lines.append(" · ".join(stats_parts))
        return "\n".join(lines)

    # ── Phase 1: Live streaming (single panel) ──
    with Live(Panel(render_panel(), title="Agent"), refresh_per_second=8) as live:
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
                    es = f"{event.elapsed:.1f}s" if event.elapsed else ""
                    preview = (event.result_preview or "")[:80]
                    tool_status[event.tool_name] = (
                        f"[bold green]done[/] ({es}) [dim]→ {preview}[/]"
                    )
            elif isinstance(event, ThinkingDelta):
                reasoning_chars.append(event.content)
            elif isinstance(event, str):
                full_text.append(event)

            live.update(Panel(render_panel(), title="Agent"))

        done_body = _build_done_panel_content(agent, start_time)
        live.update(Panel(done_body, title="Agent", border_style="green"))

    # v0.7.14: No redundant Session footer.  Done panel is the sole sink.

    return final_text or "".join(full_text) or "(no response)"


def _build_done_panel_content(agent: Any, start_time: float) -> str:
    """v0.7.13: Build Done panel with observability data (tokens + time).

    This is the primary observability display — Session footer shows status summary.
    """
    elapsed = time.monotonic() - start_time
    lines = ["[bold green]✓ Done[/bold green]"]

    if hasattr(agent, "token_counter"):
        snap = agent.token_counter.snapshot()
        total = snap.get("total_tokens", 0)
        if total > 0:
            prompt_tok = snap.get("prompt_tokens", 0)
            completion_tok = snap.get("completion_tokens", 0)
            call_count = snap.get("call_count", 0)
            lines.append(
                f"[dim]Tokens: {total:,} (in: {prompt_tok:,}, out: {completion_tok:,})"
                f" [{call_count} calls]  ⏱ {elapsed:.1f}s[/dim]"
            )
        else:
            lines.append(f"[dim]⏱ {elapsed:.1f}s[/dim]")

    return "\n".join(lines)


def _build_observability_footer(
    agent: Any, start_time: float, reasoning_count: int = 0
) -> Panel | str:
    """v0.7.13: Build Session footer with status summary (not duplicate data).

    Shows session state info instead of repeating Done panel observability.
    """
    elapsed = time.monotonic() - start_time
    parts: list[str] = []

    # Status summary
    if reasoning_count > 0:
        parts.append(f"Thinking complete ({reasoning_count} chunks)")
    parts.append(f"⏱ {elapsed:.1f}s")

    # Turn count
    if hasattr(agent, "ctx"):
        turn_count = getattr(agent.ctx, "_turn_count", 0)
        if turn_count:
            parts.append(f"Turn #{turn_count}")

    return Panel(
        " │ ".join(f"[dim]{p}[/dim]" for p in parts),
        title="Session",
        border_style="dim green",
        padding=(0, 1),
    )


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
    elif cmd == "/thinking":
        # v0.7.11: Toggle reasoning/thinking visibility for debugging
        current = getattr(agent, "_show_reasoning", True)
        agent._show_reasoning = not current
        state = "ON" if agent._show_reasoning else "OFF"
        return f"[ok]Reasoning display: {state}[/ok]"
    elif cmd.startswith("/attach "):
        # v0.7.12: Attach a file to the conversation via FileProcessor
        filepath = user_input[8:].strip()
        if hasattr(agent, "attach_file"):
            return agent.attach_file(filepath)  # type: ignore[no-any-return]
        return "[warn]File attachment not supported by this agent.[/warn]"
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
            provider = gcfg.get("agenthatch", {}).get("default", "")
            if provider.startswith("custom."):
                custom_key = provider.removeprefix("custom.")
                provider_cfg = gcfg.get("providers", {}).get("custom", {}).get(custom_key, {})
            else:
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
        "  [bold cyan]/thinking[/]   Toggle reasoning display for debugging",
        "  [bold cyan]/attach[/] <path>  Attach a file to the conversation",
        "  [bold cyan]/quit[/]       Exit (or Ctrl+D)",
        "",
        "[bold]Keyboard shortcuts:[/bold]",
        "  [bold]Ctrl+C[/]   Interrupt agent mid-execution and input your response",
        "  [bold]Ctrl+D[/]   Exit (same as /quit)",
    ]
    return "\n".join(lines)


def _render_status(agent: Any) -> str:
    lines = [
        f"[bold]Provider:[/bold] {agent.llm.provider_name if agent.llm else 'N/A'}",
        f"[bold]Model:[/bold] {agent.llm.model if agent.llm else 'N/A'}",
        f"[bold]Agent:[/bold] {agent.identity.id}",
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
