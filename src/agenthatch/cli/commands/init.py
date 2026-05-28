"""agenthatch init — Interactive provider and API key configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.prompt import Confirm, Prompt

from agenthatch.cli import console
from agenthatch.config import CONFIG_DIR, CONFIG_FILE
from agenthatch.config.validators import validate_base_url, validate_provider_name
from agenthatch.providers import (
    BUILTIN_PROVIDER_NAMES,
    BUILTIN_PROVIDERS,
    ProviderInfo,
    list_builtin_providers,
)

_BUILTIN_CHOICES: dict[int, str] = {}


def _build_choice_map() -> dict[int, str]:
    """Lazily build the provider choice map."""
    if not _BUILTIN_CHOICES:
        for idx, info in enumerate(list_builtin_providers(), start=1):
            _BUILTIN_CHOICES[idx] = info.name
    return _BUILTIN_CHOICES


def _display_provider_menu() -> None:
    """Display the provider selection menu."""
    console.print()
    console.print("[bold]Select LLM Provider[/bold]")
    console.print()
    for idx, info in enumerate(list_builtin_providers(), start=1):
        env_hint = f" (env: {info.env_key})" if info.env_key else " (local, no key needed)"
        console.print(f"  [accent][{idx}][/accent] {info.name}{env_hint}")
    console.print("  [accent][5][/accent] Custom (OpenAI-compatible)")
    console.print()


def init_command(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config file"
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-n",
        help="Run in non-interactive mode (uses environment variables)",
    ),
    provider: str = typer.Option(
        "",
        "--provider",
        "-p",
        help="Provider to configure (skips provider selection menu)",
    ),
) -> None:
    """Interactive provider and API key configuration.

    Guides the user through:
    1. Provider selection (openai / anthropic / deepseek / ollama / custom)
    2. API Key input (or confirmation that an env var is set)
    3. Default model selection

    The resulting config is written to ~/.agenthatch/config.toml.

    Non-interactive mode reads from environment variables:
      AGENTHATCH_PROVIDER  — default provider name
      AGENTHATCH_API_KEY   — API key (generic fallback)
      OPENAI_API_KEY       — provider-specific key
      ANTHROPIC_API_KEY    — provider-specific key
      DEEPSEEK_API_KEY     — provider-specific key
      AGENTHATCH_MODEL     — default model (optional)
    """
    if CONFIG_FILE.exists() and not force:
        console.print(f"[yellow]Config file already exists: {CONFIG_FILE}[/yellow]")
        if not Confirm.ask("Overwrite?", default=False):
            raise typer.Exit(code=2)
        force = True

    if non_interactive:
        _init_non_interactive(force)
        return

    if provider:
        _init_with_provider(provider, force)
        return

    _init_interactive(force)


def _init_interactive(force: bool) -> None:
    """Full interactive setup flow."""
    console.print()
    console.print("[bold]agenthatch v0.2.0[/bold] — first-time setup")
    console.print()
    console.print("This will configure your LLM provider and API key.")
    console.print()

    # Step 1 — Provider selection
    _display_provider_menu()
    choices = _build_choice_map()
    choice = Prompt.ask(
        "Enter choice",
        choices=["1", "2", "3", "4", "5"],
        default="1",
    )
    choice_idx = int(choice)

    if choice_idx == 5:
        _configure_custom_provider(force)
        return

    selected_provider = choices[choice_idx]
    _configure_builtin_provider(selected_provider, force)


def _init_with_provider(provider: str, force: bool) -> None:
    """Init with a pre-selected provider (from --provider flag)."""
    if provider.startswith("custom."):
        console.print(f"[yellow]Custom provider '{provider}' requires interactive config.[/yellow]")
        custom_name = provider.removeprefix("custom.")
        _configure_custom_provider(force, preset_name=custom_name)
        return

    if provider not in BUILTIN_PROVIDER_NAMES:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        raise typer.Exit(code=2)

    _configure_builtin_provider(provider, force)


def _configure_builtin_provider(name: str, force: bool) -> None:
    """Configure a built-in provider."""
    info: ProviderInfo = BUILTIN_PROVIDERS[name]
    console.print()
    console.print(f"[bold]Configuring [accent]{name}[/accent][/bold]")

    # Step 2 — API Key
    api_key = _gather_api_key(info)
    model = _gather_model(info)

    # Step 3 — Write config
    _write_multi_provider_config(name, api_key, model, info.base_url, force)

    console.print()
    console.print("[green]Setup complete.[/green]")
    console.print(f"  Config:   [accent]{CONFIG_FILE}[/accent]")
    console.print(f"  Provider: [accent]{name}[/accent]")
    if api_key and info.env_key:
        console.print("  [warn]API key stored in config file.[/warn]")
        console.print(f"  [warn]Consider: export {info.env_key}=<key>[/warn]")
    elif not api_key and info.env_key:
        console.print(f"  Key:      [ok]via {info.env_key} env var[/ok]")
    console.print()
    console.print("Next: run [bold]agenthatch doctor[/bold] to verify connectivity.")

    # v0.3: Silent skillhouse initialization
    _init_skillhouse()


def _configure_custom_provider(
    force: bool, preset_name: str | None = None
) -> None:
    """Configure a custom OpenAI-compatible provider."""
    console.print()
    console.print("[bold]Configure Custom Provider[/bold]")
    console.print()
    console.print("Enter the details for your OpenAI-compatible endpoint.")
    console.print()

    # Step 1 — Provider name
    name = preset_name or Prompt.ask("Provider name", default="my-llm")
    assert name is not None
    validate_provider_name(name)

    # Step 2 — Base URL
    url = Prompt.ask(
        "Base URL",
        default="http://localhost:8000/v1",
    )
    validate_base_url(url)

    # Step 3 — API Key (optional for local)
    api_key = Prompt.ask(
        "API Key (leave empty if not needed)",
        password=True,
        default="",
    ).strip()

    # Step 4 — Default model
    model = Prompt.ask("Default model ID", default="").strip()

    # Step 5 — Env var name (optional)
    env_key = Prompt.ask(
        "Environment variable for API Key (optional)",
        default="",
    ).strip()

    _write_custom_provider_config(name, api_key or "", model, url, env_key, force)

    console.print()
    console.print(f"[green]Custom provider '{name}' configured.[/green]")
    console.print(f"  Use with: agenthatch --provider custom.{name}")
    console.print()
    console.print("Next: run [bold]agenthatch doctor[/bold] to verify connectivity.")

    # v0.3: Silent skillhouse initialization
    _init_skillhouse()


def _init_skillhouse() -> None:
    """v0.3: Silently initialize skillhouse.json AND scan for existing skills.

    Non-interactive, no user prompts.
    Three sources of search roots (see _resolve_search_roots):
      1. [skills].search_dirs from config
      2. Known AI tool directories (~/.claude/skills, ~/.codex/skills, etc.)
      3. Project-level .agents/skills/

    Each discovered skill is registered as a placeholder in skillhouse.json.
    """
    # Import shared search root resolver
    from agenthatch.cli.commands.hatch import _resolve_search_roots
    from agenthatch.house.index import SkillhouseIndex
    from agenthatch.skill.parser import _is_skill_md

    # Determine skillhouse.json path from config or default
    skillhouse_path = CONFIG_DIR / "skillhouse.json"

    # Load config to get search roots
    try:
        from agenthatch.config import Config
        config = Config.load()
    except Exception:
        config = {}

    search_roots = _resolve_search_roots(config)

    skillhouse_path.parent.mkdir(parents=True, exist_ok=True)
    idx = SkillhouseIndex(str(skillhouse_path))

    # Scan all roots for skills
    discovered = 0
    seen_dirs: set[Path] = set()

    for root in search_roots:
        if not root.is_dir():
            continue

        root_count = 0
        for skill_dir in _discover_all_skills(root, _is_skill_md):
            if skill_dir.resolve() in seen_dirs:
                continue
            seen_dirs.add(skill_dir.resolve())

            skill_id = skill_dir.name
            idx.register_placeholder(skill_id=skill_id, skill_dir=str(skill_dir))
            discovered += 1
            root_count += 1

        if root_count > 0:
            console.print(f"      Root: {root} ({root_count} found)")

    idx._save()

    if discovered > 0:
        console.print()
        console.print(f"  [ok]Discovered {discovered} skills:[/ok]")
        for d in sorted(seen_dirs, key=lambda p: p.name):
            console.print(f"      - {d.name:30s} →  {d}")
        console.print(f"  [ok]Index:[/ok] {skillhouse_path} ({discovered} skills)")
    else:
        console.print(f"  [ok]Skillhouse initialized at {skillhouse_path}[/ok]")
        console.print(
            "  [dim]No skills discovered. "
            "Run [bold]agenthatch hatch <path>[/bold] to register.[/dim]",
        )


def _gather_api_key(info: ProviderInfo) -> str:
    """Gather API key: check env, then prompt."""
    import os

    if info.env_key and os.environ.get(info.env_key, "").strip():
        console.print(f"  [ok]API key found via {info.env_key} environment variable[/ok]")
        return ""

    key = Prompt.ask(
        f"API Key for {info.name} (leave empty to use env var)",
        password=True,
        default="",
    )
    if key.strip():
        console.print(
            f"  [warn]Warning: API key will be stored in {CONFIG_FILE}[/warn]"
        )
        if info.env_key:
            console.print(
                f"  [warn]  Using {info.env_key} env var is recommended instead.[/warn]"
            )
    return key.strip()


def _gather_model(info: ProviderInfo) -> str:
    """Gather default model ID."""
    return Prompt.ask(
        f"Default model ID for {info.name}",
        default=info.default_model,
    ).strip()


def _write_multi_provider_config(
    default_provider: str,
    api_key: str,
    model: str,
    base_url: str,
    force: bool,
) -> None:
    """Write the full multi-provider config.toml.

    Uses string template (tech debt from v0.1) with variable substitution.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# agenthatch configuration file",
        "# Docs: https://github.com/agenthatch/agenthatch",
        "",
        "[core]",
        "verbose = false",
        "",
        "[providers]",
        f'default = "{default_provider}"',
        "",
    ]

    # Write built-in providers
    for pname in ("openai", "anthropic", "deepseek", "ollama"):
        info = BUILTIN_PROVIDERS[pname]
        lines.append(f"# {pname.capitalize()}")
        if info.env_key:
            lines.append(f"# API key: set via environment variable {info.env_key}")
        else:
            lines.append(f"# {pname.capitalize()} — local, no API key required")
        lines.append(f"[providers.{pname}]")

        provider_api_key = api_key if pname == default_provider else ""
        provider_model = model if pname == default_provider else info.default_model
        provider_base_url = base_url if (base_url and pname == default_provider) else info.base_url

        lines.append(f'api_key = "{provider_api_key}"')
        lines.append(f'base_url = "{provider_base_url}"')
        lines.append(f'default_model = "{provider_model}"')
        lines.append("")

    # Custom providers section header
    lines.append("# Custom OpenAI-compatible providers")
    lines.append("# Add your own under [providers.custom.<name>]")
    lines.append("# [providers.custom.my-llm]")
    lines.append('# api_key = ""')
    lines.append('# base_url = "http://localhost:8000/v1"')
    lines.append('# default_model = "mixtral-8x7b"')
    lines.append("")

    content = "\n".join(lines)
    CONFIG_FILE.write_text(content, encoding="utf-8")

    # Suppress lint: no direct access to config.toml
    _chmod_owner_only(CONFIG_FILE)


def _write_custom_provider_config(
    name: str,
    api_key: str,
    model: str,
    base_url: str,
    env_key: str,
    force: bool,
) -> None:
    """Write config with a custom provider as default."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# agenthatch configuration file",
        "# Docs: https://github.com/agenthatch/agenthatch",
        "",
        "[core]",
        "verbose = false",
        "",
        "[providers]",
        f'default = "custom.{name}"',
        "",
    ]

    for pname in ("openai", "anthropic", "deepseek", "ollama"):
        info = BUILTIN_PROVIDERS[pname]
        lines.append(f"[providers.{pname}]")
        lines.append('api_key = ""')
        lines.append(f'base_url = "{info.base_url}"')
        lines.append(f'default_model = "{info.default_model}"')
        lines.append("")

    env_comment = f" (env: {env_key})" if env_key else ""
    lines.append(f"# Custom provider: {name}{env_comment}")
    lines.append(f"[providers.custom.{name}]")
    lines.append(f'api_key = "{api_key}"')
    lines.append(f'base_url = "{base_url}"')
    lines.append(f'default_model = "{model}"')
    if env_key:
        lines.append(f'env_key = "{env_key}"')
    lines.append("")

    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
    _chmod_owner_only(CONFIG_FILE)


def _chmod_owner_only(filepath: Path) -> None:
    """Set file permissions to 0600 (owner read/write only).

    Prevents other users on the system from reading the config file,
    which may contain API keys.
    """
    import os
    import stat

    try:
        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # best-effort; may fail on some filesystems


def _init_non_interactive(force: bool) -> None:
    """Non-interactive init from environment variables.

    Reads:
    - AGENTHATCH_PROVIDER or AGENTHATCH_LLM_PROVIDER → default provider
    - Provider-specific key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    - AGENTHATCH_API_KEY → fallback key
    - AGENTHATCH_LLM_MODEL → default model (optional)
    - AGENTHATCH_LLM_BASE_URL → base URL (for custom providers)
    """
    import os

    provider = (
        os.environ.get("AGENTHATCH_PROVIDER")
        or os.environ.get("AGENTHATCH_LLM_PROVIDER")
        or "openai"
    )

    model = os.environ.get("AGENTHATCH_LLM_MODEL", "")
    base_url = os.environ.get("AGENTHATCH_LLM_BASE_URL", "")

    is_custom = provider.startswith("custom.")
    if is_custom:
        name = provider.removeprefix("custom.")
        _write_custom_provider_config(name, "", model, base_url, "", force)
        console.print(f"[green]Non-interactive setup: custom.{name}[/green]")
    else:
        if provider not in BUILTIN_PROVIDER_NAMES:
            console.print(f"[yellow]Unknown provider '{provider}', defaulting to openai[/yellow]")
            provider = "openai"
        _write_multi_provider_config(
            provider,
            api_key="",
            model=model or BUILTIN_PROVIDERS[provider].default_model,
            base_url=base_url,
            force=force,
        )
        console.print(f"[green]Non-interactive setup: {provider}[/green]")

    console.print(f"  Config: [accent]{CONFIG_FILE}[/accent]")
    console.print("  API key: [ok]read from environment variables[/ok]")

    # v0.3: Silent skillhouse initialization
    _init_skillhouse()


def _discover_all_skills(
    search_root: Path,
    _is_skill_md_fn: Any,
) -> list[Path]:
    """BFS scan a search root, discover ALL directories containing SKILL.md.

    Pattern: codex discover_skills_under_root():
      - BFS with deque
      - MAX_SCAN_DEPTH limit
      - MAX_DIRS_PER_ROOT upper bound
      - Skip dot-prefixed directories
      - Deduplication via canonical path set

    Returns:
        List of skill directories (ordered by discovery).
    """
    from collections import deque

    results: list[Path] = []
    visited: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque()

    try:
        search_root = search_root.resolve(strict=True)
    except (OSError, FileNotFoundError):
        return results
    if not search_root.is_dir():
        return results

    visited.add(search_root)
    queue.append((search_root, 0))
    dirs_visited = 0

    _MAX_INIT_SCAN_DEPTH = 5    # deeper than name scan (init is one-time)
    _MAX_INIT_DIRS = 1000       # more generous than name scan
    _EXCLUDED = frozenset(
        {".git", "__pycache__", "node_modules", ".venv", "venv",
         ".mypy_cache", ".pytest_cache", ".tox", ".eggs", "dist", "build"}
    )

    while queue:
        current_dir, depth = queue.popleft()
        dirs_visited += 1
        if dirs_visited > _MAX_INIT_DIRS:
            console.print(f"[yellow]Warning:[/yellow] Scan truncated at "
                          f"{_MAX_INIT_DIRS} directories in {search_root}")
            break

        try:
            entries = list(current_dir.iterdir())
        except (OSError, PermissionError):
            continue

        has_skill_md = False
        subdirs: list[Path] = []

        for entry in entries:
            if entry.is_symlink():
                continue

            if entry.is_file() and _is_skill_md_fn(entry.name):
                has_skill_md = True
            elif entry.is_dir() and depth < _MAX_INIT_SCAN_DEPTH:
                if not entry.name.startswith(".") and entry.name not in _EXCLUDED:
                    resolved = entry.resolve()
                    if resolved not in visited:
                        subdirs.append(resolved)

        if has_skill_md:
            results.append(current_dir)

        for subdir in subdirs:
            visited.add(subdir)
            queue.append((subdir, depth + 1))

    return results
