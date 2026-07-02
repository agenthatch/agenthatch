"""Shell completion helpers for the agenthatch CLI."""

from __future__ import annotations

import os
from pathlib import Path


def _complete_skill_name(incomplete: str) -> list[tuple[str, str]]:
    """Read registered skills from skillhouse.json and return matching candidates."""
    from agenthatch.config import Config
    from agenthatch.house.index import SkillhouseIndex

    try:
        config = Config.load()
    except Exception:
        return []

    skillhouse_cfg = config.get("skillhouse", {}) if "skillhouse" in config else {}
    sh_path = Path(
        skillhouse_cfg.get("path", ".agenthatch/skillhouse.json")
        if isinstance(skillhouse_cfg, dict)
        else ".agenthatch/skillhouse.json"
    )
    if not sh_path.is_absolute():
        sh_path = Path.cwd() / sh_path
    if not sh_path.exists():
        return []

    try:
        idx = SkillhouseIndex(str(sh_path))
        entries = idx.list_all()
    except Exception:
        return []

    candidates = []
    for e in entries:
        sid = e.get("id", "")
        if sid.startswith(incomplete):
            display = e.get("display_name", sid)
            candidates.append((sid, display))
    return sorted(candidates)


def _install_shell_completion() -> None:
    """Generate and write shell completion script during init.

    Silently skips on failure (never blocks init).
    """
    try:
        shell = os.environ.get("SHELL", "")
        shell_name = Path(shell).name if shell else ""

        if shell_name == "zsh":
            _install_for_zsh()
        elif shell_name == "bash":
            _install_for_bash()
        elif shell_name == "fish":
            _install_for_fish()
    except Exception:
        pass  # never block init


def _generate_completion_script(shell: str) -> str | None:
    """Generate completion script using the Click Python API."""
    from click.shell_completion import get_completion_class
    from typer.main import get_command

    # Lazy import to avoid circular dependency
    from agenthatch.cli.main import app

    try:
        click_cmd = get_command(app)
        cls = get_completion_class(shell)
        if cls is None:
            return None
        completer = cls(
            cli=click_cmd,
            ctx_args={},
            prog_name="agenthatch",
            complete_var="_AGENTHATCH_COMPLETE",
        )
        return str(completer.source())
    except Exception:
        return None


def _install_for_zsh() -> None:
    target_dir = Path.home() / ".zsh" / "completions"
    target_dir.mkdir(parents=True, exist_ok=True)
    script = _generate_completion_script("zsh")
    if script is None:
        return
    (target_dir / "_agenthatch").write_text(script)
    _print_ok("zsh")

    # oh-my-zsh includes ~/.zsh/completions in fpath by default.
    # For bare zsh, check whether the user needs to add it manually.
    if not _zsh_has_completions_in_fpath(str(target_dir)):
        _print_zsh_fpath_hint()


def _install_for_bash() -> None:
    # macOS default bash 3.2 does not support completion
    bash_version = os.environ.get("BASH_VERSION", "")
    if bash_version:
        parts = bash_version.split(".")
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        if major < 4 or (major == 4 and minor < 4):
            _print_bash_upgrade_hint()
            return

    target_dir = Path.home() / ".bash_completion.d"
    target_dir.mkdir(parents=True, exist_ok=True)
    script = _generate_completion_script("bash")
    if script is None:
        return
    (target_dir / "agenthatch").write_text(script)
    _print_ok("bash")


def _install_for_fish() -> None:
    target_dir = Path.home() / ".config" / "fish" / "completions"
    target_dir.mkdir(parents=True, exist_ok=True)
    script = _generate_completion_script("fish")
    if script is None:
        return
    (target_dir / "agenthatch.fish").write_text(script)
    _print_ok("fish")


def _zsh_has_completions_in_fpath(target_dir: str) -> bool:
    """Check whether ~/.zsh/completions is already in $fpath."""
    import subprocess

    try:
        result = subprocess.run(
            ["zsh", "-c", f'[[ ${{fpath[(r){target_dir}]}} == "{target_dir}" ]]'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _print_zsh_fpath_hint() -> None:
    from agenthatch.cli import console

    console.print(
        "[dim]Add this line before [bold]compinit[/bold] "
        "in your .zshrc to enable completions:[/dim]"
    )
    console.print("  [dim]fpath=(~/.zsh/completions $fpath)[/dim]")


def _print_ok(shell: str) -> None:
    from agenthatch.cli import console
    console.print(f"[ok]Shell completion installed for {shell}[/ok]")


def _print_bash_upgrade_hint() -> None:
    from agenthatch.cli import console
    console.print(
        "[dim]Bash version < 4.4 detected — shell completion not supported. "
        "Consider using zsh or upgrading bash.[/dim]"
    )
