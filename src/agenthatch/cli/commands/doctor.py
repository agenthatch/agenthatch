"""agenthatch doctor — 环境健康检查。"""

import sys
from dataclasses import dataclass

import typer

from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE


@dataclass
class _Check:
    passed: bool
    message: str
    fix: str = ""


def doctor_command() -> None:
    """Run environment health checks.

    Checks Python version, core dependencies, and config file status.
    Exits with code 1 if any check fails.
    """
    console.print("\n[bold]agenthatch Health Check[/bold]\n")

    checks = [
        _check_python_version,
        _check_dependencies,
        _check_config_file,
    ]

    all_passed = True
    for check_fn in checks:
        result = check_fn()
        if result.passed:
            console.print(f"  [green]OK[/green]  {result.message}")
        else:
            all_passed = False
            console.print(f"  [red]FAIL[/red] {result.message}")
            if result.fix:
                console.print(f"        [yellow]Fix: {result.fix}[/yellow]")

    console.print()
    if all_passed:
        console.print("[bold green]All checks passed. You are ready to go.[/bold green]")
    else:
        console.print("[bold red]Some checks failed. See above for fixes.[/bold red]")
        raise typer.Exit(code=1)


def _check_python_version() -> _Check:
    """Check Python >= 3.11."""
    min_version = (3, 11)
    current = sys.version_info
    if current >= min_version:
        return _Check(
            passed=True,
            message=f"Python {current.major}.{current.minor}.{current.micro}",
        )
    min_str = ".".join(map(str, min_version))
    return _Check(
        passed=False,
        message=f"Python {current.major}.{current.minor} (need >= {min_str})",
        fix=f"Install Python {min_str}+",
    )


def _check_dependencies() -> _Check:
    """Check core dependencies are installed."""
    missing: list[str] = []
    for pkg in ("typer", "rich"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return _Check(passed=True, message="Core dependencies: typer, rich")
    return _Check(
        passed=False,
        message=f"Missing packages: {', '.join(missing)}",
        fix="pip install agenthatch",
    )


def _check_config_file() -> _Check:
    """Check config file exists."""
    if CONFIG_FILE.exists():
        return _Check(passed=True, message=f"Config file: {CONFIG_FILE}")
    return _Check(
        passed=False,
        message="No config file found",
        fix="agenthatch init",
    )
