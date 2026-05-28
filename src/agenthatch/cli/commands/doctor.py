"""agenthatch doctor — Environment health check."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import typer

from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE
from agenthatch.providers import (
    ProviderInfo,
    get_default_provider,
    get_provider,
    resolve_api_key,
    verify_api_key,
)


@dataclass
class _Check:
    passed: bool
    message: str
    fix: str = ""


def doctor_command() -> None:
    """Run environment health checks.

    v0.2 adds API key connectivity check on top of v0.1 checks.
    Exits with code 1 if any check fails.
    """
    console.print()
    console.print("[bold]agenthatch Health Check[/bold]")
    console.print()

    checks = [
        _check_python_version,
        _check_dependencies,
        _check_config_file,
        _check_api_key,
        _check_skillhouse,
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
        console.print("[bold green]All checks passed. You are ready to build.[/bold green]")
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
    for pkg in ("typer", "rich", "httpx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return _Check(passed=True, message="Core dependencies: typer, rich, httpx")
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


def _check_api_key() -> _Check:
    """Check API key is configured and can reach the provider.

    Steps:
    1. Read default provider from config
    2. Resolve API key using priority chain
    3. Verify connectivity with a lightweight HTTP request

    This check is skipped if no config file exists (handled by _check_config_file).
    """
    if not CONFIG_FILE.exists():
        return _Check(
            passed=True,
            message="API key — skipping (no config file yet)",
        )

    try:
        provider_name = get_default_provider()
        info: ProviderInfo = get_provider(provider_name)
    except Exception as e:
        return _Check(
            passed=False,
            message=f"Provider resolution failed: {e}",
            fix="agenthatch init",
        )

    if not info.env_key and info.kind == "builtin" and info.name == "ollama":
        return _Check(
            passed=True,
            message=f"Provider: {provider_name} (local, no key needed)",
        )

    key = resolve_api_key(provider_name, prompt=False)
    if not key:
        return _Check(
            passed=False,
            message=f"API key not configured for '{provider_name}'",
            fix=f"export {info.env_key}=<key>  or  agenthatch init",
        )

    ok, detail = verify_api_key(provider_name, key, info.base_url)
    if ok:
        return _Check(
            passed=True,
            message=f"Provider: {provider_name} — {detail}",
        )
    return _Check(
        passed=False,
        message=f"Provider: {provider_name} — {detail}",
        fix=f"Check your {info.env_key} environment variable or API key",
    )


def _check_skillhouse() -> _Check:
    """Check skillhouse.json integrity and accessibility.

    v0.3 health check: verifies the skillhouse.json file exists
    and is valid JSON with the expected structure.
    """
    import json
    from pathlib import Path

    from agenthatch.config import Config

    try:
        config = Config.load()
    except Exception:
        config = {}

    skillhouse_cfg = config.get("skillhouse", {}) if "skillhouse" in config else {}
    skillhouse_path = skillhouse_cfg.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_cfg, dict) else ".agenthatch/skillhouse.json"

    sh_path = Path(skillhouse_path)
    if not sh_path.is_absolute():
        sh_path = Path.cwd() / sh_path

    if not sh_path.exists():
        return _Check(
            passed=True,
            message="skillhouse.json — not yet created (run hatch to populate)",
        )

    try:
        data = json.loads(sh_path.read_text(encoding="utf-8"))
        version = data.get("version", "unknown")
        entries = len(data.get("entries", {}))
        return _Check(
            passed=True,
            message=f"skillhouse.json — v{version}, {entries} skills indexed",
        )
    except json.JSONDecodeError as e:
        return _Check(
            passed=False,
            message=f"skillhouse.json — invalid JSON: {e}",
            fix="Delete .agenthatch/skillhouse.json and re-hatch skills",
        )
    except OSError as e:
        return _Check(
            passed=False,
            message=f"skillhouse.json — read error: {e}",
            fix="Check file permissions",
        )
