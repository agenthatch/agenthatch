"""agenthatch CLI entry point.

Defines the Typer app, global options, and subcommand registration.
"""
import logging
from pathlib import Path
from typing import Any

import typer

from agenthatch import __version__
from agenthatch.cli import console
from agenthatch.cli.commands.doctor import doctor_command
from agenthatch.cli.commands.hatch import hatch_command
from agenthatch.cli.commands.hello import hello_command
from agenthatch.cli.commands.init import init_command
from agenthatch.cli.commands.search import search_command
from agenthatch.cli.commands.skills import skills_command
from agenthatch.exceptions import AgentHatchError

app = typer.Typer(
    name="agenthatch",
    help="Turn any SKILL.md into a runnable AI Agent.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _on_version_callback(value: bool) -> None:
    """Handle --version option callback.

    When --version is passed, prints the version and exits.
    No resilient_parsing check needed — is_eager=True ensures Typer
    handles shell completion internally without invoking this callback.
    """
    if not value:
        return
    console.print(f"[version]agenthatch[/version] {__version__}")
    raise typer.Exit()


@app.callback(invoke_without_command=True)
def _global_callback(
    verbose: int = typer.Option(  # noqa: B008
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v: INFO, -vv: DEBUG)",
        envvar="AGENTHATCH_VERBOSE",
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress all non-error output",
        envvar="AGENTHATCH_QUIET",
    ),
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Path to config file (default: ~/.agenthatch/config.toml)",
        envvar="AGENTHATCH_CONFIG",
    ),
    trace: bool = typer.Option(  # noqa: B008
        False,
        "--trace",
        help="Show Harness reasoning traces (for hatch command)",
        envvar="AGENTHATCH_TRACE",
    ),
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        help="Show version and exit",
        callback=_on_version_callback,
        is_eager=True,
    ),
) -> None:
    """agenthatch: AI Agent incubator. Turn any SKILL.md into a runnable AI Agent."""
    _configure_logging(verbose, quiet)


def _handle_agenthatch_error(e: AgentHatchError) -> None:
    """Unified AgentHatchError handler — prints error message and exits with exit_code."""
    console.print(f"[red]Error: {e}[/red]")
    raise typer.Exit(code=e.exit_code)


def _configure_logging(verbose: int, quiet: bool) -> None:
    """Configure logging levels based on flags.

    Uses a named logger + RichHandler instead of logging.basicConfig,
    so only agenthatch's own logs are affected, not dependencies.

    verbose=0: WARNING (default)
    verbose=1: INFO
    verbose>=2: DEBUG
    quiet=True: ERROR only (overrides verbose)
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    _logger = logging.getLogger("agenthatch")
    _logger.setLevel(level)
    _logger.propagate = False

    if not _logger.handlers:
        from rich.logging import RichHandler

        handler = RichHandler(
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            console=console,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)


app.command(name="hello")(hello_command)
app.command(name="doctor")(doctor_command)
app.command(name="init")(init_command)
app.command(name="hatch")(hatch_command)
app.command(name="search")(search_command)
app.command(name="skills")(skills_command)


def main() -> Any:
    """CLI entry point function.

    Single entry point, convenient for extending global exception handling,
    signal handling, etc. Both pyproject.toml [project.scripts] and __main__.py
    point to this function.
    """
    try:
        return app()
    except AgentHatchError as e:
        _handle_agenthatch_error(e)
