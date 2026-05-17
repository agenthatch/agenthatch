"""agenthatch hello — Verify installation."""

import sys

import typer

from agenthatch import __version__
from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE


def hello_command(name: str = typer.Argument("World", help="Name to greet")) -> None:
    """Verify agenthatch is installed correctly.

    This is the first command every user should run after installation.
    It confirms the package was installed and shows basic environment info.
    """
    console.print(f"[ok]Hello, {name}![/ok]")
    console.print()
    console.print(f"  version : [accent]{__version__}[/accent]")
    console.print(f"  python  : [accent]{sys.executable}[/accent]")
    console.print(f"  config  : [accent]{CONFIG_FILE}[/accent]")
    console.print()
    console.print("Next: run [bold]agenthatch init[/bold] to create your config.")
