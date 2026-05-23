"""agenthatch CLI layer.

Subcommands are organized by function under the commands/ directory.
"""
from rich.console import Console
from rich.theme import Theme

_theme = Theme(
    {
        "version": "bold magenta",
        "accent": "bold cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "muted": "dim white",
    }
)
console = Console(theme=_theme, highlight=False)
