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
        "agent": "bold bright_blue",
        "user": "bold green",
        "assistant": "bright_white",
        "tool": "bold yellow",
        "tool_result": "dim cyan",
        "error": "bold red",
        "streaming": "italic bright_white",
        "divider": "dim",
        "capability": "bold magenta",
        "brick": "bold cyan",
    }
)
console = Console(theme=_theme, highlight=False)
