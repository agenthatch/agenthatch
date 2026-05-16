"""agenthatch CLI 层。

子命令按功能拆分到 commands/ 目录。
"""
from rich.console import Console
from rich.theme import Theme

_theme = Theme(
    {
        "version": "bold magenta",
        "accent": "bold cyan",
        "ok": "bold green",
    }
)
console = Console(theme=_theme, highlight=False)
