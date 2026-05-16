"""agenthatch init — 生成默认配置文件。"""

import typer
from rich.prompt import Confirm

from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE, Config
from agenthatch.exceptions import ConfigError


def init_command(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config file"
    ),
) -> None:
    """Create a default agenthatch configuration file.

    This creates ~/.agenthatch/config.toml with default settings.
    API keys must be set via environment variables, not in the config file.
    """
    if CONFIG_FILE.exists() and not force:
        console.print(f"[yellow]Config file already exists: {CONFIG_FILE}[/yellow]")
        if not Confirm.ask("Overwrite?", default=False):
            raise typer.Exit(code=2)
        force = True

    try:
        path = Config.create_default(force=force)
        console.print(f"\n[green]Config created: {path}[/green]")
        console.print()
        console.print("Next steps:")
        console.print("  1. Set your LLM API key as an environment variable:")
        console.print("     [bold]export OPENAI_API_KEY=sk-xxxx[/bold]")
        console.print("     [bold]export ANTHROPIC_API_KEY=sk-ant-xxxx[/bold]")
        console.print("  2. Run [bold]agenthatch doctor[/bold] to verify your setup.")
    except ConfigError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None
