"""Global option definitions.

Reusable Typer option callbacks shared across subcommands.
Starting v0.2, --provider is shared by init and future build/run commands.
"""

from __future__ import annotations

from typing import Any

import typer

__all__ = ["provider_option"]


def _provider_autocompletion(incomplete: str) -> list[tuple[str, str]]:
    """Shell completion callback for --provider option.

    Returns matching built-in provider names as completion candidates.
    Custom providers are discovered lazily from config.toml on invocation
    and appended to the list.

    Reference: Typer shell completion uses (value, help) tuples.
    """
    from agenthatch.providers import BUILTIN_PROVIDER_NAMES

    candidates: list[tuple[str, str]] = []
    for name in sorted(BUILTIN_PROVIDER_NAMES):
        if name.startswith(incomplete):
            candidates.append((name, f"Built-in provider: {name}"))

    try:
        from agenthatch.providers import list_custom_providers

        for info in list_custom_providers():
            if info.name.startswith(incomplete):
                candidates.append((info.name, f"Custom provider: {info.name}"))
    except Exception:
        pass

    return candidates


def provider_option(
    default: str | None = None,
    help_text: str = "LLM provider to use",
) -> Any:
    """Factory for a reusable --provider / -p Typer Option.

    This function returns an Option instance that can be attached to any
    Typer command parameter. It differs from a simple typer.Option()
    in that it provides shell completion for provider names.

    Usage:
        provider: str = provider_option()
    """
    return typer.Option(
        default,
        "--provider",
        "-p",
        help=help_text,
        autocompletion=_provider_autocompletion,
        show_default=False,
    )
