"""agenthatch: Turn any SKILL.md into a runnable AI Agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agenthatch")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
