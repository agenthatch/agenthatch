"""Shared fixtures for agenthatch tests."""

from __future__ import annotations

import pytest
from typer import rich_utils
from typer.testing import CliRunner

from agenthatch.cli.main import app as _app


@pytest.fixture(autouse=True, scope="session")
def _fix_rich_output():
    """Fix Rich output width for reproducible test results.

    Without this, Rich auto-detects terminal width and wraps text
    differently depending on the environment (CI vs local).
    Reference: fastapi-cli tests/conftest.py
    """
    rich_utils.MAX_WIDTH = 3000
    rich_utils.FORCE_TERMINAL = False


@pytest.fixture(autouse=True)
def _fix_prompt_password(monkeypatch):
    """Monkeypatch Prompt.ask to not use getpass in tests.

    Rich Prompt.ask(password=True) uses getpass.getpass() which reads
    from /dev/tty directly, bypassing CliRunner's stdin simulation.
    Stripping password=True lets CliRunner's input= parameter work.
    """
    import rich.prompt

    _original_ask = rich.prompt.Prompt.ask

    @staticmethod
    def _ask(prompt="", *, password=False, **kwargs):  # noqa: B008
        return _original_ask(prompt, password=False, **kwargs)

    monkeypatch.setattr(rich.prompt.Prompt, "ask", _ask)


@pytest.fixture
def app():
    """Return the agenthatch Typer app."""
    return _app


@pytest.fixture
def runner():
    """Create a CliRunner for use with agenthatch app."""
    return CliRunner()


@pytest.fixture
def invoke(runner, app):
    """Convenience fixture: invoke directly."""

    def _invoke(*args):
        return runner.invoke(app, list(args))

    return _invoke


@pytest.fixture
def tmp_agenthatch_home(tmp_path, monkeypatch):
    """Override agenthatch config directory with a temporary one.

    Patches all modules that import CONFIG_FILE to ensure
    consistency across the config loading subsystem.
    """
    temp_config = tmp_path / "config.toml"
    monkeypatch.setattr("agenthatch.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("agenthatch.config.CONFIG_FILE", temp_config)
    monkeypatch.setattr("agenthatch.cli.commands.init.CONFIG_FILE", temp_config)
    monkeypatch.setattr("agenthatch.cli.commands.doctor.CONFIG_FILE", temp_config)
    monkeypatch.setattr("agenthatch.cli.commands.hello.CONFIG_FILE", temp_config)
    monkeypatch.setattr("agenthatch.providers.CONFIG_FILE", temp_config)
    return tmp_path


@pytest.fixture
def mock_httpx_success(monkeypatch):
    """Mock httpx to return success for API key verification."""

    class _MockResponse:
        status_code = 200
        is_success = True

    def _mock_get(*args, **kwargs):
        return _MockResponse()

    monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_get)


@pytest.fixture
def mock_httpx_unauthorized(monkeypatch):
    """Mock httpx to return 401 for API key verification."""

    class _MockResponse:
        status_code = 401
        is_success = False

    def _mock_get(*args, **kwargs):
        return _MockResponse()

    monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_get)
