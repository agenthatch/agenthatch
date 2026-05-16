"""Test agenthatch global options."""

from agenthatch import __version__


class TestVersion:
    """--version / -V tests."""

    def test_short_flag(self, runner, app):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "agenthatch" in result.output
        assert __version__ in result.output

    def test_long_flag(self, runner, app):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestHelp:
    """--help tests."""

    def test_help_no_errors(self, runner, app):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "hello" in result.output
        assert "doctor" in result.output
        assert "init" in result.output
        assert "--version" in result.output

    def test_no_args_shows_help(self, runner, app):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "hello" in result.output


class TestInvalidCommand:
    """Error handling tests."""

    def test_invalid_command_gives_error(self, runner, app):
        result = runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0
