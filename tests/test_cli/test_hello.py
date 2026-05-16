"""Test agenthatch hello command."""

from agenthatch import __version__


class TestHello:
    """agenthatch hello tests."""

    def test_default_greeting(self, runner, app):
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 0
        assert "Hello, World" in result.output
        assert __version__ in result.output
        assert "python" in result.output.lower()

    def test_custom_name(self, runner, app):
        result = runner.invoke(app, ["hello", "Agent"])
        assert result.exit_code == 0
        assert "Hello, Agent" in result.output

    def test_positional_name(self, runner, app):
        result = runner.invoke(app, ["hello", "Developer"])
        assert result.exit_code == 0
        assert "Hello, Developer" in result.output
