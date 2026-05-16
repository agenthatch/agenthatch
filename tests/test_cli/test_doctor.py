"""Test agenthatch doctor command."""


class TestDoctor:
    """agenthatch doctor tests."""

    def test_basic_run(self, runner, app):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in (0, 1)
        assert "Python" in result.output
