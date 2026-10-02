"""v1.0.23: postgen verdict stays in the log channel — never the console.

Project philosophy: the default hatch output is clean and success-oriented;
diagnostics are opt-in via -v (log stream) or --report (structured view).
The Phase 3.5 header is process narration and stays on the console; the
verdict line is a diagnostic and must not.
"""
import contextlib
import logging
from types import SimpleNamespace

import pytest

from agenthatch.cli.commands import hatch as hatch_mod


class _RecordingConsole:
    def __init__(self):
        self.printed = []

    def print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))

    def status(self, *args, **kwargs):
        return contextlib.nullcontext()


@pytest.fixture()
def fake_console(monkeypatch):
    console = _RecordingConsole()
    monkeypatch.setattr(hatch_mod, "console", console)
    return console


@pytest.fixture()
def fake_review(monkeypatch):
    """Replace iterate_until_gate with a canned report (no real work)."""
    holder = {"verdict": "WARN"}

    def _gate(**kwargs):
        return SimpleNamespace(
            verdict=holder["verdict"],
            iterations=2,
            tools_with_issues=2,
            tools_total=4,
        )

    monkeypatch.setattr(
        "agenthatch.skill.postgen_review.iterate_until_gate", _gate
    )
    return holder


def _run(quiet: bool = False):
    return hatch_mod._run_postgen_review(
        agent_output_dir=None,
        ahs_spec=None,
        skill_dir=None,
        ai_chat_fn=None,
        quiet=quiet,
    )


def _verdict_records(caplog):
    return [r for r in caplog.records if "postgen review" in r.getMessage()]


class TestVerdictChannel:
    def test_warn_verdict_logged_not_printed(self, fake_console, fake_review, caplog):
        """A WARN verdict must reach the log channel, never the console."""
        fake_review["verdict"] = "WARN"
        with caplog.at_level(logging.INFO, logger="agenthatch"):
            report = _run()

        assert report is not None and report.verdict == "WARN"
        assert not any(
            "postgen review" in text for text in fake_console.printed
        ), "verdict must not be printed to the console"
        records = _verdict_records(caplog)
        assert records, "verdict must be emitted through the logger"
        assert "postgen review: WARN (2 rounds, 2/4 tools with issues)" == records[0].getMessage()
        assert records[0].levelno == logging.WARNING

    def test_ready_verdict_logged_at_info(self, fake_console, fake_review, caplog):
        """A READY verdict is logged at INFO level, still not printed."""
        fake_review["verdict"] = "READY"
        with caplog.at_level(logging.INFO, logger="agenthatch"):
            report = _run(quiet=False)

        assert report is not None and report.verdict == "READY"
        assert not any(
            "postgen review" in text for text in fake_console.printed
        )
        records = _verdict_records(caplog)
        assert records
        assert records[0].getMessage().startswith("postgen review: READY")
        assert records[0].levelno == logging.INFO

    def test_quiet_json_mode_still_logs(self, fake_console, fake_review, caplog):
        """quiet (JSON mode) suppresses console narration, not the log channel.

        In --report --json runs the default verbose=0 (ERROR level) keeps
        stdout pure JSON; when the user adds -v the verdict is available
        from the logger regardless of the quiet flag.
        """
        with caplog.at_level(logging.INFO, logger="agenthatch"):
            report = _run(quiet=True)

        assert report is not None
        assert _verdict_records(caplog), "log channel is independent of quiet"

    def test_review_failure_swallowed_and_logged(self, fake_console, monkeypatch, caplog):
        """An exception inside the review loop is swallowed (advisory only)."""
        def _boom(**kwargs):
            raise RuntimeError("review backend down")

        monkeypatch.setattr(
            "agenthatch.skill.postgen_review.iterate_until_gate", _boom
        )
        with caplog.at_level(logging.INFO, logger="agenthatch"):
            report = _run()

        assert report is None
        assert any(
            "Post-generation review failed" in r.getMessage()
            for r in caplog.records
        )
