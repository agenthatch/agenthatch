"""Regression guard: ``EarlyInputReader`` must stay inactive without raw mode.

On Windows ``termios`` does not exist, so the raw-mode setup inside
``EarlyInputReader.start()`` always fails there.  The byte-wise reader thread
used to be started anyway.

That was the cause of the "blind input" bug in ``agenthatch run``: without raw
mode, ``os.read`` on a Windows console blocks until Enter, so ``stop()`` —
which only joins the thread for a second — left it stuck holding stdin.  The
orphaned thread then competed with prompt_toolkit for keystrokes, so the
``You:`` prompt received nothing to echo, and input only appeared to "unlock"
after pressing Enter (which released the blocked read).

``start()`` now returns early when raw mode is unavailable, so no thread is
created and the console is left to prompt_toolkit.
"""

from __future__ import annotations

import os
import sys

import pytest

from agenthatch.cli.interrupt import EarlyInputReader


class _FakeTtyStdin:
    """Minimal stdin stub that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0


def _simulate_no_termios(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import termios`` raise ImportError, as it does on Windows.

    A ``None`` entry in ``sys.modules`` makes the import machinery raise
    ImportError, which mirrors the platform rather than the host OS — so this
    guard runs identically on Linux and macOS CI.
    """
    monkeypatch.setitem(sys.modules, "termios", None)
    monkeypatch.setattr(sys, "stdin", _FakeTtyStdin())


def test_reader_stays_inactive_without_raw_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_no_termios(monkeypatch)

    reader = EarlyInputReader(object())
    reader.start()

    assert reader._thread is None, "reader thread must not start without raw mode"
    assert reader._running is False, "reader must not report itself as active"
    assert EarlyInputReader._active_instance is None, (
        "active instance must be released so a later start() does not try to "
        "stop a reader that never ran"
    )


def test_stop_and_consume_are_safe_noops_when_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_no_termios(monkeypatch)

    reader = EarlyInputReader(object())
    reader.start()
    reader.stop()

    assert reader.consume() == ""
    assert reader.interrupted is False
    assert reader.has_input is False


def test_repeated_cycles_do_not_leak_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each turn calls start()/stop(); none of them may leave a thread behind."""
    _simulate_no_termios(monkeypatch)

    for _ in range(3):
        reader = EarlyInputReader(object())
        reader.start()
        reader.stop()
        assert reader._thread is None


def test_reader_still_starts_when_raw_mode_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The POSIX path must be untouched — macOS/Linux keep early input.

    ``termios`` is present there, so the new guard never fires and the reader
    behaves exactly as before.  This test pins that: it stubs ``termios`` with
    a working implementation and asserts the thread *is* started.
    """

    class _FakeTermios:
        # Real termios indices: tcgetattr returns
        # [iflag, oflag, cflag, lflag, ispeed, ospeed, cc] — lflag holds the
        # ECHO/ICANON/ISIG bits and cc is the control-character array.
        LFLAG = 3
        CC = 6
        ICANON = 2
        ECHO = 8
        ISIG = 1
        VMIN = 6
        VTIME = 5
        TCSAFLUSH = 2

        def tcgetattr(self, fd: int) -> list[object]:
            return [0, 0, 0, 0, 0, 0, [0] * 32]

        def tcsetattr(self, fd: int, when: int, attrs: list[object]) -> None:
            return None

    monkeypatch.setitem(sys.modules, "termios", _FakeTermios())
    monkeypatch.setattr(sys, "stdin", _FakeTtyStdin())
    # Let the reader loop exit at once instead of blocking on a real fd.
    monkeypatch.setattr(os, "read", lambda fd, n: b"")

    reader = EarlyInputReader(object())
    reader.start()
    try:
        assert reader._thread is not None, "reader must run when raw mode works"
        assert reader._running is True
        reader._thread.join(timeout=2.0)
    finally:
        reader.stop()

    assert EarlyInputReader._active_instance is None
