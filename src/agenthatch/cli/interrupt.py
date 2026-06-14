"""Early-input capture and agent interrupt during streaming.

Based on Claude Code's earlyInput.ts pattern:
- Captures stdin in raw mode while agent is streaming output
- Buffers user keystrokes into a queue
- Ctrl+C → sets interrupt flag on the agent
- Text + Enter → buffered as early input for the next turn

Usage (in run.py):
    reader = EarlyInputReader(agent)
    reader.start()
    try:
        response = _stream_response(agent, user_input)
    finally:
        reader.stop()
    if reader.interrupted:
        # Inject interrupt message into conversation
        ...
    early_input = reader.consume()
    if early_input:
        # Use as next turn input
        ...
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Ctrl+C ascii code
_CTRL_C = 3
_CTRL_D = 4
_ENTER = 13
_NEWLINE = 10
_BACKSPACE = 127
_BACKSPACE_ALT = 8
_ESC = 27


class EarlyInputReader:
    """Captures stdin during agent streaming, similar to Claude Code's earlyInput.

    Runs a background thread that reads raw stdin character-by-character.
    Ctrl+C sets the agent's interrupt flag.  Text + Enter buffers input
    for the next conversation turn.

    Must be started before streaming begins and stopped after it ends.
    Only one instance should be active at a time.
    """

    _active_instance: EarlyInputReader | None = None

    def __init__(self, agent: Any):
        self._agent = agent
        self._buffer: list[str] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._interrupted = False
        self._original_stdin_settings: list[Any] | None = None

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def has_input(self) -> bool:
        return not self._queue.empty()

    def start(self) -> None:
        """Start capturing stdin in a background thread.

        Puts the terminal into a mode that allows reading individual
        keystrokes WITHOUT breaking output post-processing (OPOST).
        This is critical for Rich Live compatibility on macOS/Linux:
        tty.setraw() disables OPOST globally, which breaks \n→\r\n
        translation and causes Rich panels to duplicate instead of
        refreshing in-place.

        Disables:  ICANON (no line buffering), ECHO (no keystroke echo),
                   ISIG   (Ctrl+C handled manually)
        Preserves: OPOST (output processing for Rich)
        """
        if not sys.stdin.isatty():
            return

        if EarlyInputReader._active_instance is not None:
            logger.debug("EarlyInputReader already active, stopping previous")
            EarlyInputReader._active_instance.stop()

        EarlyInputReader._active_instance = self

        try:
            import termios
            fd = sys.stdin.fileno()
            self._original_stdin_settings = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            # Disable canonical mode, echo, and signal generation.
            # Preserve OPOST (output processing) — critical for Rich Live.
            new[termios.LFLAG] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)  # type: ignore[attr-defined]
            # Minimum characters to read: 1
            new[termios.CC][termios.VMIN] = 1  # type: ignore[attr-defined]
            # Timeout: 0 (non-blocking after first char)
            new[termios.CC][termios.VTIME] = 0  # type: ignore[attr-defined]
            termios.tcsetattr(fd, termios.TCSAFLUSH, new)
        except Exception:
            self._original_stdin_settings = None

        self._buffer = []
        self._running = True
        self._interrupted = False

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and restore terminal settings."""
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Restore terminal
        if self._original_stdin_settings is not None:
            try:
                import termios
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._original_stdin_settings
                )
            except Exception:
                pass
            self._original_stdin_settings = None

        if EarlyInputReader._active_instance is self:
            EarlyInputReader._active_instance = None

    def consume(self) -> str:
        """Return and clear any buffered early input."""
        parts: list[str] = []
        while not self._queue.empty():
            try:
                parts.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(parts).strip()

    def _read_loop(self) -> None:
        """Background thread: read stdin char by char."""
        fd = sys.stdin.fileno()
        while self._running:
            try:
                data = os.read(fd, 32)
                if not data:
                    break
                for byte in data:
                    self._handle_byte(byte)
            except Exception:
                if self._running:
                    break

    def _handle_byte(self, byte: int) -> None:
        """Process a single byte from stdin."""
        if byte == _CTRL_C:
            # Interrupt: set flag on agent
            self._interrupted = True
            self._buffer.clear()
            # Set interrupt flag on agent's loop/engine
            self._set_interrupt_flag()
            return

        if byte == _CTRL_D:
            if not self._buffer:
                # Empty buffer + Ctrl+D → signal EOF/exit via interrupt
                self._interrupted = True
                self._set_interrupt_flag()
                return

        if byte in (_ENTER, _NEWLINE):
            text = "".join(self._buffer)
            self._buffer.clear()
            if text.strip():
                self._queue.put(text)
            return

        if byte in (_BACKSPACE, _BACKSPACE_ALT):
            if self._buffer:
                self._buffer.pop()
            return

        if byte == _ESC:
            # Skip escape sequences (arrow keys etc.)
            return

        # Printable ASCII + CJK (multi-byte handled by os.read)
        if byte >= 32:
            self._buffer.append(chr(byte))

    def _set_interrupt_flag(self) -> None:
        """Set the interrupt flag on the agent's loop and context."""
        try:
            agent = self._agent
            # Set on ConversationLoop if accessible
            if hasattr(agent, "_interrupted"):
                agent._interrupted = True
            # Also set on context so the LLM knows to stop
            if hasattr(agent, "ctx"):
                agent.ctx._interrupted = True
        except Exception:
            pass
