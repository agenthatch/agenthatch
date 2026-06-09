from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agenthatch.agent.compact import CompactSummary

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    skill_id: str
    session_id: str
    started_at: str
    last_active_at: str
    summary: CompactSummary | None = None
    total_turns: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""


class StateManager:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_summary(self, summary: CompactSummary) -> Path:
        path = self.state_dir / "summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2))
        return path

    def save_history(self, history: list[dict[str, Any]]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.state_dir / f"session_{timestamp}.json"
        path.write_text(json.dumps(history, indent=2, default=str))
        return path

    def load_summary(self) -> CompactSummary | None:
        path = self.state_dir / "summary.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CompactSummary(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Summary file is corrupted: %s. Starting fresh.", e
            )
            try:
                path.unlink()
            except OSError:
                pass
            return None


@dataclass
class Checkpoint:
    session_id: str
    saved_at: str = ""
    turn_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    compact_failures: int = 0
    cb_state: str = "closed"
    cb_failures: int = 0


class CheckpointManager:
    """Saves and restores conversation state.

    v0.7.6: Per-skill process lock via fcntl.flock() prevents checkpoint
    corruption when the same skill is run twice from the same directory.
    Different skills from the same directory run concurrently without blocking.
    """

    def __init__(self, session_dir: Path):
        self._dir = session_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "checkpoint.json"
        self._lock_path = self._dir / ".lock"
        self._lock_fd: int | None = None
        self._acquire_lock()

    def _acquire_lock(self) -> None:
        """Acquire per-skill lock on startup.

        Uses fcntl.flock() which is kernel-managed — auto-released on
        process exit (even on crash). Non-blocking: raises RuntimeError
        immediately if the same skill is already running in this directory.
        """
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"Skill '{self._dir.name}' is already running in this directory. "
                f"Wait for the other session to exit or use a different working directory."
            ) from None
        self._lock_fd = fd

    def __del__(self) -> None:
        """Best-effort lock release on clean exit."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass

    def save(self, checkpoint: Checkpoint) -> None:
        checkpoint.saved_at = datetime.now().isoformat()
        tmp = self._dir / "checkpoint.tmp.json"
        with open(tmp, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(self._path)

    def load(self) -> Checkpoint | None:
        if not self._path.exists():
            return None
        try:
            with open(self._path) as f:
                data = json.load(f)
            return Checkpoint(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Checkpoint file is corrupted: %s. Renaming to .bak.", e
            )
            try:
                bak_path = self._path.with_suffix(".json.bak")
                self._path.rename(bak_path)
                logger.info("Corrupted checkpoint backed up to %s", bak_path)
            except OSError:
                pass
            return None

    def exists(self) -> bool:
        return self._path.exists()
