from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agenthatch.agent.compact import CompactSummary

logger = logging.getLogger(__name__)

# ── Process-level lock registry ───────────────────────────────────────
# Key: resolved lock file path → (fd, refcount)
# This allows multiple CheckpointManager instances in the same process
# to share the same flock without hitting BlockingIOError.
_lock_registry: dict[str, tuple[int, int]] = {}
_lock_registry_lock = threading.Lock()


def _cleanup_locks() -> None:
    """atexit handler: release all held locks."""
    with _lock_registry_lock:
        for fd, _ in _lock_registry.values():
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except OSError:
                pass
        _lock_registry.clear()


atexit.register(_cleanup_locks)


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

    v0.8.1: Process-level lock registry with reference counting.
    Multiple CheckpointManager instances in the same process share
    the same flock — no BlockingIOError on repeated from_ahspec().
    """

    def __init__(self, session_dir: Path):
        self._dir = session_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "checkpoint.json"
        self._lock_path = self._dir / ".lock"
        self._lock_fd: int | None = None
        self._owns_lock: bool = False
        self._acquire_or_share_lock()

    def _acquire_or_share_lock(self) -> None:
        """Acquire or share the per-skill lock.

        First call in this process: acquire lock, register fd.
        Subsequent calls: share existing fd, increment refcount.
        Cross-process: BlockingIOError → RuntimeError (skill already running).
        """
        lock_key = str(self._lock_path.resolve())

        with _lock_registry_lock:
            if lock_key in _lock_registry:
                fd, refcount = _lock_registry[lock_key]
                _lock_registry[lock_key] = (fd, refcount + 1)
                self._lock_fd = fd
                self._owns_lock = False
                return

        # First acquirer in this process
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"Skill '{self._dir.name}' is already running in another process. "
                f"Wait for it to exit or use a different working directory."
            ) from None

        _lock_registry[lock_key] = (fd, 1)
        self._lock_fd = fd
        self._owns_lock = True

    def __del__(self) -> None:
        """Best-effort lock release on clean exit via refcounting."""
        if self._lock_fd is not None:
            lock_key = str(self._lock_path.resolve())
            with _lock_registry_lock:
                if lock_key in _lock_registry:
                    fd, refcount = _lock_registry[lock_key]
                    if refcount <= 1:
                        del _lock_registry[lock_key]
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                            os.close(fd)
                        except OSError:
                            pass
                    else:
                        _lock_registry[lock_key] = (fd, refcount - 1)

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
