"""Portable per-skill checkpoint lock (agent/agent/offload.py).

fcntl is POSIX-only, so the lock transparently switches to
msvcrt.locking on Windows. Collecting this module already proves the
package imports on the running platform; the tests below exercise the
REAL lock primitive of whichever platform executes them — they must
pass on both.
"""

from __future__ import annotations

import os
from pathlib import Path

from agenthatch.agent.offload import (
    CheckpointManager,
    _lock_fd,
    _lock_registry,
    _unlock_fd,
)


class TestPortableLockPrimitive:
    def test_second_fd_sees_the_conflict(self, tmp_path: Path) -> None:
        """A lock held through one fd must block a second fd on the file."""
        lock = tmp_path / ".lock"
        fd1 = os.open(lock, os.O_RDWR | os.O_CREAT)
        try:
            assert _lock_fd(fd1) is True

            fd2 = os.open(lock, os.O_RDWR | os.O_CREAT)
            try:
                assert _lock_fd(fd2) is False
            finally:
                os.close(fd2)
        finally:
            _unlock_fd(fd1)
            os.close(fd1)

    def test_lock_is_reacquirable_after_unlock(self, tmp_path: Path) -> None:
        lock = tmp_path / ".lock"
        fd1 = os.open(lock, os.O_RDWR | os.O_CREAT)
        try:
            assert _lock_fd(fd1) is True
            _unlock_fd(fd1)
            assert _lock_fd(fd1) is True
        finally:
            _unlock_fd(fd1)
            os.close(fd1)


class TestRegistrySharing:
    def test_two_managers_in_one_process_share_the_lock(
        self, tmp_path: Path
    ) -> None:
        """In-process sharing must never trip the cross-process guard."""
        m1 = CheckpointManager(tmp_path)
        m2 = CheckpointManager(tmp_path)

        assert m1._owns_lock is True
        assert m2._owns_lock is False

        lock_key = str((tmp_path / ".lock").resolve())
        assert lock_key in _lock_registry
        assert _lock_registry[lock_key][1] == 2

        del m2
        del m1

    def test_registry_empty_after_both_managers_drop(
        self, tmp_path: Path
    ) -> None:
        m1 = CheckpointManager(tmp_path)
        m2 = CheckpointManager(tmp_path)
        lock_key = str((tmp_path / ".lock").resolve())

        del m2
        del m1

        assert lock_key not in _lock_registry
