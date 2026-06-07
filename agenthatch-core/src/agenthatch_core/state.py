"""StateManager — persistent agent state via STATE.json.

Level 0 — thread-safe JSON file persistence for agent conversation
state, checkpoints, and metadata.  Used by the loop engine to save
and restore state across turns.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateManager:
    """Thread-safe STATE.json manager.

    Usage:
        state = StateManager(Path(".agenthatch/STATE.json"))
        state.set("current_step", 3)
        step = state.get("current_step")  # 3
        state.save()  # persist to disk
    """

    def __init__(self, path: str | Path = "STATE.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from disk or return empty dict."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", self._path, e)
        return {
            "version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoints": [],
            "metadata": {},
            "state": {},
        }

    def save(self) -> None:
        """Persist current state to disk atomically."""
        with self._lock:
            self._data["updated_at"] = datetime.now(UTC).isoformat()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state."""
        return self._data.get("state", {}).get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state (does not persist until save())."""
        self._data.setdefault("state", {})[key] = value

    def delete(self, key: str) -> None:
        """Delete a key from state."""
        self._data.get("state", {}).pop(key, None)

    def all_state(self) -> dict[str, Any]:
        """Return all state key-value pairs."""
        return dict(self._data.get("state", {}))

    # ── Checkpoints ────────────────────────────────────────────────────

    def add_checkpoint(self, name: str, data: dict[str, Any] | None = None) -> None:
        """Record a named checkpoint."""
        checkpoint = {
            "name": name,
            "timestamp": datetime.now(UTC).isoformat(),
            "turn": self.get("turn_count", 0),
            "data": data or {},
        }
        self._data.setdefault("checkpoints", []).append(checkpoint)

    def last_checkpoint(self) -> dict[str, Any] | None:
        """Return the most recent checkpoint."""
        checkpoints = self._data.get("checkpoints", [])
        return checkpoints[-1] if checkpoints else None

    def checkpoints(self) -> list[dict[str, Any]]:
        """Return all checkpoints."""
        return list(self._data.get("checkpoints", []))

    # ── Metadata ───────────────────────────────────────────────────────

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata field."""
        self._data.setdefault("metadata", {})[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata field."""
        return self._data.get("metadata", {}).get(key, default)

    @property
    def path(self) -> Path:
        """Path to the STATE.json file."""
        return self._path
