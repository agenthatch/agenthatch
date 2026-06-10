"""StateManager — lightweight history offload persistence.

v0.7.12: Wires into ContextManager._offload_full_history() so that
compacted conversation history is saved to disk rather than discarded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class StateManager:
    """Lightweight state manager for offloading conversation history.

    Writes full history to timestamped JSONL files on context compaction.
    Wired into ContextManager._state_manager at agent init.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)

    def save_history(self, history: list) -> Path:
        """Save conversation history to a timestamped JSONL file.

        Returns the file path where history was saved.
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        filepath = self._path / f"history-{ts}.jsonl"
        filepath.write_text(
            "\n".join(json.dumps(m, default=str) for m in history)
        )
        return filepath