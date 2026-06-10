from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


# v0.7.12: CheckpointManager — lightweight JSONL-based persistence
class CheckpointManager:
    """Lightweight checkpoint persistence manager.

    Saves conversation state to JSONL files in a checkpoint directory.
    """
    def __init__(self, path: Path):
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to a timestamped JSON file."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        filepath = self._path / f"checkpoint-{ts}.json"
        filepath.write_text(json.dumps(checkpoint.__dict__, default=str))