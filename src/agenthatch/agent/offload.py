from __future__ import annotations

import json
import logging
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
    """Saves and restores conversation state."""

    def __init__(self, session_dir: Path):
        self._dir = session_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "checkpoint.json"

    def save(self, checkpoint: Checkpoint) -> None:
        checkpoint.saved_at = datetime.now().isoformat()
        tmp = self._dir / "checkpoint.tmp.json"
        with open(tmp, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)
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
                "Checkpoint file is corrupted: %s. Starting fresh.", e
            )
            try:
                self._path.unlink()
            except OSError:
                pass
            return None

    def exists(self) -> bool:
        return self._path.exists()
