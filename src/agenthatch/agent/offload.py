from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
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
        data = json.loads(path.read_text())
        return CompactSummary(**data)
