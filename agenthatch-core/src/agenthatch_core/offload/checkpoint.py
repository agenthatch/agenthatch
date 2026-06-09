from __future__ import annotations

from dataclasses import dataclass, field
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