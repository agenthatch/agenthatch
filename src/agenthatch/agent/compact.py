from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompactSummary:
    """Structured summary produced by LLM during context compaction."""

    session_intent: str = ""
    key_decisions: list[str] = field(default_factory=list)
    artifacts_created: list[str] = field(default_factory=list)
    current_state: str = ""
    pending_actions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    tool_calls_summary: str = ""
    conversation_turns: int = 0
    compressed_at: str = ""

    @property
    def key_findings(self) -> list[str]:
        """Alias for key_decisions — used by checkpoint serializer."""
        return self.key_decisions

    def to_text(self) -> str:
        lines = [
            f"Task: {self.session_intent}",
            f"State: {self.current_state}",
        ]
        if self.key_decisions:
            lines.append("Decisions: " + "; ".join(self.key_decisions))
        if self.pending_actions:
            lines.append("Pending: " + "; ".join(self.pending_actions))
        if self.errors_encountered:
            lines.append("Errors: " + "; ".join(self.errors_encountered))
        lines.append(f"Turns compressed: {self.conversation_turns}")
        lines.append(f"Compacted at: {self.compressed_at}")
        return "\n".join(lines)
