"""Context compaction types (agenthatch-core)."""

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


COMPACT_SYSTEM_PROMPT = """You are a context compression specialist. Read a
conversation history and produce a structured summary preserving ALL critical
information while discarding redundant details.

## Rules
1. NEVER omit a decision, error, or artifact from the summary.
2. If the user asked for something and it hasn't been done, it MUST appear in
   pending_actions.
3. If the agent made a choice between alternatives, record it in key_decisions.
4. Be specific: "Created /tmp/report.csv (1,234 rows)" not "Created a file".

## Output format
Respond ONLY with a JSON object. No markdown fences, no explanation, no prefix
or suffix. Start with "{" and end with "}". The JSON must match this schema:

{
  "session_intent": "<one sentence>",
  "key_decisions": ["<decision>", ...],
  "artifacts_created": ["<path or description>", ...],
  "current_state": "<one paragraph>",
  "pending_actions": ["<action>", ...],
  "open_questions": ["<question>", ...],
  "errors_encountered": ["<error description>", ...],
  "tool_calls_summary": "<one paragraph>",
  "conversation_turns": <integer>
}
"""