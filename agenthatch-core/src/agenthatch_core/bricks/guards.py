"""OutputGuard — compiled regex validators from ANCHOR_RULES.

Level 0 — converts declarative safety rules into executable code.
Not prompt-based: these are real regex validators that run against
LLM output before returning to the user.

guard_active boolean on BrickManifest controls whether validate() is
called at all — prompt-only skills with no sensitive data skip it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardRule:
    """A single compiled guard rule."""
    pattern: str
    description: str
    action: str = "redact"  # "redact" | "block" | "warn"
    replacement: str = "***"
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self._compiled = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
        except re.error:
            self._compiled = None

    def matches(self, text: str) -> bool:
        """Check if the pattern matches."""
        return bool(self._compiled and self._compiled.search(text))

    def apply(self, text: str) -> tuple[str, bool]:
        """Apply the rule to text. Returns (modified_text, was_modified)."""
        if not self._compiled:
            return text, False
        result, count = self._compiled.subn(self.replacement, text)
        return result, count > 0


@dataclass
class OutputGuard:
    """Compiled output validator from ANCHOR_RULES.

    Usage:
        guard = OutputGuard.from_rules([
            {"pattern": r"\\b\\d{16}\\b", "description": "Credit card number",
             "action": "redact", "replacement": "***CC***"},
        ])
        cleaned, violations = guard.validate(output_text)
    """

    rules: list[GuardRule] = field(default_factory=list)

    def validate(self, text: str) -> tuple[str, list[str]]:
        """Validate and clean output text.

        Returns:
            (cleaned_text, list_of_violation_descriptions)
        """
        violations: list[str] = []
        cleaned = text

        for rule in self.rules:
            if rule.action == "block" and rule.matches(cleaned):
                violations.append(f"BLOCKED: {rule.description}")
                return "", violations

            if rule.action == "redact":
                cleaned, modified = rule.apply(cleaned)
                if modified:
                    violations.append(f"REDACTED: {rule.description}")

            if rule.action == "warn" and rule.matches(cleaned):
                violations.append(f"WARN: {rule.description}")

        return cleaned, violations

    @classmethod
    def from_rules(cls, rules: list[dict[str, str] | str]) -> OutputGuard:
        """Create OutputGuard from declarative rule dicts or simple strings.

        Each rule can be:
        - A dict with 'pattern' (regex), 'description', optional 'action'/'replacement'
        - A plain string (treated as text guideline — logged but not matched)

        String-only rules are skipped during validation (they're guidelines,
        not regex patterns). Only dict rules with valid patterns are compiled.
        """
        compiled = []
        for r in rules:
            if isinstance(r, str):
                # String-only rules are guidelines, skip pattern compilation
                continue
            if isinstance(r, dict) and "pattern" in r:
                compiled.append(GuardRule(
                    pattern=r["pattern"],
                    description=r.get("description", r["pattern"]),
                    action=r.get("action", "redact"),
                    replacement=r.get("replacement", "***"),
                ))
        return cls(rules=compiled)