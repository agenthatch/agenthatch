"""CompiledGuard — compiled regex validators from ANCHOR_RULES.

Level 0 — converts declarative safety rules into executable code.
Not prompt-based: these are real regex validators that run against
LLM output before returning to the user and against tool call arguments
before execution.

v0.7.6: Unified guard — output validation + pre-tool-call validation.
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
    scope: str = "output"   # "output" | "pre_tool" | "both"
    blocked_tools: list[str] = field(default_factory=list)
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


# Compile built-in security patterns once at module load.
# Must be after GuardRule class definition.
def _compile_builtin_patterns() -> list[GuardRule]:
    patterns: list[tuple[str, str, str, str]] = [
        (r"\bsk-[A-Za-z0-9]{24,}\b", "API key leak (sk-...)",
         "redact", "***API_KEY***"),
        (r"\bsk-ant-[A-Za-z0-9]{24,}\b", "Anthropic API key leak",
         "redact", "***ANTHROPIC_KEY***"),
        (r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", "Bearer token leak",
         "redact", "***BEARER_TOKEN***"),
        (r"api[_-]?key[=:]\s*[A-Za-z0-9\-_]{20,}", "API key in output",
         "redact", "api_key=***REDACTED***"),
        (r"secret[=:]\s*[A-Za-z0-9\-_]{16,}", "Secret in output",
         "redact", "secret=***REDACTED***"),
        (r"password[=:]\s*\S+", "Password in output",
         "redact", "password=***REDACTED***"),
        (r"token[=:]\s*[A-Za-z0-9\-_]{16,}", "Token in output",
         "redact", "token=***REDACTED***"),
    ]
    rules: list[GuardRule] = []
    for pat, desc, action, repl in patterns:
        try:
            compiled = re.compile(pat, re.IGNORECASE | re.MULTILINE)
            rules.append(GuardRule(
                pattern=pat, description=desc, action=action,
                replacement=repl, _compiled=compiled,
            ))
        except re.error:
            pass
    return rules


_BUILTIN_SECURITY_PATTERNS: list[GuardRule] = _compile_builtin_patterns()


@dataclass
class CompiledGuard:
    """Unified guard: output validation + pre-tool-call validation.

    Compiled from declarative rules in agenthatch.yaml instructions.rules.
    Replaces OutputGuard (v0.7.5) with added pre-tool validation.

    Usage:
        guard = CompiledGuard.from_rules([
            {"pattern": r"\\b\\d{16}\\b", "description": "Credit card number",
             "action": "redact", "replacement": "***CC***"},
            {"pattern": "delete|remove", "description": "Never delete",
             "scope": "pre_tool", "blocked_tools": ["delete_document"]},
        ])
        cleaned, violations = guard.validate_output(output_text)
        allowed, msg = guard.check_pre_tool_call(tool_name, arguments)
    """

    output_rules: list[GuardRule] = field(default_factory=list)
    pre_rules: list[GuardRule] = field(default_factory=list)

    def validate_output(self, text: str) -> tuple[str, list[str]]:
        """Validate and clean output text.

        Returns:
            (cleaned_text, list_of_violation_descriptions)
        """
        violations: list[str] = []
        cleaned = text

        for rule in self.output_rules:
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

    # Backward compatibility alias
    def validate(self, text: str) -> tuple[str, list[str]]:
        """Backward-compatible alias for validate_output()."""
        return self.validate_output(text)

    def check_pre_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Check before tool execution. Returns (allowed, rejection_message)."""
        # 1. Check if this tool is blocked by any pre_tool rule
        for rule in self.pre_rules:
            if tool_name in rule.blocked_tools:
                return False, f"Blocked: {rule.description}"

        # 2. Search arguments for forbidden patterns
        for rule in self.pre_rules:
            if rule._compiled is None:
                continue
            if self._search_recursive(arguments, rule._compiled):
                return False, f"Blocked: {rule.description}"

        return True, ""

    @staticmethod
    def _search_recursive(obj: Any, pattern: re.Pattern) -> bool:
        """Recursively search for pattern in nested dicts/lists/strings."""
        if isinstance(obj, str):
            return bool(pattern.search(obj))
        if isinstance(obj, dict):
            return any(CompiledGuard._search_recursive(v, pattern) for v in obj.values())
        if isinstance(obj, list):
            return any(CompiledGuard._search_recursive(item, pattern) for item in obj)
        return False

    @classmethod
    def from_rules(cls, rules: list[dict[str, Any] | str]) -> CompiledGuard:
        """Create CompiledGuard from declarative rule dicts or simple strings.

        Each rule can be:
        - A dict with 'pattern', 'description', optional 'action'/'scope'/'blocked_tools'
        - A plain string (treated as text guideline — logged but not matched)

        String-only rules are skipped during compilation.
        Built-in security patterns (API key leaks, tokens, credentials) are
        always included in output_rules.
        """
        output_rules: list[GuardRule] = list(_BUILTIN_SECURITY_PATTERNS)
        pre_rules: list[GuardRule] = []

        for r in rules:
            if isinstance(r, str):
                continue
            if isinstance(r, dict) and "pattern" in r:
                scope = r.get("scope", "output")
                compiled = GuardRule(
                    pattern=r["pattern"],
                    description=r.get("description", r["pattern"]),
                    action=r.get("action", "redact"),
                    scope=scope,
                    blocked_tools=r.get("blocked_tools", []),
                    replacement=r.get("replacement", "***"),
                )
                if scope in ("output", "both"):
                    output_rules.append(compiled)
                if scope in ("pre_tool", "both"):
                    pre_rules.append(compiled)
                # If scope is "both", intentionally append to both lists

        return cls(output_rules=output_rules, pre_rules=pre_rules)

    @property
    def rules(self) -> list[GuardRule]:
        """Backward-compatible access: returns output_rules."""
        return self.output_rules


# Backward-compatible alias for one release
OutputGuard = CompiledGuard