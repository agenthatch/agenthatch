"""OutputSanitizer — clean LLM output for terminal safety.

Level 0 — wraps Unicode box-drawing art in ```text blocks to prevent
terminal rendering issues.  Also handles other common artifacts.

Part of agenthatch-core: independent Agents may also need this
without depending on the full agenthatch package.
"""

from __future__ import annotations

import re

# Unicode box-drawing and block characters that may cause terminal issues
_BOX_DRAWING = re.compile(
    r"[─-╿▀-▟■-◿☀-⛿✀-➿]"
)

# Detect lines that are mostly box-drawing characters
_BOX_LINE = re.compile(r"^[\s─-╿▀-▟]+$")


class OutputSanitizer:
    """Clean LLM output for safe terminal rendering.

    Usage:
        sanitizer = OutputSanitizer()
        cleaned = sanitizer.sanitize(llm_output)
    """

    def sanitize(self, text: str) -> str:
        """Sanitize output text for terminal rendering.

        Wraps content that looks like ASCII art or box drawings in
        ```text fences to prevent terminal escape sequence issues.
        """
        if not text:
            return text

        lines = text.split("\n")
        result: list[str] = []
        in_art_block = False
        art_lines: list[str] = []

        for line in lines:
            if self._is_art_line(line):
                if not in_art_block:
                    in_art_block = True
                    art_lines = []
                art_lines.append(line)
            else:
                if in_art_block:
                    self._flush_art_block(art_lines, result)
                    in_art_block = False
                    art_lines = []
                result.append(line)

        if in_art_block:
            self._flush_art_block(art_lines, result)

        return "\n".join(result)

    @staticmethod
    def _is_art_line(line: str) -> bool:
        """Check if a line is primarily box-drawing or art characters."""
        if not line.strip():
            return False
        return bool(_BOX_LINE.match(line)) and bool(_BOX_DRAWING.search(line))

    @staticmethod
    def _flush_art_block(art_lines: list[str], result: list[str]) -> None:
        """Wrap collected art lines in a code block."""
        if len(art_lines) >= 1:
            result.append("```text")
            result.extend(art_lines)
            result.append("```")
        else:
            result.extend(art_lines)