"""File I/O builtin capabilities."""

import logging
from pathlib import Path
from typing import Any

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability, with_enriched_errors

logger = logging.getLogger(__name__)


class FileReaderCap(BuiltinCapability):
    name = "file_reader"
    cap_type = "io"
    description = "Read file contents"
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    }

    @with_enriched_errors
    def execute(self, path: str = "", **kwargs: Any) -> str:
        if kwargs:
            logger.warning(
                "%s received unknown parameters: %s (ignored)",
                self.__class__.__name__, list(kwargs.keys()),
            )
        p = Path(path)
        if not p.exists():
            return f"Error: file '{path}' not found"
        try:
            return p.read_text()[:10000]
        except Exception as e:
            return f"Error reading file: {e}"


class FileWriterCap(BuiltinCapability):
    name = "file_writer"
    cap_type = "io"
    description = "Write content to a file"
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    @with_enriched_errors
    def execute(self, path: str = "", content: str = "", **kwargs: Any) -> str:
        if kwargs:
            logger.warning(
                "%s received unknown parameters: %s (ignored)",
                self.__class__.__name__, list(kwargs.keys()),
            )
        try:
            Path(path).write_text(content)
            return f"File written: {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing file: {e}"


BUILTIN_REGISTRY["file_reader"] = FileReaderCap
BUILTIN_REGISTRY["file_writer"] = FileWriterCap
