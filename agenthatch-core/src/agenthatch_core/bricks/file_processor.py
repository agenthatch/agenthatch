"""FileProcessor — file ingestion, chunking, and format detection.

Level 0 — processes uploaded/attached files for agent consumption.
Detects format, chunks large files for context window fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum chunk size in characters (~4K tokens)
DEFAULT_CHUNK_SIZE = 12000

# Recognized extensions for format detection
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".bash", ".zsh", ".fish",
    ".html", ".css", ".xml", ".csv", ".tsv", ".log", ".env",
    ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
    ".rb", ".php", ".r", ".sql", ".graphql", ".proto",
})


@dataclass
class FileChunk:
    """A chunk of processed file content."""
    index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedFile:
    """Result of file processing."""
    path: Path
    format: str           # "text" | "binary" | "unknown"
    encoding: str
    total_chars: int
    chunks: list[FileChunk] = field(default_factory=list)
    error: str = ""


class FileProcessor:
    """Process files for agent consumption.

    Usage:
        fp = FileProcessor(chunk_size=12000)
        result = fp.process(Path("document.txt"))
        for chunk in result.chunks:
            agent.ctx.add_file_context(chunk.content)
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._chunk_size = chunk_size

    def process(self, filepath: Path) -> ProcessedFile:
        """Process a single file into chunks."""
        if not filepath.exists():
            return ProcessedFile(
                path=filepath, format="unknown", encoding="",
                total_chars=0, error=f"File not found: {filepath}",
            )

        fmt = self._detect_format(filepath)
        if fmt != "text":
            return ProcessedFile(
                path=filepath, format=fmt, encoding="",
                total_chars=0, error=f"Unsupported format: {fmt}",
            )

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = filepath.read_text(encoding="latin-1")
            except Exception as e:
                return ProcessedFile(
                    path=filepath, format="binary", encoding="",
                    total_chars=0, error=f"Cannot decode: {e}",
                )
        except Exception as e:
            return ProcessedFile(
                path=filepath, format="unknown", encoding="",
                total_chars=0, error=str(e),
            )

        chunks = self._chunk_content(content)
        return ProcessedFile(
            path=filepath,
            format="text",
            encoding="utf-8",
            total_chars=len(content),
            chunks=chunks,
        )

    def process_multiple(self, paths: list[Path]) -> list[ProcessedFile]:
        """Process multiple files."""
        return [self.process(p) for p in paths]

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension and content sniff."""
        suffix = filepath.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return "text"
        # Try reading first bytes to detect text
        try:
            with open(filepath, "rb") as f:
                head = f.read(1024)
            # Check for null bytes (binary indicator)
            if b"\x00" in head:
                return "binary"
            # Try decoding first chunk
            head.decode("utf-8")
            return "text"
        except (OSError, UnicodeDecodeError):
            return "binary"

    def _chunk_content(self, content: str) -> list[FileChunk]:
        """Split content into chunks at paragraph boundaries."""
        if len(content) <= self._chunk_size:
            return [FileChunk(index=0, content=content)]

        chunks: list[FileChunk] = []
        paragraphs = content.split("\n\n")
        current_chunk: list[str] = []
        current_size = 0
        chunk_index = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for the \n\n separator
            if current_size + para_size > self._chunk_size and current_chunk:
                chunks.append(FileChunk(
                    index=chunk_index,
                    content="\n\n".join(current_chunk),
                ))
                chunk_index += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append(FileChunk(
                index=chunk_index,
                content="\n\n".join(current_chunk),
            ))

        return chunks
