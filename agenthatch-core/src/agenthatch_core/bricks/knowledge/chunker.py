"""KBChunker — document chunking for knowledge base indexing (v1.0.0).

Splits documents into small chunks (500-1000 chars) at paragraph
boundaries for optimal RAG retrieval.  Inspired by OpenClaw's
knowledge-base skill, which found 800-char chunks to be the sweet
spot between context completeness and retrieval precision.

Unlike FileProcessor's 12KB session chunks (which serve context
window management), KB chunks are optimized for vector/keyword
search recall.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KBChunk:
    """A single chunk produced by KBChunker."""
    doc_id: str               # unique identifier: "{source}#{chunk_index}"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: source, chunk_index, char_offset, heading


class KBChunker:
    """Document chunker for knowledge base indexing.

    Splits text at paragraph boundaries (double newlines), keeping
    chunks within ``chunk_size`` ± 20%.  Overlap between consecutive
    chunks preserves context across boundaries.
    """

    # v1.0.1 (R3-H1): Files larger than this are skipped to prevent
    # OOM when chunking multi-GB log files that may end up in a KB
    # source directory.  10MB is ~2.5M tokens worth of text — well
    # beyond any reasonable single-file KB contribution.
    _MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters (default 800).
            chunk_overlap: Overlap between consecutive chunks (default 100).
            min_chunk_size: Minimum chunk size; smaller chunks are merged
                            into the previous chunk.

        Raises:
            ValueError: If ``chunk_size <= 0``, ``chunk_overlap < 0``,
                or ``chunk_overlap >= chunk_size`` (which would produce
                out-of-range chunks and infinite loops).
        """
        # v1.0.1 (R3-M4): Validate parameters at construction time.
        # Previously invalid values like ``chunk_size=0`` or
        # ``chunk_overlap >= chunk_size`` produced silent garbage
        # (empty chunks, oversized chunks, or infinite loops).
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}"
            )
        if chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
                f"({chunk_size}) — overlap >= size produces oversized chunks"
            )
        if min_chunk_size < 0:
            raise ValueError(
                f"min_chunk_size must be non-negative, got {min_chunk_size}"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[KBChunk]:
        """Chunk a single text document.

        Args:
            text: The full document text.
            source: Source identifier (e.g. "geography.md").
            extra_metadata: Additional metadata to attach to every chunk
                           (e.g. {"content_type": "article"}).

        Returns:
            List of KBChunk, each with a unique doc_id.
        """
        if not text or not text.strip():
            return []

        extra_metadata = extra_metadata or {}

        # Split into paragraphs (double newline boundary)
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        # Group paragraphs into chunks respecting chunk_size
        chunks: list[KBChunk] = []
        current_parts: list[str] = []
        current_size = 0
        current_heading = ""
        char_offset = 0

        for para_text, para_heading in paragraphs:
            para_len = len(para_text)

            # Track the most recent heading for metadata
            if para_heading:
                current_heading = para_heading

            # If adding this paragraph exceeds chunk_size, flush current
            if current_parts and current_size + para_len + 2 > self._chunk_size:
                chunk_content = "\n\n".join(current_parts)
                if len(chunk_content) >= self._min_chunk_size:
                    chunks.append(self._make_chunk(
                        source=source,
                        index=len(chunks),
                        content=chunk_content,
                        heading=current_heading,
                        char_offset=char_offset,
                        extra_metadata=extra_metadata,
                    ))
                    # v1.0.1 (R2b-M9): Correctly advance char_offset.
                    # The next chunk starts with `overlap_text` (taken
                    # from the END of this chunk's content), so its
                    # starting offset is:
                    #   current_offset + len(chunk_content) + 2  (past this chunk)
                    #   - len(overlap_text) - 2                 (back up for overlap)
                    # = current_offset + len(chunk_content) - len(overlap_text)
                    #
                    # The previous code only did `+ len(chunk_content) + 2`
                    # which made every chunk N>0's char_offset point PAST
                    # the overlap region — metadata was misleading for
                    # callers using char_offset to extract original text.
                    overlap_text = self._get_overlap(current_parts)
                    char_offset += len(chunk_content) + 2
                    if overlap_text:
                        char_offset -= len(overlap_text) + 2
                    # Start new chunk with overlap
                    current_parts = [overlap_text, para_text] if overlap_text else [para_text]
                    current_size = sum(len(p) for p in current_parts) + 2 * (len(current_parts) - 1)
                else:
                    # v1.0.1: Chunk too small to flush as a standalone
                    # chunk — append the new paragraph and keep
                    # accumulating so we don't lose content.  Previously
                    # this branch reset current_parts to
                    # ``[overlap_text, para_text]``, which silently dropped
                    # everything accumulated before the overlap.  The next
                    # iteration will flush once the buffer grows past
                    # ``chunk_size`` (and at that point the chunk_content
                    # will be large enough to satisfy min_chunk_size).
                    # char_offset is not advanced.
                    current_parts.append(para_text)
                    current_size += para_len + 2
            else:
                current_parts.append(para_text)
                current_size += para_len + 2  # +2 for the join

        # Flush remaining
        if current_parts:
            chunk_content = "\n\n".join(current_parts)
            # v1.0.1 (R2b-M11): Don't silently drop the tail chunk if
            # it's smaller than min_chunk_size.  Previously a final
            # paragraph of <100 chars would be lost entirely (not indexed).
            # Two cases:
            #   1. There's a previous chunk → merge the small tail into it.
            #      We append the tail content to the last chunk's content
            #      (re-creating the chunk with combined text).  This may
            #      push the previous chunk slightly over chunk_size, which
            #      is acceptable — better than losing data.
            #   2. This is the only chunk (whole document is tiny) →
            #      emit it anyway so the document has at least one entry
            #      in the index.
            if len(chunk_content) < self._min_chunk_size and chunks:
                # Merge into previous chunk
                last_chunk = chunks[-1]
                # Only merge if the combined size is reasonable (< 2x chunk_size)
                combined = last_chunk.content + "\n\n" + chunk_content
                if len(combined) <= self._chunk_size * 2:
                    chunks[-1] = KBChunk(
                        doc_id=last_chunk.doc_id,
                        content=combined,
                        metadata={**last_chunk.metadata, "merged_tail": True},
                    )
                else:
                    # Combined too big — emit small chunk anyway
                    chunks.append(self._make_chunk(
                        source=source,
                        index=len(chunks),
                        content=chunk_content,
                        heading=current_heading,
                        char_offset=char_offset,
                        extra_metadata=extra_metadata,
                    ))
            elif len(chunk_content) < self._min_chunk_size and not chunks:
                # Whole document is tiny — emit anyway so it's indexed
                chunks.append(self._make_chunk(
                    source=source,
                    index=len(chunks),
                    content=chunk_content,
                    heading=current_heading,
                    char_offset=char_offset,
                    extra_metadata=extra_metadata,
                ))
            else:
                # Normal case — chunk meets min size
                chunks.append(self._make_chunk(
                    source=source,
                    index=len(chunks),
                    content=chunk_content,
                    heading=current_heading,
                    char_offset=char_offset,
                    extra_metadata=extra_metadata,
                ))

        return chunks

    def chunk_file(
        self,
        file_path: Path,
        extra_metadata: dict[str, Any] | None = None,
        source_label: str | None = None,
    ) -> list[KBChunk]:
        """Chunk a file by reading it and delegating to chunk_text.

        Args:
            file_path: Path to the file to chunk.
            extra_metadata: Additional metadata to attach to every chunk.
            source_label: Override for the ``source`` identifier used in
                ``doc_id`` and ``metadata["source"]``.  v1.0.1 (R2-C1):
                previously this was always ``file_path.name`` (basename
                only), which caused doc_id collisions like
                ``notes.md#0`` when two files in different subdirectories
                shared the same name — ``INSERT OR REPLACE`` silently
                dropped the earlier file's chunks.  Callers should pass
                a path-unique label (e.g. relative path from KB root).

        Returns:
            List of KBChunk.
        """
        # v1.0.1 (R3-H1): Skip oversized files to prevent OOM.  A 500MB
        # log file in a KB source directory would otherwise be read into
        # memory in full.  10MB cap matches the ``_MAX_FILE_SIZE_BYTES``
        # class constant — adjust there if a larger cap is needed.
        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            logger.warning(
                "KBChunker: cannot stat %s: %s — skipping",
                file_path, e,
            )
            return []
        if file_size > self._MAX_FILE_SIZE_BYTES:
            logger.warning(
                "KBChunker: %s is %d bytes (exceeds %d limit) — skipping. "
                "If this is intentional, split the file or raise "
                "KBChunker._MAX_FILE_SIZE_BYTES.",
                file_path, file_size, self._MAX_FILE_SIZE_BYTES,
            )
            return []

        # v1.0.1 (R3-M3): Detect binary files via null-byte presence in
        # the first 4KB.  ``read_text(encoding="utf-8")`` would catch
        # most binaries via ``UnicodeDecodeError``, but files with
        # embedded null bytes (PDFs, images with ASCII headers) may
        # partially decode and produce garbage chunks.
        try:
            with open(file_path, "rb") as bf:
                head = bf.read(4096)
            if b"\x00" in head:
                logger.warning(
                    "KBChunker: %s appears to be binary (null bytes in "
                    "first 4KB) — skipping",
                    file_path,
                )
                return []
        except OSError as e:
            logger.warning(
                "KBChunker: cannot read %s header: %s — skipping",
                file_path, e,
            )
            return []

        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # v1.0.1 (M7): Log the failure so empty KB indices are
            # debuggable.  Previously the exception was captured but
            # never logged, making "why is my KB empty?" unanswerable.
            logger.warning(
                "KBChunker: failed to read %s: %s — skipping",
                file_path, e,
            )
            return []
        source = source_label if source_label is not None else file_path.name
        meta = {"file_path": str(file_path)}
        if extra_metadata:
            meta.update(extra_metadata)
        return self.chunk_text(text, source=source, extra_metadata=meta)

    def _split_paragraphs(self, text: str) -> list[tuple[str, str]]:
        """Split text into paragraphs, tracking the nearest heading.

        Returns list of (paragraph_text, heading) tuples.  The heading
        is the most recent markdown heading (## or ###) before the paragraph.
        """
        paragraphs: list[tuple[str, str]] = []
        current_heading = ""

        # Split on double newlines (paragraph boundaries)
        raw_paragraphs = re.split(r"\n\s*\n", text.strip())

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph is a heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", para)
            if heading_match:
                current_heading = heading_match.group(2).strip()
                # Headings are kept as part of the next chunk's content
                # but also tracked separately in metadata
                paragraphs.append((para, current_heading))
            else:
                paragraphs.append((para, current_heading))

        return paragraphs

    def _get_overlap(self, parts: list[str]) -> str:
        """Get overlap text from the end of the current parts.

        Returns the last paragraph if it fits within chunk_overlap,
        otherwise an empty string.
        """
        if not parts or self._chunk_overlap <= 0:
            return ""
        last = parts[-1]
        if len(last) <= self._chunk_overlap:
            return last
        # Take the last chunk_overlap characters of the last paragraph
        return last[-self._chunk_overlap:]

    @staticmethod
    def _make_chunk(
        source: str,
        index: int,
        content: str,
        heading: str,
        char_offset: int,
        extra_metadata: dict[str, Any],
    ) -> KBChunk:
        """Create a KBChunk with metadata."""
        doc_id = f"{source}#{index}"
        metadata = {
            "source": source,
            "chunk_index": index,
            "char_offset": char_offset,
            "heading": heading,
        }
        metadata.update(extra_metadata)
        return KBChunk(doc_id=doc_id, content=content, metadata=metadata)
