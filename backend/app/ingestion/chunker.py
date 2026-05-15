"""
ingestion/chunker.py
====================
Multi-granular chunking strategy for the Industrial Energy Efficiency Copilot.

This module implements several chunking strategies applied in combination:

1. SECTION CHUNKS (~1000-1500 tokens)
   - One chunk per detected section span
   - Used for summarization and navigation queries
   - Captures full engineering context of a section

2. SEMANTIC CHUNKS (~300-500 tokens)
   - Fine-grained sliding window over section text
   - Used for precise QA and concept explanation
   - Overlapping to avoid boundary artifacts

3. TABLE CHUNKS
   - Each extracted table becomes a dedicated chunk
   - Preserves tabular structure as formatted text
   - Tagged as content_type=TABLE

4. LIST/CHECKLIST CHUNKS
   - Detected bullet lists / numbered lists
   - Preserved as discrete chunks
   - Tagged as content_type=LIST

All chunks get rich metadata from the section span they come from.
Chunks are assigned stable deterministic IDs based on their content hash.

Usage:
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc, spans)
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

from app.ingestion.structure_parser import SectionSpan
from app.ingestion.ontology import detect_tags
from app.models.schemas import (
    DocumentChunk,
    ChunkMetadata,
    ChunkType,
    ContentType,
    ParsedDocument,
    UtilityDomain,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# CHUNKER
# ============================================================


class DocumentChunker:
    """
    Produces multi-granular DocumentChunk objects from section spans.

    Args:
        semantic_chunk_size: Target token count for semantic chunks.
        semantic_chunk_overlap: Overlap in tokens between semantic chunks.
        section_chunk_max_size: Maximum token count for section chunks.
    """

    def __init__(
        self,
        semantic_chunk_size: int = settings.SEMANTIC_CHUNK_SIZE,
        semantic_chunk_overlap: int = settings.SEMANTIC_CHUNK_OVERLAP,
        section_chunk_max_size: int = settings.SECTION_CHUNK_SIZE,
    ):
        self.semantic_chunk_size = semantic_chunk_size
        self.semantic_chunk_overlap = semantic_chunk_overlap
        self.section_chunk_max_size = section_chunk_max_size

    def chunk_document(
        self, doc: ParsedDocument, spans: list[SectionSpan]
    ) -> list[DocumentChunk]:
        """
        Produce all chunks for a document.

        Args:
            doc: The ParsedDocument (for document-level metadata).
            spans: List of SectionSpan objects from StructureParser.

        Returns:
            List of DocumentChunk objects ready for indexing.
        """
        logger.info(f"Chunking: {doc.book_name} ({len(spans)} spans)")
        all_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for span in spans:
            span_text = span.full_text.strip()
            if not span_text or len(span_text) < 50:
                continue

            # 1. Section chunk (whole span)
            section_chunk = self._make_section_chunk(doc, span, span_text)
            if section_chunk and section_chunk.chunk_id not in seen_ids:
                all_chunks.append(section_chunk)
                seen_ids.add(section_chunk.chunk_id)

            # 2. Semantic chunks (sliding window over section text)
            for chunk in self._make_semantic_chunks(doc, span, span_text):
                if chunk.chunk_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.chunk_id)

            # 3. Table chunks
            for i, table in enumerate(span.tables):
                table_chunk = self._make_table_chunk(doc, span, table, i)
                if table_chunk and table_chunk.chunk_id not in seen_ids:
                    all_chunks.append(table_chunk)
                    seen_ids.add(table_chunk.chunk_id)

            # 4. List chunks
            for list_chunk in self._make_list_chunks(doc, span, span_text):
                if list_chunk.chunk_id not in seen_ids:
                    all_chunks.append(list_chunk)
                    seen_ids.add(list_chunk.chunk_id)

        logger.info(f"  Created {len(all_chunks)} chunks for {doc.book_name}")
        chunk_type_counts = {}
        for c in all_chunks:
            t = c.metadata.chunk_type.value
            chunk_type_counts[t] = chunk_type_counts.get(t, 0) + 1
        logger.info(f"  Chunk type breakdown: {chunk_type_counts}")

        return all_chunks

    # ------------------------------------------------------------------
    # SECTION CHUNKS
    # ------------------------------------------------------------------

    def _make_section_chunk(
        self, doc: ParsedDocument, span: SectionSpan, text: str
    ) -> DocumentChunk | None:
        """Create a single section-level chunk for the span."""
        # Truncate if too large (avoid embedding model limits)
        text = self._truncate_to_tokens(text, self.section_chunk_max_size)
        if len(text.strip()) < 50:
            return None

        meta = self._base_metadata(doc, span, ChunkType.SECTION, ContentType.TEXT)
        meta.chunk_id = self._make_id(doc.document_id, "section", span.page_start, 0)
        self._enrich_tags(meta, text, doc.utility_domain)

        return DocumentChunk(text=text, metadata=meta)

    # ------------------------------------------------------------------
    # SEMANTIC CHUNKS
    # ------------------------------------------------------------------

    def _make_semantic_chunks(
        self, doc: ParsedDocument, span: SectionSpan, text: str
    ) -> Iterator[DocumentChunk]:
        """Sliding-window semantic chunking over section text."""
        words = text.split()
        if len(words) < 30:
            return  # Too short for fine chunking

        step = max(1, self.semantic_chunk_size - self.semantic_chunk_overlap)
        chunk_idx = 0

        for start in range(0, len(words), step):
            end = min(start + self.semantic_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if len(chunk_text) < 80:
                continue

            meta = self._base_metadata(doc, span, ChunkType.SEMANTIC, ContentType.TEXT)
            meta.chunk_id = self._make_id(doc.document_id, "semantic", span.page_start, chunk_idx)
            self._enrich_tags(meta, chunk_text, doc.utility_domain)

            yield DocumentChunk(text=chunk_text, metadata=meta)
            chunk_idx += 1

            if end >= len(words):
                break

    # ------------------------------------------------------------------
    # TABLE CHUNKS
    # ------------------------------------------------------------------

    def _make_table_chunk(
        self,
        doc: ParsedDocument,
        span: SectionSpan,
        table: list[list[str]],
        table_idx: int,
    ) -> DocumentChunk | None:
        """Convert a table to a text chunk preserving tabular structure."""
        if not table:
            return None

        # Format table as pipe-separated markdown-style text
        rows = []
        for row in table:
            cells = [str(cell).strip().replace("\n", " ") for cell in row]
            rows.append(" | ".join(cells))

        table_text = "\n".join(rows)

        # Add context header from section
        context = ""
        if span.section_title:
            context = f"Table in: {span.section_title}\n"
        elif span.chapter_title:
            context = f"Table in: {span.chapter_title}\n"

        full_text = context + table_text
        full_text = self._truncate_to_tokens(full_text, settings.TABLE_CHUNK_MAX_SIZE)

        if len(full_text.strip()) < 20:
            return None

        meta = self._base_metadata(doc, span, ChunkType.TABLE, ContentType.TABLE)
        meta.chunk_id = self._make_id(doc.document_id, "table", span.page_start, table_idx)
        self._enrich_tags(meta, full_text, doc.utility_domain)

        return DocumentChunk(text=full_text, metadata=meta)

    # ------------------------------------------------------------------
    # LIST / CHECKLIST CHUNKS
    # ------------------------------------------------------------------

    def _make_list_chunks(
        self, doc: ParsedDocument, span: SectionSpan, text: str
    ) -> Iterator[DocumentChunk]:
        """
        Detect and extract bullet/numbered lists from section text.
        Lists are chunked at the list level (not item level).
        """
        for list_idx, list_text in enumerate(self._extract_lists(text)):
            if len(list_text.strip()) < 50:
                continue

            # Add context header
            context = ""
            if span.section_title:
                context = f"Checklist/List in: {span.section_title}\n"

            full_text = context + list_text
            meta = self._base_metadata(doc, span, ChunkType.LIST, ContentType.LIST)
            meta.chunk_id = self._make_id(doc.document_id, "list", span.page_start, list_idx)
            self._enrich_tags(meta, full_text, doc.utility_domain)

            yield DocumentChunk(text=full_text, metadata=meta)

    def _extract_lists(self, text: str) -> list[str]:
        """Extract list blocks from text using pattern matching."""
        # Match blocks where 3+ consecutive lines start with list markers
        list_pattern = re.compile(
            r"((?:^[\s]*(?:[•\-\*\u2022\u2023\u25E6]|\d+[.)]\s).+\n?){3,})",
            re.MULTILINE,
        )
        return [m.group(0).strip() for m in list_pattern.finditer(text)]

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _base_metadata(
        self,
        doc: ParsedDocument,
        span: SectionSpan,
        chunk_type: ChunkType,
        content_type: ContentType,
    ) -> ChunkMetadata:
        """Build a base ChunkMetadata from document and span info."""
        return ChunkMetadata(
            chunk_id="",  # Set by caller
            document_id=doc.document_id,
            book_name=doc.book_name,
            utility_domain=doc.utility_domain,
            chapter_num=span.chapter_num,
            chapter_title=span.chapter_title,
            section_title=span.section_title,
            subsection_title=span.subsection_title,
            page_start=span.page_start,
            page_end=span.page_end,
            chunk_type=chunk_type,
            content_type=content_type,
            word_count=0,
            char_count=0,
        )

    def _enrich_tags(
        self, meta: ChunkMetadata, text: str, domain: UtilityDomain
    ) -> None:
        """Add equipment and concept tags via the ontology."""
        try:
            tags = detect_tags(text, domain.value)
            meta.equipment_tags = tags.equipment_tags
            meta.concept_tags = tags.concept_tags
        except Exception:
            pass
        meta.word_count = len(text.split())
        meta.char_count = len(text)

    def _make_id(
        self, doc_id: str, chunk_type: str, page: int, idx: int
    ) -> str:
        """Create a stable, deterministic chunk ID."""
        raw = f"{doc_id}_{chunk_type}_{page}_{idx}"
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{doc_id}_{chunk_type}_p{page}_{idx}_{h}"

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Approximate truncation: 1 token ≈ 4 chars for English text.
        Truncates at word boundary.
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Truncate at last word boundary
        last_space = truncated.rfind(" ")
        return truncated[:last_space] if last_space > 0 else truncated
