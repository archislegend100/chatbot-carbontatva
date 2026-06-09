"""
ingestion/structure_parser.py
==============================
Builds a hierarchical document structure (Book → Chapter → Section → Subsection)
from parsed PDF pages.

This module takes ParsedDocument (from pdf_loader.py) and produces
a structured representation used to assign chapter/section metadata
to every chunk.

The BEE manuals have a consistent structure:
- Chapter headings (large, bold, often numbered like "Chapter 1")
- Section headings (medium, numbered like "1.1 Section Title")
- Subsection headings (smaller, "1.1.1 Subsection")

We use a heuristic state machine that tracks the current position
in the hierarchy as we iterate through pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from app.models.schemas import ParsedDocument, ParsedPage
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class SectionSpan:
    """
    Represents a contiguous span of pages belonging to a single
    chapter / section / subsection.
    """
    chapter_num: Optional[int]
    chapter_title: Optional[str]
    section_title: Optional[str]
    subsection_title: Optional[str]
    page_start: int
    page_end: int
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def tables(self) -> list:
        t = []
        for p in self.pages:
            t.extend(p.tables)
        return t


# ============================================================
# STRUCTURE PARSER
# ============================================================


class StructureParser:
    """
    Parses a ParsedDocument into a list of SectionSpan objects.

    Each SectionSpan groups consecutive pages under the same
    chapter/section/subsection heading.

    Usage:
        parser = StructureParser()
        spans = parser.parse(parsed_doc)
    """

    # Patterns for heading detection
    # These match the BEE manual numbering conventions
    CHAPTER_PATTERNS = [
        re.compile(r"^chapter\s+(\d+)\s*[:\-–—]?\s*(.+)$", re.IGNORECASE),
        re.compile(r"^(\d+)\.\s+([A-Z][A-Za-z\s\/\-,&()]{5,80})$"),
    ]
    SECTION_PATTERNS = [
        re.compile(r"^(\d+\.\d+)\s+(.{3,100})$"),
    ]
    SUBSECTION_PATTERNS = [
        re.compile(r"^(\d+\.\d+\.\d+)\s+(.{3,100})$"),
    ]

    def parse(self, doc: ParsedDocument) -> list[SectionSpan]:
        """
        Parse a ParsedDocument into section spans.

        Args:
            doc: Fully loaded ParsedDocument.

        Returns:
            List of SectionSpan objects covering all pages.
        """
        logger.info(f"Building structure for: {doc.book_name} ({len(doc.pages)} pages)")

        spans: list[SectionSpan] = []
        current_span = self._new_span(page_start=1)

        for page in doc.pages:
            # Check if this page starts a new structural unit
            heading = page.detected_heading
            heading_level = page.heading_level

            # Also scan all blocks for heading patterns even if the loader missed them
            if not heading:
                heading, heading_level = self._scan_blocks_for_heading(page)

            if heading and heading_level:
                # Finalize current span
                if current_span.pages:
                    current_span.page_end = page.page_num - 1
                    spans.append(current_span)

                # Start new span
                current_span = self._new_span(page_start=page.page_num)
                self._apply_heading(current_span, heading, heading_level)

            current_span.pages.append(page)

        # Finalize last span
        if current_span.pages:
            current_span.page_end = current_span.pages[-1].page_num
            spans.append(current_span)

        logger.info(f"  Built {len(spans)} section spans")
        self._log_structure_sample(spans)
        return spans

    def _new_span(self, page_start: int) -> SectionSpan:
        return SectionSpan(
            chapter_num=None,
            chapter_title=None,
            section_title=None,
            subsection_title=None,
            page_start=page_start,
            page_end=page_start,
        )

    def _apply_heading(
        self, span: SectionSpan, heading: str, level: int
    ) -> None:
        """Apply a detected heading to the current span."""
        heading_clean = heading.strip()

        if level == 1:
            # Chapter
            chapter_num, chapter_title = self._extract_chapter_info(heading_clean)
            span.chapter_num = chapter_num
            span.chapter_title = chapter_title
            span.section_title = None
            span.subsection_title = None
        elif level == 2:
            # Section — retain parent chapter if we have it
            span.section_title = heading_clean
            span.subsection_title = None
        elif level == 3:
            # Subsection
            span.subsection_title = heading_clean

    def _extract_chapter_info(
        self, heading: str
    ) -> tuple[Optional[int], str]:
        """Try to extract chapter number and clean title from a heading."""
        for pattern in self.CHAPTER_PATTERNS:
            m = pattern.match(heading)
            if m:
                try:
                    num = int(m.group(1))
                    title = m.group(2).strip()
                    return num, title
                except (ValueError, IndexError):
                    pass
        return None, heading

    def _scan_blocks_for_heading(
        self, page: ParsedPage
    ) -> tuple[Optional[str], Optional[int]]:
        """
        Scan page blocks for heading patterns that the font-based
        detector might have missed (e.g., numbered sections).
        """
        for block in page.blocks:
            text = block.get("text", "").strip()
            if not text or len(text) > 150:
                continue

            # Try subsection
            for p in self.SUBSECTION_PATTERNS:
                if p.match(text):
                    return text, 3

            # Try section
            for p in self.SECTION_PATTERNS:
                if p.match(text):
                    return text, 2

            # Chapter
            for p in self.CHAPTER_PATTERNS:
                if p.match(text):
                    return text, 1

        return None, None

    def _log_structure_sample(self, spans: list[SectionSpan]) -> None:
        """Log a sample of the detected structure."""
        logger.info("  Structure sample (first 5 spans):")
        for span in spans[:5]:
            logger.info(
                f"    pp.{span.page_start}-{span.page_end} | "
                f"Ch{span.chapter_num or '?'}: {span.chapter_title or 'Unknown'} | "
                f"Sec: {span.section_title or '-'}"
            )
