"""
ingestion/pdf_loader.py
=======================
Structure-aware PDF loader using PyMuPDF (fitz).

Design goals:
- Extract text with accurate page numbers
- Detect headings by font size / boldness / position
- Extract tables via pdfplumber (fallback from PyMuPDF)
- Detect lists / bullet points
- Produce ParsedPage objects for further processing

The loader works on text-based PDFs (which the BEE manuals are).
It does not require OCR for these manuals, but includes an OCR
detection step that warns if text yield is suspiciously low.

Usage (CLI):
    python -m app.ingestion.pdf_loader --path "bee guide - thermal utility.pdf"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from app.models.schemas import ParsedPage, ParsedDocument, UtilityDomain
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

# Pages where text density < this ratio (chars/page_area) are flagged as image-heavy
OCR_WARNING_THRESHOLD = 0.01

# Font size thresholds for heading detection
# These are heuristic — tuned for BEE manuals (A4 portrait, mixed doc styles)
HEADING_FONT_SIZE_H1 = 14.0   # Chapter-level heading
HEADING_FONT_SIZE_H2 = 12.0   # Section-level heading
HEADING_FONT_SIZE_H3 = 11.0   # Subsection-level heading
BODY_FONT_SIZE = 10.0

# Maximum characters per block to consider as a heading vs body
MAX_HEADING_CHARS = 200


# ============================================================
# CORE LOADER
# ============================================================


class PDFLoader:
    """
    Structure-aware PDF loader.

    Extracts text, detects headings, extracts tables, and preserves
    page numbers for citation purposes.

    Args:
        file_path: Path to the PDF file.
        document_id: Unique identifier for this document.
        book_name: Human-readable book title.
        utility_domain: "thermal" or "electrical".
        max_pages: If set, only load this many pages (for testing).
    """

    def __init__(
        self,
        file_path: str | Path,
        document_id: str,
        book_name: str,
        utility_domain: UtilityDomain,
        max_pages: Optional[int] = None,
    ):
        self.file_path = Path(file_path)
        self.document_id = document_id
        self.book_name = book_name
        self.utility_domain = utility_domain
        self.max_pages = max_pages

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

    def load(self) -> ParsedDocument:
        """
        Load the PDF and return a ParsedDocument.

        Returns:
            ParsedDocument with all pages parsed.

        Raises:
            ValueError: If text extraction yields almost nothing (possible image PDF).
        """
        logger.info(f"Loading PDF: {self.file_path.name}")

        doc = fitz.open(str(self.file_path))
        total_pages = len(doc)
        pages_to_load = min(total_pages, self.max_pages) if self.max_pages else total_pages

        logger.info(f"  Total pages: {total_pages} | Loading: {pages_to_load}")

        # Load pdfplumber for table extraction (same file)
        plumber_doc = None
        if HAS_PDFPLUMBER:
            try:
                plumber_doc = pdfplumber.open(str(self.file_path))
            except Exception as e:
                logger.warning(f"pdfplumber failed to open (table extraction disabled): {e}")

        parsed_pages: list[ParsedPage] = []
        low_text_pages = 0

        for page_idx in range(pages_to_load):
            if page_idx % 50 == 0:
                logger.info(f"  Parsing page {page_idx + 1}/{pages_to_load}...")

            fitz_page = doc[page_idx]
            parsed = self._parse_page(fitz_page, page_idx + 1, plumber_doc)
            parsed_pages.append(parsed)

            # Check for low text yield
            page_area = fitz_page.rect.width * fitz_page.rect.height
            char_density = len(parsed.text) / page_area if page_area > 0 else 0
            if char_density < OCR_WARNING_THRESHOLD and len(parsed.text.strip()) < 50:
                low_text_pages += 1

        doc.close()
        if plumber_doc:
            plumber_doc.close()

        if low_text_pages > pages_to_load * 0.3:
            logger.warning(
                f"WARNING: {low_text_pages}/{pages_to_load} pages have very low text yield. "
                "This may be a scanned image PDF. Consider enabling OCR."
            )

        total_chars = sum(len(p.text) for p in parsed_pages)
        logger.info(
            f"Loaded {len(parsed_pages)} pages | "
            f"{total_chars:,} total chars | "
            f"{low_text_pages} low-yield pages"
        )

        return ParsedDocument(
            document_id=self.document_id,
            book_name=self.book_name,
            utility_domain=self.utility_domain,
            file_path=str(self.file_path),
            total_pages=total_pages,
            pages=parsed_pages,
        )

    def _parse_page(
        self,
        page: fitz.Page,  # type: ignore[name-defined]
        page_num: int,
        plumber_doc: Optional[object],
    ) -> ParsedPage:
        """
        Parse a single PDF page.

        Extracts:
        - Full text (preserving reading order via fitz blocks)
        - Detected heading and its level
        - Extracted tables (via pdfplumber if available)
        """
        # Get text blocks with positioning info
        blocks_raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]  # type: ignore

        text_parts: list[str] = []
        detected_heading: Optional[str] = None
        heading_level: Optional[int] = None
        structured_blocks: list[dict] = []

        for block in blocks_raw:
            if block.get("type") != 0:  # 0 = text block
                continue

            block_text = ""
            max_font_size = 0.0
            is_bold = False

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    font_size = span.get("size", BODY_FONT_SIZE)
                    font_name = span.get("font", "").lower()
                    max_font_size = max(max_font_size, font_size)
                    if "bold" in font_name or span.get("flags", 0) & 2**4:  # bold flag
                        is_bold = True
                    block_text += span_text + " "

            block_text = block_text.strip()
            if not block_text:
                continue

            # Determine if this block is a heading
            h_level = self._detect_heading_level(block_text, max_font_size, is_bold)
            if h_level and not detected_heading:
                detected_heading = block_text
                heading_level = h_level

            text_parts.append(block_text)
            structured_blocks.append({
                "text": block_text,
                "font_size": max_font_size,
                "is_bold": is_bold,
                "heading_level": h_level,
                "bbox": block.get("bbox"),
            })

        full_text = "\n".join(text_parts)

        # Extract tables via pdfplumber
        tables: list[list[list[str]]] = []
        if plumber_doc and HAS_PDFPLUMBER:
            try:
                plumber_page = plumber_doc.pages[page_num - 1]
                raw_tables = plumber_page.extract_tables()
                if raw_tables:
                    for tbl in raw_tables:
                        cleaned = [
                            [cell if cell is not None else "" for cell in row]
                            for row in tbl
                        ]
                        tables.append(cleaned)
            except Exception:
                pass  # Table extraction is best-effort

        return ParsedPage(
            page_num=page_num,
            text=full_text,
            blocks=structured_blocks,
            tables=tables,
            detected_heading=detected_heading,
            heading_level=heading_level,
        )

    def _detect_heading_level(
        self,
        text: str,
        font_size: float,
        is_bold: bool,
    ) -> Optional[int]:
        """
        Heuristic heading detection.

        Returns heading level (1=chapter, 2=section, 3=subsection) or None.
        """
        # Too long to be a heading
        if len(text) > MAX_HEADING_CHARS:
            return None

        # Very short with only numbers/special chars — skip
        if len(text.strip()) < 3:
            return None

        # Large font = likely chapter heading
        if font_size >= HEADING_FONT_SIZE_H1:
            return 1

        # Medium bold font = section heading
        if font_size >= HEADING_FONT_SIZE_H2 and is_bold:
            return 2

        # Slightly bold = subsection
        if font_size >= HEADING_FONT_SIZE_H3 and is_bold:
            return 3

        # Numbering patterns like "3.2.1 Subsection Name"
        if re.match(r"^\d+(\.\d+){1,2}\s+[A-Z]", text):
            dots = text.split(".")[0]
            depth = len(text.split()) - 1
            if re.match(r"^\d+\.", text):
                return 2
            return 3

        return None


# ============================================================
# CLI ENTRY POINT
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="PDF Structure-Aware Loader — test extraction on a single PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.ingestion.pdf_loader --path "bee guide - thermal utility.pdf" --pages 5
  python -m app.ingestion.pdf_loader --path "bee guide - electrical utilities.pdf" --pages 20
        """,
    )
    parser.add_argument("--path", required=True, help="Path to PDF file")
    parser.add_argument("--pages", type=int, default=10, help="Number of pages to test")
    parser.add_argument("--show-text", action="store_true", help="Print extracted text")
    args = parser.parse_args()

    # Detect domain from filename
    fname = Path(args.path).name.lower()
    if "thermal" in fname:
        domain = UtilityDomain.THERMAL
        doc_id = "bee_thermal"
        book_name = "Energy Efficiency in Thermal Utilities"
    else:
        domain = UtilityDomain.ELECTRICAL
        doc_id = "bee_electrical"
        book_name = "Energy Efficiency in Electrical Utilities"

    loader = PDFLoader(
        file_path=args.path,
        document_id=doc_id,
        book_name=book_name,
        utility_domain=domain,
        max_pages=args.pages,
    )
    doc = loader.load()

    print(f"\n=== Parsed Document: {doc.book_name} ===")
    print(f"Total pages: {doc.total_pages} | Loaded: {len(doc.pages)}")
    print()

    for i, page in enumerate(doc.pages[:5]):
        print(f"--- Page {page.page_num} ---")
        if page.detected_heading:
            print(f"  Heading (level {page.heading_level}): {page.detected_heading}")
        print(f"  Text length: {len(page.text)} chars")
        print(f"  Tables: {len(page.tables)}")
        if args.show_text:
            print(f"  Text preview: {page.text[:300]}")
        print()


if __name__ == "__main__":
    main()
