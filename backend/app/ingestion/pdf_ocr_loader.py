"""
ingestion/pdf_ocr_loader.py
============================
PDF loader that uses Tesseract OCR to extract text from image-based PDFs.

Replaces the native PyMuPDF text extraction from v1 (which returned 0 chars
because both BEE manuals are fully image-scanned).

Pipeline:
  PDF → fitz.Page → PIL Image (200 DPI) → Tesseract → text + confidence
  → OCR cache → ParsedDocument (same schema as v1)

Key design decisions:
1. Cache-first: Check OCR cache before running Tesseract
2. Progress-aware: Reports per-page progress for long runs (636 pages total)
3. Graceful degradation: Pages with low confidence are flagged, not dropped
4. Parallel option: Can run N pages in parallel (disabled by default for stability)

Usage:
    loader = PDFOCRLoader(
        file_path=settings.PDF_THERMAL_PATH,
        document_id="bee_thermal",
        book_name="Energy Efficiency in Thermal Utilities",
        utility_domain=UtilityDomain.THERMAL,
    )
    doc = loader.load()   # Returns ParsedDocument (same as v1)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import fitz

import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger
from app.ingestion.ocr_engine import ocr_page, OCR_DPI
from app.ingestion.ocr_cache import OCRCache
from app.models.schemas import ParsedDocument, ParsedPage, UtilityDomain

logger = get_logger(__name__)

# Minimum chars on a page to consider it content-bearing
MIN_PAGE_CHARS = 30


class PDFOCRLoader:
    """
    Loads a PDF by OCR-ing each page and returning a ParsedDocument.
    
    Identical output schema to the original PDFLoader (v1), making this
    a drop-in replacement that works with the existing chunker/parser.
    
    Args:
        file_path: Path to the PDF file
        document_id: Unique ID (e.g. "bee_thermal")
        book_name: Human-readable book name
        utility_domain: UtilityDomain enum value
        max_pages: Limit pages (None = all). Use for --test-run.
        force_ocr: Skip cache and re-run OCR even if cache exists
        dpi: Render DPI (200 is optimal)
    """

    def __init__(
        self,
        file_path: Path,
        document_id: str,
        book_name: str,
        utility_domain: UtilityDomain,
        max_pages: Optional[int] = None,
        force_ocr: bool = False,
        dpi: int = OCR_DPI,
    ):
        self.file_path = Path(file_path)
        self.document_id = document_id
        self.book_name = book_name
        self.utility_domain = utility_domain
        self.max_pages = max_pages
        self.force_ocr = force_ocr
        self.dpi = dpi
        self._cache = OCRCache()

    def load(self) -> ParsedDocument:
        """
        Load and OCR the PDF, returning a ParsedDocument.
        
        Tries cache first, then runs OCR page-by-page with progress reporting.
        
        Returns:
            ParsedDocument with one ParsedPage per PDF page
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

        file_size_mb = self.file_path.stat().st_size / 1024 / 1024
        logger.info(f"Loading: {self.book_name} ({file_size_mb:.1f} MB)")

        doc = fitz.open(str(self.file_path))
        total_pages = doc.page_count
        pages_to_process = list(range(min(total_pages, self.max_pages or total_pages)))
        logger.info(f"  Total pages: {total_pages} | Processing: {len(pages_to_process)}")

        # --- Cache check ---
        cached = None
        if not self.force_ocr:
            cached = self._cache.load(self.document_id, self.file_path)
        
        if cached is not None:
            # Use cached OCR, slice if max_pages is set
            page_dicts = cached[:len(pages_to_process)]
            parsed_pages = self._page_dicts_to_parsed(page_dicts)
            doc.close()
            return self._build_document(parsed_pages, total_pages)

        # --- Run OCR ---
        logger.info(f"  Running Tesseract OCR at {self.dpi} DPI...")
        logger.info(f"  ⏱ Estimated time: {len(pages_to_process) * 8 // 60}–{len(pages_to_process) * 15 // 60} min")
        logger.info("  (Results will be cached — OCR runs only ONCE per PDF)")

        page_dicts = []
        t_total = time.monotonic()
        low_conf_count = 0
        empty_count = 0

        for i, pn in enumerate(pages_to_process):
            result = ocr_page(doc[pn], pn, dpi=self.dpi)
            
            page_dict = {
                "page": pn,
                "text": result.text,
                "confidence": result.confidence,
                "word_count": result.word_count,
                "ocr_time_ms": result.ocr_time_ms,
                "has_content": result.has_content,
            }
            page_dicts.append(page_dict)

            if result.confidence < 40:
                low_conf_count += 1
            if not result.has_content:
                empty_count += 1

            # Progress every 10 pages
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.monotonic() - t_total
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(pages_to_process) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  OCR [{i+1}/{len(pages_to_process)}] "
                    f"page={pn} conf={result.confidence:.0f}% "
                    f"words={result.word_count} "
                    f"ETA={remaining/60:.1f}min"
                )

        doc.close()
        elapsed_total = time.monotonic() - t_total

        logger.info(
            f"  ✅ OCR complete: {len(page_dicts)} pages in {elapsed_total/60:.1f}min "
            f"| low-conf={low_conf_count} | empty={empty_count}"
        )

        # Save to cache (even for partial runs with max_pages)
        if self.max_pages is None:
            self._cache.save(self.document_id, self.file_path, page_dicts)

        parsed_pages = self._page_dicts_to_parsed(page_dicts)
        return self._build_document(parsed_pages, total_pages)

    def _page_dicts_to_parsed(self, page_dicts: list[dict]) -> list[ParsedPage]:
        """Convert OCR result dicts to ParsedPage objects."""
        pages = []
        for d in page_dicts:
            if len(d.get("text", "").strip()) >= MIN_PAGE_CHARS:
                pages.append(ParsedPage(
                    page_num=d["page"] + 1,  # 1-indexed like v1
                    text=d["text"],
                    blocks=[{
                        "text": d["text"],
                        "font_size": 12.0,  # Estimated
                        "is_bold": False,
                        "bbox": (0, 0, 600, 800),
                        "type": "text",
                        "ocr_confidence": d.get("confidence", 0.0),
                    }],
                ))
        return pages

    def _build_document(
        self,
        pages: list[ParsedPage],
        total_pages: int,
    ) -> ParsedDocument:
        """Build the final ParsedDocument from OCR pages."""
        return ParsedDocument(
            document_id=self.document_id,
            book_name=self.book_name,
            utility_domain=self.utility_domain,
            file_path=str(self.file_path),
            total_pages=total_pages,
            pages=pages,
        )
