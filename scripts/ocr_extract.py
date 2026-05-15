#!/usr/bin/env python3
"""
scripts/ocr_extract.py
=======================
Standalone CLI tool to run Tesseract OCR on the BEE PDFs.

This is a separate utility from ingest.py.
Use it to:
  - Preview OCR quality on a few pages
  - Extract OCR text to a plain text file for inspection
  - Rebuild the OCR cache independently of the index

Usage:
    # Preview OCR on first 5 pages of thermal PDF
    python scripts/ocr_extract.py --thermal --pages 0-4

    # Extract all pages of electrical PDF and save to text file
    python scripts/ocr_extract.py --electrical --output data/electrical_ocr.txt

    # Test OCR quality: show confidence and word count per page
    python scripts/ocr_extract.py --thermal --pages 10-15 --quality-report

    # Rebuild just the OCR cache (without rebuilding the index)
    python scripts/ocr_extract.py --thermal --electrical --rebuild-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.ingestion.ocr_engine import ocr_page, OCR_DPI
from app.ingestion.ocr_cache import OCRCache
import fitz

setup_logging()
logger = get_logger("ocr_extract")


def parse_page_range(spec: str, max_pages: int) -> list[int]:
    """Parse page range like '0-9' or '5' or '0,3,10-15'."""
    pages = set()
    for part in spec.split(','):
        if '-' in part:
            start, end = part.split('-', 1)
            pages.update(range(int(start), min(int(end) + 1, max_pages)))
        else:
            p = int(part)
            if p < max_pages:
                pages.add(p)
    return sorted(pages)


def run_ocr_extract(
    pdf_path: Path,
    doc_id: str,
    page_range: list[int] | None,
    output_path: Path | None,
    quality_report: bool,
    rebuild_cache: bool,
    dpi: int,
) -> None:
    cache = OCRCache()

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    total = doc.page_count
    pages = page_range if page_range else list(range(total))
    logger.info(f"PDF: {pdf_path.name} | {total} pages total | Processing: {len(pages)}")

    force = rebuild_cache or bool(page_range)

    # Check cache
    if not force and cache.exists(doc_id, pdf_path):
        logger.info("Using cached OCR results. Use --rebuild-cache to re-run OCR.")
        cached = cache.load(doc_id, pdf_path) or []
        results = [c for c in cached if c["page"] in pages]
    else:
        logger.info(f"Running OCR at {dpi} DPI on {len(pages)} pages...")
        results = []
        t_total = time.monotonic()

        for i, pn in enumerate(pages):
            result = ocr_page(doc[pn], pn, dpi=dpi)
            d = {
                "page": pn,
                "text": result.text,
                "confidence": result.confidence,
                "word_count": result.word_count,
                "ocr_time_ms": result.ocr_time_ms,
            }
            results.append(d)

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.monotonic() - t_total
                logger.info(
                    f"  [{i+1}/{len(pages)}] page={pn} "
                    f"conf={result.confidence:.0f}% "
                    f"words={result.word_count} "
                    f"time={result.ocr_time_ms:.0f}ms"
                )

        if rebuild_cache and not page_range:
            cache.save(doc_id, pdf_path, results)
            logger.info(f"Cache saved: {doc_id}")

    doc.close()

    # Quality report
    if quality_report:
        confs = [r["confidence"] for r in results]
        words = [r["word_count"] for r in results]
        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_words = sum(words) / len(words) if words else 0
        low_conf = [r for r in results if r["confidence"] < 50]
        empty = [r for r in results if r["word_count"] < 10]

        print(f"\n{'='*50}")
        print(f"OCR Quality Report: {pdf_path.name}")
        print(f"{'='*50}")
        print(f"Pages analyzed:    {len(results)}")
        print(f"Avg confidence:    {avg_conf:.1f}%")
        print(f"Avg word count:    {avg_words:.0f}")
        print(f"Low conf pages:    {len(low_conf)} (<50%)")
        print(f"Near-empty pages:  {len(empty)} (<10 words)")
        print(f"{'='*50}")
        print("\nPer-page breakdown:")
        for r in results[:20]:
            bar = '█' * int(r['confidence'] / 10) + '░' * (10 - int(r['confidence'] / 10))
            print(f"  p{r['page']:3d}  [{bar}] {r['confidence']:5.1f}%  {r['word_count']:4d} words")

    # Save to text file
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"PAGE {r['page'] + 1}  [conf={r['confidence']:.0f}% | words={r['word_count']}]\n")
                f.write(f"{'='*60}\n\n")
                f.write(r['text'])
                f.write('\n')
        size_kb = output_path.stat().st_size // 1024
        logger.info(f"Saved OCR text: {output_path} ({size_kb} KB)")

    # Print preview
    if not output_path and not quality_report:
        for r in results[:3]:
            print(f"\n--- Page {r['page'] + 1} (conf={r['confidence']:.0f}%) ---")
            print(r['text'][:500])
            if len(r['text']) > 500:
                print("... [truncated]")


def main():
    parser = argparse.ArgumentParser(description="BEE PDF OCR Extractor")
    parser.add_argument("--thermal", action="store_true", help="Process thermal PDF")
    parser.add_argument("--electrical", action="store_true", help="Process electrical PDF")
    parser.add_argument("--pages", help="Page range (e.g. 0-9, 5, 0,3,10-15). Default: all")
    parser.add_argument("--output", help="Save extracted text to file")
    parser.add_argument("--quality-report", action="store_true", help="Show per-page quality stats")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force re-OCR and update cache")
    parser.add_argument("--dpi", type=int, default=OCR_DPI, help=f"Render DPI (default: {OCR_DPI})")
    args = parser.parse_args()

    if not args.thermal and not args.electrical:
        args.thermal = True
        args.electrical = True

    docs = []
    if args.thermal:
        docs.append(("bee_thermal", settings.PDF_THERMAL_PATH, "thermal"))
    if args.electrical:
        docs.append(("bee_electrical", settings.PDF_ELECTRICAL_PATH, "electrical"))

    for doc_id, pdf_path, name in docs:
        doc = fitz.open(str(pdf_path))
        total = doc.page_count
        doc.close()

        page_range = parse_page_range(args.pages, total) if args.pages else None
        output = Path(args.output.replace("{domain}", name)) if args.output else None

        run_ocr_extract(
            pdf_path=pdf_path,
            doc_id=doc_id,
            page_range=page_range,
            output_path=output,
            quality_report=args.quality_report,
            rebuild_cache=args.rebuild_cache,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
