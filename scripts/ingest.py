#!/usr/bin/env python3
"""
scripts/ingest.py
=================
Full ingestion pipeline (v2 — OCR-powered) for the Industrial Energy Efficiency Copilot.

Both BEE PDFs are fully image-scanned (0 chars from native extraction).
This script uses Tesseract OCR to extract text before indexing.

Pipeline:
  1. OCR each PDF page (Tesseract, 200 DPI) → cached to data/ocr_cache/
  2. Parse structure (chapters/sections)
  3. Multi-granular chunking
  4. Dense embedding (all-MiniLM-L6-v2, local)
  5. Build ChromaDB vector index
  6. Build BM25 sparse index

Usage:
    python scripts/ingest.py                    # Full ingestion (both PDFs)
    python scripts/ingest.py --test-run         # First 20 pages per PDF (~5 min)
    python scripts/ingest.py --max-pages 50     # Custom page limit
    python scripts/ingest.py --thermal-only     # Only thermal PDF
    python scripts/ingest.py --electrical-only  # Only electrical PDF
    python scripts/ingest.py --force-rebuild    # Rebuild index (reuse OCR cache)
    python scripts/ingest.py --force-ocr        # Re-run OCR (slow, ~60 min)

OCR Cache:
    OCR results are stored in data/ocr_cache/ and reused on subsequent runs.
    The cache is invalidated automatically when the PDF file changes.

Prerequisites:
    sudo apt-get install tesseract-ocr tesseract-ocr-eng
    pip install pytesseract
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add backend/ to Python path so we can import app modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.ingestion.pdf_ocr_loader import PDFOCRLoader   # v2: OCR-based loader
from app.ingestion.structure_parser import StructureParser
from app.ingestion.chunker import DocumentChunker
from app.indexing.embedder import Embedder
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.models.schemas import UtilityDomain, DocumentChunk

setup_logging()
logger = get_logger("ingest")


# ============================================================
# DOCUMENT DEFINITIONS
# ============================================================

DOCUMENTS = [
    {
        "file_path": settings.PDF_THERMAL_PATH,
        "document_id": "bee_thermal",
        "book_name": "Energy Efficiency in Thermal Utilities",
        "utility_domain": UtilityDomain.THERMAL,
    },
    {
        "file_path": settings.PDF_ELECTRICAL_PATH,
        "document_id": "bee_electrical",
        "book_name": "Energy Efficiency in Electrical Utilities",
        "utility_domain": UtilityDomain.ELECTRICAL,
    },
]


# ============================================================
# PIPELINE STEPS
# ============================================================


def load_and_parse(
    doc_def: dict,
    max_pages: int | None,
    force_ocr: bool = False,
) -> list[DocumentChunk]:
    """
    Run OCR + structure parse + chunking.
    Uses PDFOCRLoader (v2) which caches OCR results to avoid re-processing.
    """
    file_path = doc_def["file_path"]
    doc_id = doc_def["document_id"]
    book_name = doc_def["book_name"]
    domain = doc_def["utility_domain"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {book_name}")
    logger.info(f"File: {file_path}")
    if not Path(file_path).exists():
        logger.error(f"PDF not found: {file_path}")
        logger.error("Make sure the PDFs are in the project root directory.")
        raise FileNotFoundError(f"PDF not found: {file_path}")

    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024
    logger.info(f"File size: {file_size_mb:.1f} MB")

    # Step 1: OCR Load (uses cache if available)
    t0 = time.time()
    loader = PDFOCRLoader(
        file_path=file_path,
        document_id=doc_id,
        book_name=book_name,
        utility_domain=domain,
        max_pages=max_pages,
        force_ocr=force_ocr,
        dpi=settings.OCR_DPI,
    )
    parsed_doc = loader.load()
    logger.info(f"OCR load: {time.time() - t0:.1f}s | {len(parsed_doc.pages)} pages with content")

    # Step 2: Parse structure
    t0 = time.time()
    parser = StructureParser()
    spans = parser.parse(parsed_doc)
    logger.info(f"Structure parse: {time.time() - t0:.1f}s | {len(spans)} spans")

    # Step 3: Chunk
    t0 = time.time()
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(parsed_doc, spans)
    logger.info(f"Chunking: {time.time() - t0:.1f}s | {len(chunks)} chunks")

    return chunks


def save_chunks_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    """Save all chunks as JSONL for inspection and debugging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Chunks saved to {output_path} ({size_mb:.1f} MB)")


def build_vector_index(
    chunks: list[DocumentChunk],
    embedder: Embedder,
    vector_store: VectorStore,
) -> None:
    """Generate embeddings and add to ChromaDB."""
    logger.info(f"\nGenerating embeddings for {len(chunks)} chunks...")
    t0 = time.time()

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts, show_progress=True)

    logger.info(f"Embedding: {time.time() - t0:.1f}s")

    t0 = time.time()
    vector_store.add_chunks(chunks, embeddings)
    logger.info(f"ChromaDB indexing: {time.time() - t0:.1f}s")


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline(
    documents: list[dict],
    max_pages: int | None,
    force_rebuild: bool,
    force_ocr: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.

    Returns:
        Summary statistics dict.
    """
    total_start = time.time()
    stats = {
        "total_chunks": 0,
        "thermal_chunks": 0,
        "electrical_chunks": 0,
        "documents_processed": 0,
    }

    # Initialize components
    logger.info("\nInitializing components...")
    embedder = Embedder()
    vector_store = VectorStore()
    bm25_index = BM25Index()

    # Reset index if force-rebuild
    if force_rebuild:
        logger.warning("Force rebuild — resetting existing index...")
        vector_store.reset()

    # Check if already indexed and not forcing rebuild
    existing_count = vector_store.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            f"Index already exists with {existing_count} chunks. "
            "Use --force-rebuild to rebuild. Skipping."
        )
        return stats

    # Process each document
    all_chunks: list[DocumentChunk] = []

    for doc_def in documents:
        try:
            chunks = load_and_parse(doc_def, max_pages, force_ocr=force_ocr)
            all_chunks.extend(chunks)
            stats["documents_processed"] += 1

            domain_key = f"{doc_def['utility_domain'].value}_chunks"
            stats[domain_key] = len(chunks)

            logger.info(f"✅ {doc_def['book_name']}: {len(chunks)} chunks")
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        except Exception as e:
            logger.error(f"Failed to process {doc_def['book_name']}: {e}", exc_info=True)
            continue

    if not all_chunks:
        logger.error("No chunks created. Check PDF files and try again.")
        return stats

    stats["total_chunks"] = len(all_chunks)
    logger.info(f"\n{'='*60}")
    logger.info(f"Total chunks to index: {len(all_chunks)}")

    # Save JSONL for debugging
    chunks_path = settings.CHUNKS_CACHE_PATH_ABS
    logger.info(f"\nSaving chunks to JSONL...")
    save_chunks_jsonl(all_chunks, chunks_path)

    # Build vector index
    logger.info("\nBuilding ChromaDB vector index...")
    build_vector_index(all_chunks, embedder, vector_store)

    # Build BM25 index
    logger.info("\nBuilding BM25 sparse index...")
    t0 = time.time()
    bm25_index.build(all_chunks)
    bm25_index.save()
    logger.info(f"BM25 build: {time.time() - t0:.1f}s")

    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ INGESTION COMPLETE")
    logger.info(f"   Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"   Total chunks: {stats['total_chunks']}")
    logger.info(f"   ChromaDB: {vector_store.count()} chunks")
    logger.info(f"   BM25: {bm25_index.chunk_count} chunks")
    logger.info(f"{'='*60}")

    return stats


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Energy Efficiency Copilot — Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run on a small subset of pages per PDF (for testing)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to process per PDF",
    )
    parser.add_argument(
        "--thermal-only",
        action="store_true",
        help="Only ingest the thermal manual",
    )
    parser.add_argument(
        "--electrical-only",
        action="store_true",
        help="Only ingest the electrical manual",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Reset and rebuild the entire index (reuse OCR cache)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Re-run Tesseract OCR even if cache exists (slow: ~60-90 min)",
    )

    args = parser.parse_args()

    # Determine which documents to process
    docs = DOCUMENTS.copy()
    if args.thermal_only:
        docs = [d for d in docs if d["utility_domain"] == UtilityDomain.THERMAL]
    elif args.electrical_only:
        docs = [d for d in docs if d["utility_domain"] == UtilityDomain.ELECTRICAL]

    max_pages = args.max_pages
    if args.test_run:
        max_pages = 30
        logger.info("TEST RUN MODE: Processing first 30 pages per PDF")

    logger.info(f"Documents to process: {[d['book_name'] for d in docs]}")
    logger.info(f"Max pages per PDF: {max_pages or 'ALL'}")

    stats = run_pipeline(
        documents=docs,
        max_pages=max_pages,
        force_rebuild=args.force_rebuild,
        force_ocr=args.force_ocr,
    )

    if stats["total_chunks"] > 0:
        logger.info("\n✅ Ingestion successful. You can now start the backend.")
        logger.info("   Run: uvicorn app.main:app --reload")
    else:
        logger.error("\n❌ Ingestion failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
