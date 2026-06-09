"""
ingestion/ocr_cache.py
=======================
Per-PDF page-level OCR result cache.

OCR is expensive (~5-15s per page on CPU). This cache persists OCR results
as JSONL files so the ingestion pipeline can be re-run without re-OCR.

Cache format: data/ocr_cache/{doc_id}_{sha256[:8]}.jsonl
Each line is one page:
  {"page": 0, "text": "...", "confidence": 87.3, "word_count": 412, "ocr_time_ms": 4200}

Cache invalidation:
- Based on PDF SHA-256 hash (embedded in filename)
- Re-OCR only when PDF changes or --force-ocr flag is set

Usage:
    cache = OCRCache()
    results = cache.load("bee_thermal")      # Returns list[dict] or None
    cache.save("bee_thermal", results_list)   # Saves to disk
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from app.config.settings import settings
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

CACHE_DIR = Path(str(settings.BASE_DIR)) / "data" / "ocr_cache"


class OCRCache:
    """Manages page-level OCR result persistence."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _pdf_hash(self, pdf_path: Path) -> str:
        """Compute SHA-256 of first 2MB of PDF (fast fingerprint)."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            h.update(f.read(2 * 1024 * 1024))
        return h.hexdigest()[:12]

    def _cache_path(self, doc_id: str, pdf_path: Path) -> Path:
        pdf_hash = self._pdf_hash(pdf_path)
        return self.cache_dir / f"{doc_id}_{pdf_hash}.jsonl"

    def exists(self, doc_id: str, pdf_path: Path) -> bool:
        """Check if a valid cache exists for this PDF."""
        p = self._cache_path(doc_id, pdf_path)
        return p.exists() and p.stat().st_size > 100

    def load(self, doc_id: str, pdf_path: Path) -> Optional[list[dict]]:
        """
        Load cached OCR results.
        
        Returns:
            List of page dicts, or None if cache miss.
        """
        path = self._cache_path(doc_id, pdf_path)
        if not path.exists():
            return None

        pages = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pages.append(json.loads(line))
            logger.info(f"OCR cache hit: {doc_id} — {len(pages)} pages loaded")
            return pages
        except Exception as e:
            logger.warning(f"OCR cache corrupt for {doc_id}: {e}")
            return None

    def save(self, doc_id: str, pdf_path: Path, pages: list[dict]) -> None:
        """
        Save OCR results to disk as JSONL.
        
        Args:
            doc_id: Document identifier
            pdf_path: Path to source PDF (for hash)
            pages: List of page result dicts
        """
        path = self._cache_path(doc_id, pdf_path)
        with open(path, "w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        size_kb = path.stat().st_size // 1024
        logger.info(f"OCR cache saved: {doc_id} → {path.name} ({size_kb}KB, {len(pages)} pages)")

    def clear(self, doc_id: str, pdf_path: Path) -> None:
        """Delete cache for a specific document."""
        path = self._cache_path(doc_id, pdf_path)
        if path.exists():
            path.unlink()
            logger.info(f"OCR cache cleared: {path.name}")

    def clear_all(self) -> None:
        """Delete all OCR caches."""
        for f in self.cache_dir.glob("*.jsonl"):
            f.unlink()
        logger.info("All OCR caches cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        files = list(self.cache_dir.glob("*.jsonl"))
        total_pages = 0
        for f in files:
            try:
                with open(f) as fh:
                    total_pages += sum(1 for _ in fh if _.strip())
            except Exception:
                pass
        return {
            "cache_dir": str(self.cache_dir),
            "cached_documents": len(files),
            "total_pages_cached": total_pages,
            "files": [f.name for f in files],
        }
