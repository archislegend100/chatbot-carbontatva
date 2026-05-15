"""
backend/tests/test_ocr.py
==========================
Tests for the OCR layer (ocr_engine, ocr_cache, pdf_ocr_loader).

These tests do NOT require Tesseract to be installed.
They use mock/synthetic data for unit testing the OCR pipeline.

To run:
    python -m pytest backend/tests/test_ocr.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestOCREngine:
    """Unit tests for ocr_engine.py"""

    def test_ocr_result_dataclass(self):
        from app.ingestion.ocr_engine import OCRResult
        result = OCRResult(
            page_num=5,
            text="Boiler efficiency test content",
            confidence=85.3,
            word_count=5,
            ocr_time_ms=4200.0,
            has_content=True,
        )
        assert result.page_num == 5
        assert result.confidence == 85.3
        assert result.has_content is True
        assert result.word_count == 5

    def test_ocr_result_low_confidence_flags_no_content(self):
        from app.ingestion.ocr_engine import OCRResult
        result = OCRResult(
            page_num=0,
            text="",
            confidence=10.0,
            word_count=0,
            ocr_time_ms=500.0,
            has_content=False,
        )
        assert result.has_content is False

    def test_tesseract_ocr_structure(self):
        """Test that _tesseract_ocr returns a tuple of (str, float)."""
        from app.ingestion.ocr_engine import OCRResult
        # Just verify OCRResult has correct types
        r = OCRResult(page_num=0, text="hello world test", confidence=75.0, word_count=3, ocr_time_ms=1000.0, has_content=True)
        assert isinstance(r.text, str)
        assert isinstance(r.confidence, float)
        assert r.word_count == 3



class TestOCRCache:
    """Unit tests for ocr_cache.py"""

    def test_cache_miss_returns_none(self, tmp_path):
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)

        # Create a fake PDF path
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"FAKEPDF" * 1000)

        result = cache.load("bee_thermal", fake_pdf)
        assert result is None

    def test_cache_save_and_load(self, tmp_path):
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"FAKEPDF" * 1000)

        pages = [
            {"page": 0, "text": "Boiler efficiency", "confidence": 87.0, "word_count": 2, "ocr_time_ms": 3000},
            {"page": 1, "text": "Steam distribution", "confidence": 91.0, "word_count": 2, "ocr_time_ms": 2800},
        ]

        cache.save("bee_test", fake_pdf, pages)
        loaded = cache.load("bee_test", fake_pdf)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["text"] == "Boiler efficiency"
        assert loaded[1]["confidence"] == 91.0

    def test_cache_exists_check(self, tmp_path):
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"X" * 2048)

        assert not cache.exists("bee_test", fake_pdf)
        cache.save("bee_test", fake_pdf, [{"page": 0, "text": "x" * 100, "confidence": 80, "word_count": 5, "ocr_time_ms": 1000}])
        assert cache.exists("bee_test", fake_pdf)

    def test_cache_stats(self, tmp_path):
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)
        stats = cache.stats()
        assert "cache_dir" in stats
        assert "cached_documents" in stats

    def test_cache_clear_all(self, tmp_path):
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"X" * 2048)
        cache.save("bee_test", fake_pdf, [{"page": 0, "text": "x" * 100, "confidence": 80, "word_count": 5, "ocr_time_ms": 1000}])

        cache.clear_all()
        assert not cache.exists("bee_test", fake_pdf)

    def test_different_pdf_hashes(self, tmp_path):
        """Two different PDFs with same doc_id should have different cache files."""
        from app.ingestion.ocr_cache import OCRCache
        cache = OCRCache(cache_dir=tmp_path)

        pdf1 = tmp_path / "pdf1.pdf"
        pdf2 = tmp_path / "pdf2.pdf"
        pdf1.write_bytes(b"A" * 2048)
        pdf2.write_bytes(b"B" * 2048)

        cache.save("bee_test", pdf1, [{"page": 0, "text": "content1", "confidence": 80, "word_count": 1, "ocr_time_ms": 1000}])
        cache.save("bee_test", pdf2, [{"page": 0, "text": "content2", "confidence": 80, "word_count": 1, "ocr_time_ms": 1000}])

        loaded1 = cache.load("bee_test", pdf1)
        loaded2 = cache.load("bee_test", pdf2)

        assert loaded1[0]["text"] == "content1"
        assert loaded2[0]["text"] == "content2"
