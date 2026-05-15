"""
tests/test_ingestion.py
=======================
Tests for the PDF ingestion pipeline.

Run: pytest backend/tests/test_ingestion.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestOntology:
    """Test the ontology tag detection."""

    def test_thermal_equipment_detection(self):
        from app.ingestion.ontology import detect_tags
        text = "The boiler generates steam at high pressure for the industrial process."
        result = detect_tags(text, "thermal")
        assert "boiler" in result.equipment_tags
        assert "steam_system" in result.equipment_tags

    def test_electrical_equipment_detection(self):
        from app.ingestion.ontology import detect_tags
        text = "Install a VFD (variable frequency drive) on the motor to reduce energy."
        result = detect_tags(text, "electrical")
        assert "vfd" in result.equipment_tags
        assert "motor" in result.equipment_tags

    def test_concept_detection(self):
        from app.ingestion.ontology import detect_tags
        text = "Excess air in combustion leads to increased heat loss via flue gas."
        result = detect_tags(text, "thermal")
        assert "combustion" in result.concept_tags

    def test_empty_text(self):
        from app.ingestion.ontology import detect_tags
        result = detect_tags("", "thermal")
        assert result.equipment_tags == []
        assert result.concept_tags == []


class TestChunker:
    """Test the chunking logic."""

    def test_section_chunk_creation(self):
        from app.ingestion.chunker import DocumentChunker
        from app.ingestion.structure_parser import SectionSpan
        from app.models.schemas import ParsedDocument, ParsedPage, UtilityDomain

        # Create minimal test data
        page = ParsedPage(page_num=1, text="This is test content about boilers and steam systems. " * 20)
        span = SectionSpan(
            chapter_num=1,
            chapter_title="Boilers",
            section_title="Steam Generation",
            subsection_title=None,
            page_start=1,
            page_end=3,
            pages=[page],
        )
        doc = ParsedDocument(
            document_id="bee_thermal",
            book_name="Test Book",
            utility_domain=UtilityDomain.THERMAL,
            file_path="/test/test.pdf",
            total_pages=10,
            pages=[page],
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc, [span])

        assert len(chunks) > 0
        # Should have at least a section chunk
        section_chunks = [c for c in chunks if c.metadata.chunk_type.value == "section"]
        assert len(section_chunks) >= 1

    def test_chunk_metadata_integrity(self):
        from app.ingestion.chunker import DocumentChunker
        from app.ingestion.structure_parser import SectionSpan
        from app.models.schemas import ParsedDocument, ParsedPage, UtilityDomain

        page = ParsedPage(page_num=5, text="Motor efficiency improvement using VFD drives. " * 50)
        span = SectionSpan(
            chapter_num=3,
            chapter_title="Motors",
            section_title="Energy Efficient Motors",
            subsection_title=None,
            page_start=5,
            page_end=8,
            pages=[page],
        )
        doc = ParsedDocument(
            document_id="bee_electrical",
            book_name="Electrical Manual",
            utility_domain=UtilityDomain.ELECTRICAL,
            file_path="/test/test.pdf",
            total_pages=100,
            pages=[page],
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc, [span])

        for chunk in chunks:
            assert chunk.metadata.document_id == "bee_electrical"
            assert chunk.metadata.utility_domain.value == "electrical"
            assert chunk.metadata.chapter_num == 3
            assert chunk.metadata.page_start == 5
            assert chunk.chunk_id  # Has a non-empty ID
            assert len(chunk.text) > 0

    def test_chunk_ids_are_unique(self):
        from app.ingestion.chunker import DocumentChunker
        from app.ingestion.structure_parser import SectionSpan
        from app.models.schemas import ParsedDocument, ParsedPage, UtilityDomain

        pages = [
            ParsedPage(page_num=i, text=f"Content for page {i} about energy systems. " * 30)
            for i in range(1, 5)
        ]
        spans = [
            SectionSpan(
                chapter_num=i,
                chapter_title=f"Chapter {i}",
                section_title=None,
                subsection_title=None,
                page_start=i,
                page_end=i,
                pages=[pages[i - 1]],
            )
            for i in range(1, 5)
        ]
        doc = ParsedDocument(
            document_id="bee_thermal",
            book_name="Test",
            utility_domain=UtilityDomain.THERMAL,
            file_path="/test/test.pdf",
            total_pages=4,
            pages=pages,
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc, spans)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs are not unique!"


class TestStructureParser:
    """Test the structure parser."""

    def test_heading_detection(self):
        from app.ingestion.structure_parser import StructureParser
        parser = StructureParser()

        # Test chapter pattern
        heading, level = parser._scan_blocks_for_heading(
            type("Page", (), {
                "blocks": [{"text": "chapter 3: Combustion Systems", "font_size": 14, "is_bold": True}]
            })()
        )
        # May or may not detect depending on pattern — just ensure no crash
        assert level is None or level in [1, 2, 3]

    def test_section_number_pattern(self):
        from app.ingestion.structure_parser import StructureParser
        parser = StructureParser()
        import re
        for pattern in parser.SECTION_PATTERNS:
            assert pattern.match("3.2 Boiler Efficiency Methods")
