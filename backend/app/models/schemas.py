"""
models/schemas.py
=================
Pydantic data models (schemas) used throughout the copilot backend.

These schemas define:
- The internal document/chunk representation
- API request/response shapes
- Tool output structures

Keeping all models in one place makes it easy to understand
the full data flow of the system.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ============================================================
# ENUMS
# ============================================================


class UtilityDomain(str, Enum):
    """Which BEE manual does a chunk / query belong to?"""
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Granularity level of a chunk in the knowledge base."""
    SECTION = "section"        # ~1000-1500 tok — whole section
    SEMANTIC = "semantic"      # ~300-500 tok  — fine semantic unit
    TABLE = "table"            # extracted table content
    LIST = "list"              # checklist / bullet list
    FORMULA = "formula"        # equation or formula block
    FIGURE = "figure"          # figure caption + description


class ContentType(str, Enum):
    """Nature of the chunk content."""
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    FORMULA = "formula"
    FIGURE = "figure"
    MIXED = "mixed"


class ToolMode(str, Enum):
    """Which tool / mode is being used for this query?"""
    QA = "qa"
    EXPLAINER = "explainer"
    TROUBLESHOOT = "troubleshoot"
    OPPORTUNITY = "opportunity"
    COMPARISON = "comparison"
    NAVIGATION = "navigation"
    CHECKLIST = "checklist"
    SUMMARIZE = "summarize"
    AUTO = "auto"    # Let the router classify


class ExplanationLevel(str, Enum):
    """Verbosity/depth for explanations."""
    BEGINNER = "beginner"
    ENGINEER = "engineer"


# ============================================================
# DOCUMENT / CHUNK MODELS (Internal)
# ============================================================


class ChunkMetadata(BaseModel):
    """
    Rich metadata attached to every chunk in the knowledge base.
    This is stored alongside the chunk text in the vector store
    and used for metadata filtering during retrieval.
    """
    chunk_id: str                          # Unique stable ID
    document_id: str                       # "bee_thermal" | "bee_electrical"
    book_name: str                         # Full book title
    utility_domain: UtilityDomain

    chapter_num: Optional[int] = None
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None

    page_start: int = 0
    page_end: int = 0

    chunk_type: ChunkType = ChunkType.SEMANTIC
    content_type: ContentType = ContentType.TEXT

    equipment_tags: list[str] = Field(default_factory=list)
    concept_tags: list[str] = Field(default_factory=list)

    word_count: int = 0
    char_count: int = 0
    ocr_confidence: Optional[float] = None   # 0–1, if OCR was used


class DocumentChunk(BaseModel):
    """A single chunk of text with its metadata, as stored in the KB."""
    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata.model_dump()}


class ParsedPage(BaseModel):
    """Represents a single parsed PDF page with structure information."""
    page_num: int
    text: str
    blocks: list[dict[str, Any]] = Field(default_factory=list)  # Raw fitz blocks
    tables: list[list[list[str]]] = Field(default_factory=list)  # Extracted tables
    detected_heading: Optional[str] = None
    heading_level: Optional[int] = None    # 1=chapter, 2=section, 3=subsection


class ParsedDocument(BaseModel):
    """A fully parsed PDF document before chunking."""
    document_id: str
    book_name: str
    utility_domain: UtilityDomain
    file_path: str
    total_pages: int
    pages: list[ParsedPage] = Field(default_factory=list)


# ============================================================
# API REQUEST MODELS
# ============================================================


class QueryRequest(BaseModel):
    """
    Primary query request — the main API endpoint receives this.
    
    Example:
        {
          "query": "What is the optimum excess air for a coal boiler?",
          "tool_mode": "auto",
          "domain_filter": "thermal",
          "explanation_level": "engineer"
        }
    """
    query: str = Field(..., min_length=3, max_length=2000,
                       description="User's question or request")
    tool_mode: ToolMode = Field(default=ToolMode.AUTO,
                                description="Which tool to use. AUTO = let the router decide.")
    domain_filter: Optional[UtilityDomain] = Field(
        default=None,
        description="Filter retrieval to only thermal or only electrical content."
    )
    explanation_level: ExplanationLevel = Field(
        default=ExplanationLevel.ENGINEER,
        description="Verbosity level for explanations."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20,
                                 description="Override number of chunks to retrieve.")


class ClassifyRequest(BaseModel):
    """Request to classify a query without generating an answer."""
    query: str = Field(..., min_length=3)


class RetrieveRequest(BaseModel):
    """Request for raw retrieval without answer generation."""
    query: str = Field(..., min_length=3)
    domain_filter: Optional[UtilityDomain] = None
    top_k: int = Field(default=8, ge=1, le=30)
    chunk_types: Optional[list[ChunkType]] = None


# ============================================================
# API RESPONSE MODELS
# ============================================================


class SourceCitation(BaseModel):
    """A single citation reference included in an answer."""
    chunk_id: str
    book_name: str
    utility_domain: str
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    page_start: int
    page_end: int
    relevance_score: float = 0.0
    snippet: str = ""   # Short preview of the chunk text


class ClassificationResult(BaseModel):
    """Result of the query classifier."""
    tool_mode: ToolMode
    utility_domain: Optional[UtilityDomain]
    equipment_tags: list[str] = Field(default_factory=list)
    concept_tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None


class AnswerResponse(BaseModel):
    """
    Full structured answer response returned by the API.
    
    The 'answer' field contains the main generated answer.
    The 'structured_sections' field contains tool-specific
    structured content (e.g., checklist items, comparison table).
    """
    query: str
    tool_mode: ToolMode
    answer: str

    # Structured tool-specific output
    structured_sections: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific structured data (checklists, comparisons, etc.)"
    )

    citations: list[SourceCitation] = Field(default_factory=list)
    classification: Optional[ClassificationResult] = None

    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions or actions."
    )

    # Telemetry
    retrieval_count: int = 0
    generation_model: str = ""
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    index_loaded: bool
    embedding_model: str
    llm_provider: str
    chunk_count: int


class IngestStatusResponse(BaseModel):
    """Status after triggering ingestion."""
    status: str
    message: str
    chunks_created: int = 0
    thermal_chunks: int = 0
    electrical_chunks: int = 0
