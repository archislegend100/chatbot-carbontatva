"""
api/query.py
============
Primary query and retrieval API endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    AnswerResponse,
    ClassificationResult,
    QueryRequest,
    ClassifyRequest,
    RetrieveRequest,
)
from app.services import get_service
from app.core.logging import get_logger

router = APIRouter(prefix="/api", tags=["Query"])
logger = get_logger(__name__)


@router.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    """
    Primary endpoint — classify the query, retrieve evidence,
    rerank, and generate a grounded answer.
    
    This is the main endpoint used by the frontend.
    """
    svc = get_service()

    if not svc.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base indexes are not loaded. "
                "Please run ingestion first: python scripts/ingest.py"
            ),
        )

    try:
        response = await svc.tool_router.handle(request)
        return response
    except Exception as e:
        logger.error(f"Query handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/classify", response_model=ClassificationResult)
async def classify(request: ClassifyRequest):
    """
    Classify a query without generating an answer.
    Useful for debugging the router and for frontend UX.
    """
    svc = get_service()
    if not svc.classifier:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = svc.classifier.classify(request.query)
    return result


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    Raw retrieval endpoint — returns evidence chunks without generating an answer.
    Useful for testing retrieval quality and debugging.
    """
    svc = get_service()

    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="Indexes not loaded")

    try:
        domain = request.domain_filter.value if request.domain_filter else None
        chunk_types = [ct.value for ct in request.chunk_types] if request.chunk_types else None

        candidates = svc.hybrid_retriever.retrieve(
            query=request.query,
            total_top_k=request.top_k,
            domain_filter=domain,
            chunk_types=chunk_types,
        )

        # Rerank
        top = svc.reranker.rerank(
            query=request.query,
            candidates=candidates,
            top_k=request.top_k,
        )

        # Format output
        return {
            "query": request.query,
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "score": c.get("rerank_score", c.get("rrf_score", c.get("score", 0.0))),
                    "retrieval_source": c.get("retrieval_source", "unknown"),
                }
                for c in top
            ],
            "total_candidates": len(candidates),
        }
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Fetch a specific chunk by ID (for source inspectionfrom the frontend)."""
    svc = get_service()
    if not svc.vector_store:
        raise HTTPException(status_code=503, detail="Service not initialized")

    chunk = svc.vector_store.get_chunk_by_id(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

    return chunk


@router.get("/sources")
async def list_sources():
    """
    List all source documents and their structure.
    Used by the frontend to populate the domain/navigation panel.
    """
    svc = get_service()
    if not svc.is_loaded:
        return {"sources": [], "message": "Index not loaded"}

    return {
        "sources": [
            {
                "document_id": "bee_thermal",
                "book_name": "Energy Efficiency in Thermal Utilities",
                "domain": "thermal",
            },
            {
                "document_id": "bee_electrical",
                "book_name": "Energy Efficiency in Electrical Utilities",
                "domain": "electrical",
            },
        ],
        "total_chunks": svc.chunk_count,
    }
