"""
api/health.py
=============
Health check and status endpoints.
"""

from fastapi import APIRouter, Query
from app.models.schemas import HealthResponse
from app.services import get_service
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(reload: bool = Query(False, description="Force index reload check")):
    """
    Health check endpoint.
    
    Returns the status of the backend, whether indexes are loaded,
    and key configuration details.
    
    Add ?reload=true to force a live re-check of the index state.
    """
    svc = get_service()
    if reload:
        svc.reload()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        index_loaded=svc.is_loaded,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_provider=settings.LLM_PROVIDER,
        chunk_count=svc.chunk_count,
    )


@router.get("/status")
async def detailed_status():
    """Detailed status for debugging and monitoring."""
    svc = get_service()
    return {
        "status": "ok",
        "index_loaded": svc.is_loaded,
        "chunk_count": svc.chunk_count,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranking_enabled": settings.ENABLE_RERANKING,
        "hybrid_alpha": settings.HYBRID_ALPHA,
        "max_context_chunks": settings.MAX_CONTEXT_CHUNKS,
        "pdf_thermal": str(settings.PDF_THERMAL_PATH),
        "pdf_electrical": str(settings.PDF_ELECTRICAL_PATH),
        "thermal_pdf_exists": settings.PDF_THERMAL_PATH.exists(),
        "electrical_pdf_exists": settings.PDF_ELECTRICAL_PATH.exists(),
        "chroma_dir": str(settings.CHROMA_DIR_ABS),
        "bm25_index": str(settings.BM25_INDEX_PATH_ABS),
        "bm25_exists": settings.BM25_INDEX_PATH_ABS.exists(),
    }


@router.post("/reload")
async def reload_indexes():
    """Force a full reload of all indexes. Call this after running ingestion."""
    svc = get_service()
    loaded = svc.reload()
    return {
        "status": "reloaded",
        "index_loaded": loaded,
        "chunk_count": svc.chunk_count,
    }
