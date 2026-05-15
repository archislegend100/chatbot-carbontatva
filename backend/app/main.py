from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.ingest import router as ingest_router
from app.api.stream import router as stream_router
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.services import get_service

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting up | provider={settings.LLM_PROVIDER} | model={settings.MISTRAL_MODEL if settings.LLM_PROVIDER == 'mistral' else settings.OLLAMA_MODEL}")
    svc = get_service()
    loaded = svc.load()
    if loaded:
        logger.info(f"Knowledge base ready: {svc.chunk_count} chunks")
    else:
        logger.warning("Knowledge base not indexed. Run: ./run_ingest.sh")
    yield
    logger.info("Backend shutting down")


app = FastAPI(
    title="Industrial Energy Efficiency Copilot",
    description="Multi-tool RAG chatbot grounded in BEE Thermal and Electrical Utility Manuals.",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(stream_router)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error": str(exc)},
    )


@app.get("/")
async def root():
    return {
        "name": "Industrial Energy Efficiency Copilot",
        "version": "2.1.0",
        "docs": "/api/docs",
        "health": "/api/health",
    }
