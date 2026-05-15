"""
api/ingest.py
=============
Ingestion trigger endpoint — allows re-running ingestion via API.
"""

from __future__ import annotations

import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.models.schemas import IngestStatusResponse
from app.core.logging import get_logger

router = APIRouter(prefix="/api", tags=["Ingestion"])
logger = get_logger(__name__)

_ingestion_running = False


@router.post("/ingest", response_model=IngestStatusResponse)
async def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Trigger a full re-ingestion and re-indexing of the PDFs.
    
    This runs in the background. Poll /api/health to see when
    index_loaded becomes true.
    
    WARNING: This will reset and rebuild the entire index.
    Existing queries will fail during re-ingestion.
    """
    global _ingestion_running

    if _ingestion_running:
        return IngestStatusResponse(
            status="running",
            message="Ingestion is already running. Please wait.",
        )

    async def run_ingestion():
        global _ingestion_running
        _ingestion_running = True
        try:
            logger.info("Starting background ingestion via API trigger...")
            # Import here to avoid circular imports at module load
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "scripts/ingest.py", "--force-rebuild"],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour max
            )
            if result.returncode == 0:
                logger.info("Ingestion completed successfully via API")
                # Reload service
                from app.services import get_service
                get_service().load()
            else:
                logger.error(f"Ingestion failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Background ingestion error: {e}")
        finally:
            _ingestion_running = False

    background_tasks.add_task(run_ingestion)

    return IngestStatusResponse(
        status="started",
        message=(
            "Ingestion started in background. "
            "This may take 15-40 minutes for the full PDFs. "
            "Poll GET /api/health to monitor progress."
        ),
    )


@router.get("/ingest/status")
async def ingest_status():
    """Check if ingestion is currently running."""
    return {"ingestion_running": _ingestion_running}
