from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.schemas import UtilityDomain, ToolMode, ExplanationLevel
from app.services import get_service
from app.graph.planner import run_planner, _rule_based_plan
from app.generation.prompts import build_prompt
from app.core.logging import get_logger

router = APIRouter(prefix="/api", tags=["Streaming"])
logger = get_logger(__name__)


class StreamRequest(BaseModel):
    query: str
    tool_mode: ToolMode = ToolMode.AUTO
    domain_filter: UtilityDomain | None = None
    explanation_level: ExplanationLevel = ExplanationLevel.ENGINEER


async def _stream_pipeline(request: StreamRequest) -> AsyncGenerator[str, None]:
    svc = get_service()
    t0 = time.monotonic()

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield _sse({"type": "status", "content": "Analyzing query..."})

    if not svc.is_loaded:
        yield _sse({"type": "error", "content": "Knowledge base not indexed. Run: ./run_ingest.sh"})
        yield "data: [DONE]\n\n"
        return

    domain_hint = request.domain_filter.value if request.domain_filter else None

    try:
        plan, plan_raw, plan_error = await asyncio.wait_for(
            run_planner(request.query, svc.llm_client, domain_hint),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        plan = _rule_based_plan(request.query, domain_hint)
        plan_raw = json.dumps(plan)
        plan_error = True

    if request.tool_mode != ToolMode.AUTO:
        plan["selected_tool"] = request.tool_mode.value
    if domain_hint:
        plan["utility_domain"] = domain_hint

    yield _sse({"type": "planner", "planner": plan, "error": plan_error})
    yield _sse({"type": "status", "content": "Retrieving relevant passages..."})

    domain_filter = plan["utility_domain"] if plan["utility_domain"] != "both" else None
    try:
        candidates = svc.hybrid_retriever.retrieve(
            query=request.query,
            total_top_k=16,
            domain_filter=domain_filter,
        )
        chunks = (
            svc.reranker.rerank(query=request.query, candidates=candidates, top_k=4)
            if svc.reranker
            else candidates[:4]
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        chunks = []

    yield _sse({"type": "status", "content": f"Generating answer..."})

    tool_mode = ToolMode(plan.get("selected_tool", "qa"))
    system_prompt, user_prompt = build_prompt(
        tool_mode=tool_mode,
        query=request.query,
        chunks=chunks,
        explanation_level=request.explanation_level,
    )

    token_count = 0
    try:
        async for token in svc.llm_client.stream(system_prompt, user_prompt):
            yield _sse({"type": "token", "content": token})
            token_count += 1
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield _sse({"type": "error", "content": f"Generation failed: {e}"})

    citations = []
    if svc.vector_store:
        try:
            citations = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in (svc.vector_store.result_to_citation(c) for c in chunks)
            ]
        except Exception:
            pass

    total_ms = round((time.monotonic() - t0) * 1000, 1)
    yield _sse({
        "type": "done",
        "citations": citations,
        "planner": plan,
        "tool_mode": plan.get("selected_tool", "qa"),
        "retrieval_count": len(chunks),
        "latency_ms": total_ms,
        "token_count": token_count,
        "follow_up_suggestions": _follow_ups(plan.get("selected_tool", "qa")),
    })
    yield "data: [DONE]\n\n"


def _follow_ups(tool: str) -> list[str]:
    m = {
        "qa": ["Explain this in more detail", "What are the energy saving opportunities?"],
        "troubleshoot": ["Generate an inspection checklist", "Find energy saving opportunities"],
        "opportunity": ["Create a detailed checklist", "Summarize the relevant manual section"],
        "comparison": ["Explain the better option in detail", "Show saving opportunities for each"],
        "checklist": ["Explain any unfamiliar terms", "Show saving opportunities for this system"],
        "explainer": ["Generate a checklist", "Compare with alternatives"],
        "navigation": ["Summarize that chapter"],
        "summarize": ["Generate an audit checklist", "Explain a concept from this summary"],
    }
    return m.get(tool, ["Ask a follow-up question"])


@router.post("/stream")
async def stream_query(request: StreamRequest):
    return StreamingResponse(
        _stream_pipeline(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
