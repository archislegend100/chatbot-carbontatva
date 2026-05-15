"""
graph/nodes.py
===============
All LangGraph node functions for the copilot pipeline.

Each node:
  - Takes CopilotState as input
  - Returns a partial dict (LangGraph merges it into state)
  - Is async where IO-bound
  - Records its own latency in state["latency"]
  - Appends its name to state["debug_trace"]

Node execution order (see graph.py):
  normalize_query → planner_router → [retrieve_dense ‖ retrieve_sparse]
  → merge_results → rerank → tool_dispatch → answer_generation
  → citation_assembly → response_formatter

All retrieval + generation components are injected via a dependency
container rather than imported globally (makes testing easier).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from app.graph.state import CopilotState
from app.graph.planner import run_planner
from app.generation.prompts import build_prompt
from app.models.schemas import ExplanationLevel, ToolMode
from app.core.logging import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------
# NODE: normalize_query
# -------------------------------------------------------

def normalize_query(state: CopilotState) -> dict:
    """
    Lightweight query normalization — no LLM call.
    
    - Strip whitespace
    - Lowercase for routing (preserve original for display)
    - Detect obvious empty queries
    """
    t0 = time.monotonic()
    
    raw = state.get("user_query", "").strip()
    normalized = raw.lower().strip()
    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["normalize_query"]
    latency = dict(state.get("latency", {}))
    latency["normalize_query"] = round(ms, 2)
    
    return {
        "normalized_query": normalized,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: planner_router
# -------------------------------------------------------

async def planner_router(state: CopilotState, llm_client) -> dict:
    """
    Structured reasoning planner — the routing brain.
    
    Calls LLM with a compact JSON-mode prompt to determine:
    - selected_tool, domain, equipment_tags, retrieval_profile, etc.
    
    Falls back to rule-based if LLM fails.
    """
    t0 = time.monotonic()
    
    query = state.get("normalized_query") or state.get("user_query", "")
    domain_hint = state.get("domain_filter")
    
    plan, raw, had_error = await run_planner(
        query=query,
        llm_client=llm_client,
        domain_hint=domain_hint,
    )
    
    # Apply user domain override
    if domain_hint and domain_hint in ("thermal", "electrical"):
        plan["utility_domain"] = domain_hint
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["planner_router"]
    latency = dict(state.get("latency", {}))
    latency["planner_router"] = round(ms, 2)
    
    return {
        "planner": plan,
        "planner_raw": raw,
        "planner_error": had_error,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: retrieve_dense
# -------------------------------------------------------

async def retrieve_dense(state: CopilotState, hybrid_retriever) -> dict:
    """Dense vector retrieval from ChromaDB."""
    t0 = time.monotonic()
    
    query = state.get("user_query", "")
    plan = state.get("planner", {})
    domain = plan.get("utility_domain")
    domain_filter = None if domain == "both" else domain
    
    # Tool-specific chunk type preferences
    tool = plan.get("selected_tool", "qa")
    chunk_type_map = {
        "checklist": ["list", "semantic"],
        "comparison": ["semantic", "table"],
        "navigation": ["section"],
        "summarize": ["section", "semantic"],
        "qa": ["semantic", "table"],
        "troubleshoot": ["semantic", "list"],
    }
    chunk_types = chunk_type_map.get(tool)
    
    candidates = hybrid_retriever.dense_retriever.retrieve(
        query=query,
        top_k=20,
        domain_filter=domain_filter,
        chunk_types=chunk_types,
    )
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["retrieve_dense"]
    latency = dict(state.get("latency", {}))
    latency["retrieve_dense"] = round(ms, 2)
    
    return {
        "dense_candidates": candidates,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: retrieve_sparse
# -------------------------------------------------------

async def retrieve_sparse(state: CopilotState, hybrid_retriever) -> dict:
    """Sparse BM25 retrieval."""
    t0 = time.monotonic()
    
    query = state.get("user_query", "")
    plan = state.get("planner", {})
    domain = plan.get("utility_domain")
    domain_filter = None if domain == "both" else domain
    
    candidates = hybrid_retriever.sparse_retriever.retrieve(
        query=query,
        top_k=20,
        domain_filter=domain_filter,
    )
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["retrieve_sparse"]
    latency = dict(state.get("latency", {}))
    latency["retrieve_sparse"] = round(ms, 2)
    
    return {
        "sparse_candidates": candidates,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: merge_results  (RRF fusion)
# -------------------------------------------------------

def merge_results(state: CopilotState, hybrid_retriever) -> dict:
    """Reciprocal Rank Fusion of dense and sparse candidates."""
    t0 = time.monotonic()
    
    dense = state.get("dense_candidates", [])
    sparse = state.get("sparse_candidates", [])
    
    merged = hybrid_retriever._rrf_fuse(dense, sparse)
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["merge_results"]
    latency = dict(state.get("latency", {}))
    latency["merge_results"] = round(ms, 2)
    
    return {
        "merged_candidates": merged,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: rerank
# -------------------------------------------------------

def rerank(state: CopilotState, reranker) -> dict:
    """Cross-encoder reranking of merged candidates."""
    t0 = time.monotonic()
    
    query = state.get("user_query", "")
    candidates = state.get("merged_candidates", [])
    
    from app.core.config import settings
    top_k = settings.MAX_CONTEXT_CHUNKS
    
    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["rerank"]
    latency = dict(state.get("latency", {}))
    latency["rerank"] = round(ms, 2)
    
    return {
        "reranked_chunks": reranked,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: tool_dispatch (builds tool-specific system prompt)
# -------------------------------------------------------

def tool_dispatch(state: CopilotState) -> dict:
    """
    Selects and builds the tool-specific prompt.
    No LLM call — pure deterministic routing.
    """
    t0 = time.monotonic()
    
    plan = state.get("planner", {})
    tool_str = plan.get("selected_tool", "qa")
    
    # Map string to ToolMode enum
    try:
        tool_mode = ToolMode(tool_str)
    except ValueError:
        tool_mode = ToolMode.QA
    
    level_str = state.get("explanation_level", "engineer")
    try:
        level = ExplanationLevel(level_str)
    except ValueError:
        level = ExplanationLevel.ENGINEER
    
    chunks = state.get("reranked_chunks", [])
    query = state.get("user_query", "")
    
    system_prompt, user_prompt = build_prompt(
        tool_mode=tool_mode,
        query=query,
        chunks=chunks,
        explanation_level=level,
    )
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["tool_dispatch"]
    latency = dict(state.get("latency", {}))
    latency["tool_dispatch"] = round(ms, 2)
    
    return {
        "system_prompt": system_prompt,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: answer_generation
# -------------------------------------------------------

async def answer_generation(state: CopilotState, llm_client) -> dict:
    """Generate the final answer using the LLM."""
    t0 = time.monotonic()
    
    system_prompt = state.get("system_prompt", "You are a helpful industrial energy efficiency assistant.")
    user_query = state.get("user_query", "")
    chunks = state.get("reranked_chunks", [])
    
    # Build a compact user message (system_prompt already has context from build_prompt)
    from app.generation.prompts import build_prompt
    from app.models.schemas import ToolMode, ExplanationLevel
    plan = state.get("planner", {})
    tool_mode = ToolMode(plan.get("selected_tool", "qa"))
    level = ExplanationLevel(state.get("explanation_level", "engineer"))
    
    _, user_prompt = build_prompt(tool_mode, user_query, chunks, level)
    
    try:
        answer = await llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1200,
            temperature=0.15,
        )
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        answer = (
            "I encountered an error generating a response. "
            "Please ensure Ollama is running: `ollama serve`"
        )
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["answer_generation"]
    latency = dict(state.get("latency", {}))
    latency["answer_generation"] = round(ms, 2)
    
    return {
        "answer_draft": answer,
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: citation_assembly
# -------------------------------------------------------

def citation_assembly(state: CopilotState, vector_store) -> dict:
    """Convert reranked chunks to citation objects."""
    t0 = time.monotonic()
    
    chunks = state.get("reranked_chunks", [])
    citations = [vector_store.result_to_citation(c) for c in chunks]
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["citation_assembly"]
    latency = dict(state.get("latency", {}))
    latency["citation_assembly"] = round(ms, 2)
    
    return {
        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
        "debug_trace": trace,
        "latency": latency,
    }


# -------------------------------------------------------
# NODE: response_formatter
# -------------------------------------------------------

def response_formatter(state: CopilotState) -> dict:
    """Assemble the final structured response dict."""
    t0 = time.monotonic()
    
    plan = state.get("planner", {})
    total_latency = sum(state.get("latency", {}).values())
    
    # Follow-up suggestions by tool
    follow_ups_map = {
        "qa": ["Explain this in more detail", "What are energy saving opportunities?", "Generate an inspection checklist"],
        "explainer": ["Show me a comparison", "What are common problems?", "Summarize this chapter"],
        "troubleshoot": ["Generate an inspection checklist", "Find energy saving opportunities", "Explain the underlying concept"],
        "opportunity": ["Create an implementation checklist", "Compare the top options", "Summarize this section"],
        "comparison": ["Explain the better option in detail", "Show saving opportunities for each", "What does the manual recommend?"],
        "navigation": ["Summarize that chapter", "Get key energy saving tips from that section"],
        "checklist": ["Explain any unfamiliar terms", "Show saving opportunities for this system"],
        "summarize": ["Explain a concept from this summary", "Generate an audit checklist", "What are the opportunities?"],
    }
    tool = plan.get("selected_tool", "qa")
    follow_ups = follow_ups_map.get(tool, ["Ask a follow-up question"])
    
    final_response = {
        "query": state.get("user_query", ""),
        "tool_mode": tool,
        "answer": state.get("answer_draft", ""),
        "citations": state.get("citations", []),
        "classification": {
            "tool_mode": tool,
            "utility_domain": plan.get("utility_domain"),
            "equipment_tags": plan.get("equipment_tags", []),
            "concept_tags": plan.get("equipment_tags", []),
            "confidence": plan.get("confidence", 0.5),
            "reasoning": plan.get("why_selected", ""),
            "planner_raw": state.get("planner_raw", ""),
            "planner_error": state.get("planner_error", False),
        },
        "follow_up_suggestions": follow_ups,
        "retrieval_count": len(state.get("reranked_chunks", [])),
        "generation_model": "ollama",
        "latency_ms": round(total_latency, 1),
        "debug_trace": state.get("debug_trace", []),
        "node_latency": state.get("latency", {}),
    }
    
    ms = (time.monotonic() - t0) * 1000
    trace = list(state.get("debug_trace", [])) + ["response_formatter"]
    latency = dict(state.get("latency", {}))
    latency["response_formatter"] = round(ms, 2)
    
    return {
        "final_response": final_response,
        "debug_trace": trace,
        "latency": latency,
    }
