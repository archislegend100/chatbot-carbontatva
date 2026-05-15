"""
graph/graph.py
===============
LangGraph compiled graph for the copilot pipeline.

Graph topology (stateful, directed):

  normalize_query
       ↓
  planner_router
       ↓
  [retrieve_dense ‖ retrieve_sparse]  ← parallel branches (fan-out)
       ↓
  merge_results                        ← fan-in
       ↓
  rerank
       ↓
  tool_dispatch
       ↓
  answer_generation
       ↓
  citation_assembly
       ↓
  response_formatter
       ↓
      END

Design notes:
- Dense + sparse retrieval run in parallel via LangGraph fan-out/fan-in
- All components are injected, not imported globally
- Graph is compiled once at startup and reused per request
- State checkpointing is disabled (stateless per-request is faster here)

Usage:
    graph = build_graph(components)
    result = await graph.ainvoke(initial_state)
    response = result["final_response"]
"""

from __future__ import annotations

import functools
from typing import Any

from langgraph.graph import StateGraph, END

from app.graph.state import CopilotState
from app.graph.nodes import (
    normalize_query,
    planner_router,
    retrieve_dense,
    retrieve_sparse,
    merge_results,
    rerank,
    tool_dispatch,
    answer_generation,
    citation_assembly,
    response_formatter,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_graph(
    llm_client,
    hybrid_retriever,
    reranker,
    vector_store,
):
    """
    Build and compile the LangGraph copilot pipeline.
    
    Args:
        llm_client: LLM client (Ollama or fallback)
        hybrid_retriever: HybridRetriever with dense + sparse components
        reranker: Cross-encoder reranker
        vector_store: VectorStore for citation lookup
    
    Returns:
        Compiled LangGraph ready for .ainvoke()
    """
    
    # Bind component dependencies to node functions via functools.partial
    # This is cleaner than using global state or a service locator

    bound_planner = functools.partial(planner_router, llm_client=llm_client)
    bound_dense = functools.partial(retrieve_dense, hybrid_retriever=hybrid_retriever)
    bound_sparse = functools.partial(retrieve_sparse, hybrid_retriever=hybrid_retriever)
    bound_merge = functools.partial(merge_results, hybrid_retriever=hybrid_retriever)
    bound_rerank = functools.partial(rerank, reranker=reranker)
    bound_answer = functools.partial(answer_generation, llm_client=llm_client)
    bound_citation = functools.partial(citation_assembly, vector_store=vector_store)
    
    # Build the graph
    builder = StateGraph(CopilotState)
    
    # Register all nodes
    builder.add_node("normalize_query", normalize_query)
    builder.add_node("planner_router", bound_planner)
    builder.add_node("retrieve_dense", bound_dense)
    builder.add_node("retrieve_sparse", bound_sparse)
    builder.add_node("merge_results", bound_merge)
    builder.add_node("rerank", bound_rerank)
    builder.add_node("tool_dispatch", tool_dispatch)
    builder.add_node("answer_generation", bound_answer)
    builder.add_node("citation_assembly", bound_citation)
    builder.add_node("response_formatter", response_formatter)
    
    # ---- EDGES ----
    
    # Entry point
    builder.set_entry_point("normalize_query")
    
    # Linear: normalize → planner
    builder.add_edge("normalize_query", "planner_router")
    
    # Fan-out: planner → [dense ‖ sparse] in parallel
    builder.add_edge("planner_router", "retrieve_dense")
    builder.add_edge("planner_router", "retrieve_sparse")
    
    # Fan-in: both retrieval branches → merge
    builder.add_edge("retrieve_dense", "merge_results")
    builder.add_edge("retrieve_sparse", "merge_results")
    
    # Linear pipeline after merge
    builder.add_edge("merge_results", "rerank")
    builder.add_edge("rerank", "tool_dispatch")
    builder.add_edge("tool_dispatch", "answer_generation")
    builder.add_edge("answer_generation", "citation_assembly")
    builder.add_edge("citation_assembly", "response_formatter")
    builder.add_edge("response_formatter", END)
    
    # Compile and return
    graph = builder.compile()
    logger.info("LangGraph pipeline compiled: 10 nodes, parallel retrieval")
    return graph


async def run_graph(
    graph,
    query: str,
    session_id: str = "default",
    domain_filter: str = None,
    explanation_level: str = "engineer",
) -> dict[str, Any]:
    """
    Run the compiled graph for a single query.
    
    Args:
        graph: Compiled LangGraph
        query: User query string
        session_id: For logging/caching
        domain_filter: "thermal" | "electrical" | None
        explanation_level: "beginner" | "engineer"
    
    Returns:
        final_response dict
    """
    initial_state: CopilotState = {
        "user_query": query,
        "session_id": session_id,
        "domain_filter": domain_filter,
        "explanation_level": explanation_level,
        "debug_trace": [],
        "latency": {},
    }
    
    result = await graph.ainvoke(initial_state)
    return result.get("final_response", {})
