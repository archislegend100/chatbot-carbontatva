"""
graph/state.py
===============
Typed state object for the LangGraph copilot pipeline.

The CopilotState flows through every node in the graph.
Each node reads from it and writes back to it.

Using TypedDict for LangGraph compatibility (not Pydantic —
LangGraph requires TypedDict for state checkpointing).
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class PlannerOutput(TypedDict, total=False):
    """Structured JSON output from the planner node."""
    query_type: str            # qa | explainer | troubleshoot | opportunity | comparison | navigation | checklist | summarize
    utility_domain: str        # thermal | electrical | both
    equipment_tags: list[str]  # ["boiler", "steam_trap"]
    candidate_tools: list[str] # ["qa", "troubleshoot"]
    selected_tool: str         # "troubleshoot"
    confidence: float          # 0.0–1.0
    why_selected: str          # "Query describes a performance problem..."
    retrieval_profile: str     # "semantic_dense" | "table_chunks" | "metadata_first"
    response_style: str        # "structured_diagnosis" | "bullet_list" | "comparison_table"


class CopilotState(TypedDict, total=False):
    """
    Full state object flowing through the LangGraph pipeline.
    
    Nodes read from this state and return partial updates.
    LangGraph merges the updates into the state automatically.
    """

    # ---- INPUT ----
    user_query: str                  # Raw query from user
    session_id: str                  # For caching / logging
    explanation_level: str           # "beginner" | "engineer"
    domain_filter: Optional[str]     # Optional user domain override

    # ---- NORMALIZATION ----
    normalized_query: str            # Cleaned, lowercased query

    # ---- PLANNER OUTPUT ----
    planner: PlannerOutput           # Structured reasoning object
    planner_raw: str                 # Raw JSON string (for debug)
    planner_error: bool              # True if planner failed (fallback used)

    # ---- RETRIEVAL ----
    dense_candidates: list[dict]     # From ChromaDB
    sparse_candidates: list[dict]    # From BM25
    merged_candidates: list[dict]    # After RRF fusion
    reranked_chunks: list[dict]      # After cross-encoder reranking

    # ---- GENERATION ----
    system_prompt: str               # Built system prompt
    context_block: str               # Formatted evidence for LLM
    answer_draft: str                # LLM answer text
    citations: list[dict]            # Formatted citation objects

    # ---- FINAL OUTPUT ----
    final_response: dict[str, Any]   # Full structured response

    # ---- DEBUG ----
    debug_trace: list[str]           # Ordered list of node names executed
    latency: dict[str, float]        # Per-node latency in ms
    error: Optional[str]             # Any error message
