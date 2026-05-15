"""
router/tool_router.py
=====================
Tool router — dispatches classified queries to the correct handler
and orchestrates the full RAG pipeline:

  Query → Classify → Retrieve → Rerank → Generate → Response

This is the central orchestration module used by the API layer.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.models.schemas import (
    AnswerResponse,
    ClassificationResult,
    ExplanationLevel,
    QueryRequest,
    SourceCitation,
    ToolMode,
    UtilityDomain,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.router.query_classifier import QueryClassifier
from app.generation.llm_client import BaseLLMClient
from app.generation.prompts import build_prompt
from app.indexing.vector_store import VectorStore
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ToolRouter:
    """
    Central orchestrator for the RAG pipeline.

    Given a QueryRequest, it:
    1. Classifies the query (tool mode, domain, tags)
    2. Retrieves relevant chunks (hybrid search)
    3. Reranks candidates
    4. Generates a grounded answer using the appropriate tool prompt
    5. Returns a structured AnswerResponse

    Args:
        hybrid_retriever: Configured hybrid retriever.
        reranker: Cross-encoder reranker.
        classifier: Query classifier.
        llm_client: LLM client for generation.
        vector_store: Vector store (for citation lookup).
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        classifier: QueryClassifier,
        llm_client: BaseLLMClient,
        vector_store: VectorStore,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.classifier = classifier
        self.llm_client = llm_client
        self.vector_store = vector_store

    async def handle(self, request: QueryRequest) -> AnswerResponse:
        """
        Handle a full query request end-to-end.

        Args:
            request: The incoming QueryRequest.

        Returns:
            Structured AnswerResponse.
        """
        t0 = time.monotonic()

        # ---- Step 1: Classify ----
        classification = self.classifier.classify(request.query)

        # Honor explicit tool_mode override
        if request.tool_mode != ToolMode.AUTO:
            classification.tool_mode = request.tool_mode

        # Honor explicit domain filter override
        effective_domain: Optional[str] = None
        if request.domain_filter and request.domain_filter != UtilityDomain.UNKNOWN:
            effective_domain = request.domain_filter.value
        elif classification.utility_domain:
            effective_domain = classification.utility_domain.value

        logger.info(
            f"Query classified → tool={classification.tool_mode.value} "
            f"domain={effective_domain or 'all'} "
            f"confidence={classification.confidence:.2f}"
        )

        # ---- Step 2: Retrieve ----
        top_k = request.top_k or settings.RERANK_TOP_K * 3

        # Choose chunk types based on tool mode
        chunk_types = self._get_chunk_types_for_tool(classification.tool_mode)

        candidates = self.hybrid_retriever.retrieve(
            query=request.query,
            total_top_k=top_k,
            domain_filter=effective_domain,
            chunk_types=chunk_types,
        )

        logger.info(f"Hybrid retrieval: {len(candidates)} candidates")

        # ---- Step 3: Rerank ----
        top_chunks = self.reranker.rerank(
            query=request.query,
            candidates=candidates,
            top_k=settings.MAX_CONTEXT_CHUNKS,
        )

        logger.info(f"After reranking: {len(top_chunks)} context chunks")

        # ---- Step 4: Generate Answer ----
        system_prompt, user_prompt = build_prompt(
            tool_mode=classification.tool_mode,
            query=request.query,
            chunks=top_chunks,
            explanation_level=request.explanation_level,
        )

        try:
            answer_text = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer_text = (
                "I encountered an error generating a response. "
                "Please check the server logs and your API key configuration."
            )

        # ---- Step 5: Build Citations ----
        citations = [
            self.vector_store.result_to_citation(chunk)
            for chunk in top_chunks
        ]

        # ---- Step 6: Build Response ----
        latency_ms = (time.monotonic() - t0) * 1000
        follow_ups = self._generate_follow_ups(classification.tool_mode, request.query)

        return AnswerResponse(
            query=request.query,
            tool_mode=classification.tool_mode,
            answer=answer_text,
            structured_sections={},
            citations=citations,
            classification=classification,
            follow_up_suggestions=follow_ups,
            retrieval_count=len(top_chunks),
            generation_model=settings.LLM_PROVIDER,
            latency_ms=round(latency_ms, 1),
        )

    def _get_chunk_types_for_tool(self, tool_mode: ToolMode) -> Optional[list[str]]:
        """
        Return preferred chunk types for each tool mode.
        None means all types are acceptable.
        """
        preferences = {
            ToolMode.QA: ["semantic", "table", "list"],
            ToolMode.EXPLAINER: ["semantic", "section"],
            ToolMode.TROUBLESHOOT: ["semantic", "list"],
            ToolMode.OPPORTUNITY: ["semantic", "list", "section"],
            ToolMode.COMPARISON: ["semantic", "table"],
            ToolMode.NAVIGATION: ["section"],
            ToolMode.CHECKLIST: ["list", "semantic"],
            ToolMode.SUMMARIZE: ["section", "semantic"],
            ToolMode.AUTO: None,
        }
        return preferences.get(tool_mode)

    def _generate_follow_ups(
        self, tool_mode: ToolMode, query: str
    ) -> list[str]:
        """Generate contextual follow-up suggestions."""
        base_suggestions = {
            ToolMode.QA: [
                "Explain this concept in detail",
                "What are energy saving opportunities here?",
                "Generate an inspection checklist for this",
            ],
            ToolMode.EXPLAINER: [
                "Show me a comparison with alternatives",
                "What are common problems with this?",
                "Summarize the full chapter on this topic",
            ],
            ToolMode.TROUBLESHOOT: [
                "Generate an inspection checklist",
                "What are the energy saving opportunities?",
                "Explain the underlying concept",
            ],
            ToolMode.OPPORTUNITY: [
                "Create a detailed implementation checklist",
                "Compare the top two opportunities",
                "Summarize this section of the manual",
            ],
            ToolMode.COMPARISON: [
                "Show me energy saving opportunities for each",
                "Explain the better option in detail",
                "What does the manual recommend?",
            ],
            ToolMode.NAVIGATION: [
                "Summarize the chapter you found",
                "What are the key energy saving tips in that section?",
                "Explain the main concept from that chapter",
            ],
            ToolMode.CHECKLIST: [
                "Explain any unfamiliar terms",
                "Show energy saving opportunities for this system",
                "Summarize the relevant manual sections",
            ],
            ToolMode.SUMMARIZE: [
                "Explain any concept from this summary in detail",
                "Generate an audit checklist for this topic",
                "What are the energy saving opportunities?",
            ],
        }
        return base_suggestions.get(tool_mode, [
            "Ask a follow-up question",
            "Explore related topics",
        ])
