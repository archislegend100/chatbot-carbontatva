"""
retrieval/reranker.py
======================
Cross-encoder reranking for the Industrial Energy Efficiency Copilot.

After hybrid retrieval produces ~20-30 candidate chunks, the reranker
scores each (query, chunk) pair using a cross-encoder model.

Cross-encoders are much more expensive than bi-encoders (per-chunk
scoring vs. single query embedding) but are far more accurate.
We apply them to a small candidate set (20-30) to keep latency acceptable.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage reranking
- Works well for technical Q&A retrieval
- ~22MB model, runs on CPU in ~100ms for 20 candidates

Usage:
    reranker = Reranker()
    top_k = reranker.rerank(query, candidates, top_k=8)
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for re-scoring retrieved candidates.

    Lazily loads the cross-encoder model on first use.
    Falls back to score-based sorting if model loading fails.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading reranker model: {RERANKER_MODEL}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(RERANKER_MODEL)
                logger.info("Reranker model loaded.")
            except Exception as e:
                logger.warning(f"Reranker model failed to load: {e}. Using score fallback.")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = settings.RERANK_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate chunks by cross-encoder score.

        Args:
            query: The user query string.
            candidates: List of result dicts from hybrid retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Top-k results sorted by cross-encoder score descending,
            with a "rerank_score" field added.
        """
        if not candidates:
            return []

        if not settings.ENABLE_RERANKING:
            return candidates[:top_k]

        self._load_model()

        if self._model is None:
            # Fallback: sort by rrf_score or score
            sorted_by_score = sorted(
                candidates,
                key=lambda x: x.get("rrf_score", x.get("score", 0.0)),
                reverse=True,
            )
            return sorted_by_score[:top_k]

        # Prepare (query, passage) pairs
        pairs = [(query, c["text"]) for c in candidates]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Reranker prediction failed: {e}. Using fallback.")
            return candidates[:top_k]

        # Attach rerank scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
