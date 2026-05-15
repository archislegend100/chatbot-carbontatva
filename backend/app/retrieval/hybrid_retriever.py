"""
retrieval/hybrid_retriever.py
==============================
Hybrid retrieval using Reciprocal Rank Fusion (RRF) to merge
dense (ChromaDB) and sparse (BM25) results.

Why RRF?
- RRF is simple, parameter-free, and consistently outperforms
  score-based merging across many retrieval benchmarks.
- It penalizes low-ranked results regardless of raw score,
  which helps when dense and sparse scores are on different scales.

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is a standard hyperparameter.

Reference:
  Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and
  individual Rank Learning Methods." SIGIR 2009.
"""

from __future__ import annotations

from typing import Any, Optional

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RRF_K = 60  # Standard RRF constant


class HybridRetriever:
    """
    Combines dense and sparse retrieval results using RRF fusion.

    Args:
        dense_retriever: DenseRetriever instance.
        sparse_retriever: SparseRetriever instance.
        dense_top_k: How many candidates from dense search.
        sparse_top_k: How many candidates from sparse search.
        alpha: Weight for dense vs sparse (0.0=sparse only, 1.0=dense only).
               Applied as a score blending step before RRF for tie-breaking.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_top_k: int = settings.DENSE_TOP_K,
        sparse_top_k: int = settings.SPARSE_TOP_K,
        alpha: float = settings.HYBRID_ALPHA,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        total_top_k: int = 30,
        domain_filter: Optional[str] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and merge results from dense + sparse search using RRF.

        Args:
            query: User query string.
            total_top_k: Total number of merged results to return.
            domain_filter: Optional domain filter ("thermal"/"electrical").
            chunk_types: Optional chunk type filter.

        Returns:
            Merged, RRF-ranked list of result dicts, sorted by score desc.
        """
        # Step 1: Retrieve from both retrievers
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=self.dense_top_k,
            domain_filter=domain_filter,
            chunk_types=chunk_types,
        )
        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            top_k=self.sparse_top_k,
            domain_filter=domain_filter,
        )

        logger.debug(
            f"Hybrid: dense={len(dense_results)}, sparse={len(sparse_results)}"
        )

        # Step 2: RRF fusion
        merged = self._rrf_merge(dense_results, sparse_results, total_top_k)
        return merged

    def _rrf_merge(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Returns the top_k results by combined RRF score.
        """
        # Map chunk_id → result dict (dense wins over sparse for same chunk)
        all_results: dict[str, dict[str, Any]] = {}

        for result in dense_results:
            cid = result["chunk_id"]
            all_results[cid] = result

        for result in sparse_results:
            cid = result["chunk_id"]
            if cid not in all_results:
                all_results[cid] = result
            else:
                # Merge retrieval_source info
                all_results[cid]["retrieval_source"] = "hybrid"

        # Compute RRF scores
        rrf_scores: dict[str, float] = {cid: 0.0 for cid in all_results}

        # Dense rankings
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)

        # Sparse rankings
        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        merged = []
        for cid in sorted_ids[:top_k]:
            result = all_results[cid].copy()
            result["rrf_score"] = rrf_scores[cid]
            merged.append(result)

        return merged
