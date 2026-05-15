"""
retrieval/sparse_retriever.py
==============================
Sparse BM25 retrieval wrapper.
"""

from __future__ import annotations

from typing import Any, Optional

from app.indexing.bm25_index import BM25Index
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """
    Retrieves chunks using BM25 keyword matching.

    Args:
        bm25_index: BM25Index instance.
        default_top_k: Default number of results.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        default_top_k: int = settings.SPARSE_TOP_K,
    ):
        self.bm25_index = bm25_index
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        domain_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks by BM25 relevance.

        Args:
            query: Raw query text.
            top_k: Number of results.
            domain_filter: "thermal" or "electrical" (or None).

        Returns:
            List of result dicts sorted by BM25 score descending.
        """
        k = top_k or self.default_top_k

        results = self.bm25_index.query(
            query_text=query,
            top_k=k,
            domain_filter=domain_filter,
        )

        for r in results:
            r["retrieval_source"] = "sparse"

        return results
