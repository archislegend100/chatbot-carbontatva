"""
retrieval/dense_retriever.py
=============================
Dense vector retrieval from ChromaDB.

Wraps VectorStore.query() with a clean retrieval interface
that accepts typed queries and returns sorted results.
"""

from __future__ import annotations

from typing import Any, Optional

from app.indexing.embedder import Embedder
from app.indexing.vector_store import VectorStore
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Retrieves chunks using dense vector similarity search.

    Args:
        embedder: Embedder instance.
        vector_store: VectorStore instance.
        default_top_k: Default number of results.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = settings.DENSE_TOP_K,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks by dense similarity.

        Args:
            query: Raw query text.
            top_k: Number of results.
            domain_filter: "thermal" or "electrical" (or None for both).
            chunk_types: Filter to specific chunk types.

        Returns:
            List of result dicts sorted by score descending.
        """
        k = top_k or self.default_top_k
        query_embedding = self.embedder.embed_query(query)

        # Build ChromaDB where clause
        where: dict = {}
        if domain_filter and domain_filter in ("thermal", "electrical"):
            where["utility_domain"] = domain_filter

        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            where=where if where else None,
            chunk_types=chunk_types,
        )

        # Tag results with retrieval source
        for r in results:
            r["retrieval_source"] = "dense"

        return results
