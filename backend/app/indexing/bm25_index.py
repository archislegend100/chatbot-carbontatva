"""
indexing/bm25_index.py
======================
BM25 sparse retrieval index for the Industrial Energy Efficiency Copilot.

BM25 (Best Match 25) is a bag-of-words relevance function that works well
for exact keyword matching — complementing dense vector search.

The index is built over all chunk texts during ingestion and:
- Serialized to disk as a pickle file
- Reloaded on server startup
- Used in parallel with dense retrieval (hybrid search)

Why BM25 in addition to dense search?
- Dense search excels at semantic similarity  
- BM25 excels at exact term matching (e.g., specific formula names, standards,
  equipment model codes, or acronyms like "VFD", "LMTD", "BEP")
- Combining both yields better coverage

Usage:
    index = BM25Index()
    index.build(chunks)
    index.save()
    results = index.query("excess air combustion", top_k=20)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

from app.config.settings import settings
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger
from app.models.schemas import DocumentChunk

logger = get_logger(__name__)


# Simple tokenizer for BM25
def _tokenize(text: str) -> list[str]:
    """
    Basic tokenizer: lowercase, split on non-alphanumeric, filter short tokens.
    """
    import re
    tokens = re.split(r"[^a-zA-Z0-9]", text.lower())
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """
    BM25 retrieval index over document chunks.

    Builds the index from chunk texts during ingestion, then
    persists it to disk for reuse.

    Args:
        index_path: Path to save/load the serialized index.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or settings.BM25_INDEX_PATH_ABS
        self._index = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadatas: list[dict] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        """
        Build the BM25 index from a list of document chunks.

        Args:
            chunks: All DocumentChunk objects to index.
        """
        from rank_bm25 import BM25Okapi

        logger.info(f"Building BM25 index over {len(chunks)} chunks...")

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunk_texts = [c.text for c in chunks]
        self._chunk_metadatas = [c.metadata.model_dump() for c in chunks]

        tokenized_corpus = [_tokenize(t) for t in self._chunk_texts]
        self._index = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built.")

    def save(self) -> None:
        """Serialize the index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "index": self._index,
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "chunk_metadatas": self._chunk_metadatas,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = self.index_path.stat().st_size / 1024 / 1024
        logger.info(f"BM25 index saved to {self.index_path} ({size_mb:.1f} MB)")

    def load(self) -> bool:
        """
        Load the index from disk.

        Returns:
            True if loaded successfully, False if index file doesn't exist.
        """
        if not self.index_path.exists():
            logger.info("BM25 index not found on disk (run ingestion first).")
            return False

        logger.info(f"Loading BM25 index from {self.index_path}...")
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)

        self._index = payload["index"]
        self._chunk_ids = payload["chunk_ids"]
        self._chunk_texts = payload["chunk_texts"]
        self._chunk_metadatas = payload.get("chunk_metadatas", [])

        logger.info(f"BM25 index loaded. {len(self._chunk_ids)} chunks.")
        return True

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        domain_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Query the BM25 index.

        Args:
            query_text: The search query string.
            top_k: Number of results to return.
            domain_filter: If set ("thermal"/"electrical"), filter results.

        Returns:
            List of result dicts with keys: chunk_id, text, metadata, score.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not loaded. Call build() or load() first.")

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Build indexed results
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed:
            if score <= 0:
                continue
            if len(results) >= top_k:
                break

            meta = self._chunk_metadatas[idx] if idx < len(self._chunk_metadatas) else {}

            # Domain filter
            if domain_filter and meta.get("utility_domain") != domain_filter:
                continue

            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": self._chunk_texts[idx],
                "metadata": meta,
                "score": float(score),
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)
