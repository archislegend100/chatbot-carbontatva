"""
indexing/vector_store.py
=========================
ChromaDB vector store wrapper for the Industrial Energy Efficiency Copilot.

This module provides a clean interface to:
- Add document chunks with their embeddings and metadata
- Query by dense vector similarity
- Filter by metadata (domain, chapter, chunk_type, etc.)
- Fetch chunks by ID

ChromaDB is used in persistent mode — the index is saved to disk
at CHROMA_DIR and reloaded on startup. No separate server needed.

Usage:
    store = VectorStore()
    store.add_chunks(chunks, embeddings)
    results = store.query(query_embedding, top_k=10, where={"utility_domain": "thermal"})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import settings
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger
from app.models.schemas import DocumentChunk, SourceCitation, UtilityDomain

logger = get_logger(__name__)

COLLECTION_NAME = "energy_efficiency_copilot"


class VectorStore:
    """
    ChromaDB persistent vector store.

    The collection stores:
    - documents: chunk text
    - embeddings: 384-dim float vectors (all-MiniLM-L6-v2)
    - metadatas: full ChunkMetadata as flat key-value dict
    - ids: stable chunk IDs
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or settings.CHROMA_DIR_ABS)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            import logging
            import os
            # Suppress chromadb posthog telemetry noise (chromadb 0.5.x + posthog 7.x compat issue)
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
            logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(
                f"Collection '{COLLECTION_NAME}' ready. "
                f"Items: {self._collection.count()}"
            )
        return self._collection

    @property
    def collection(self):
        return self._get_collection()

    def count(self) -> int:
        """Return total number of stored chunks."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 500,
    ) -> None:
        """
        Add chunks and their embeddings to the collection.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Corresponding list of embedding vectors.
            batch_size: Batch size for ChromaDB upsert calls.
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have same length"

        total = len(chunks)
        logger.info(f"Adding {total} chunks to vector store...")

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = [self._flatten_metadata(c) for c in batch_chunks]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

            if batch_end % 2000 == 0 or batch_end == total:
                logger.info(f"  Indexed {batch_end}/{total} chunks")

        logger.info(f"Vector store now has {self.count()} total chunks")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: Optional[dict[str, Any]] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store by dense similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: ChromaDB metadata filter dict.
            chunk_types: If set, filter to these chunk types.

        Returns:
            List of result dicts with keys: chunk_id, text, metadata, distance.
        """
        # Build where clause
        filter_clause: Optional[dict] = where.copy() if where else None

        if chunk_types and len(chunk_types) > 0:
            type_filter = {"chunk_type": {"$in": chunk_types}}
            if filter_clause:
                filter_clause = {"$and": [filter_clause, type_filter]}
            else:
                filter_clause = type_filter

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.count()),
                where=filter_clause if filter_clause else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Vector store query failed: {e}")
            return []

        output = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": max(0.0, 1.0 - results["distances"][0][i]),  # Cosine sim
                })

        return output

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Fetch a specific chunk by its ID."""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if result and result["ids"]:
                return {
                    "chunk_id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.warning(f"get_chunk_by_id({chunk_id}) failed: {e}")
        return None

    def reset(self) -> None:
        """Delete and recreate the collection (used for re-indexing)."""
        logger.warning(f"Resetting collection '{COLLECTION_NAME}'...")
        client = self._get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        self._get_collection()

    def _flatten_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """
        Convert ChunkMetadata to a flat dict suitable for ChromaDB.
        ChromaDB only supports str/int/float/bool values (no nested objects).
        Lists are JSON-serialized.
        """
        m = chunk.metadata
        return {
            "document_id": m.document_id,
            "book_name": m.book_name,
            "utility_domain": m.utility_domain.value,
            "chapter_num": m.chapter_num or 0,
            "chapter_title": m.chapter_title or "",
            "section_title": m.section_title or "",
            "subsection_title": m.subsection_title or "",
            "page_start": m.page_start,
            "page_end": m.page_end,
            "chunk_type": m.chunk_type.value,
            "content_type": m.content_type.value,
            "equipment_tags": json.dumps(m.equipment_tags),
            "concept_tags": json.dumps(m.concept_tags),
            "word_count": m.word_count,
            "char_count": m.char_count,
        }

    def result_to_citation(self, result: dict[str, Any]) -> SourceCitation:
        """Convert a query result dict to a SourceCitation."""
        meta = result["metadata"]
        return SourceCitation(
            chunk_id=result["chunk_id"],
            book_name=meta.get("book_name", ""),
            utility_domain=meta.get("utility_domain", "unknown"),
            chapter_title=meta.get("chapter_title") or None,
            section_title=meta.get("section_title") or None,
            page_start=meta.get("page_start", 0),
            page_end=meta.get("page_end", 0),
            relevance_score=result.get("score", 0.0),
            snippet=result["text"][:250] + "..." if len(result["text"]) > 250 else result["text"],
        )
