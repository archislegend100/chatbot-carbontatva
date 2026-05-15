"""
indexing/embedder.py
====================
Dense embedding generation for the Industrial Energy Efficiency Copilot.

Uses sentence-transformers to generate 384-dimensional embeddings
for all document chunks. These are stored in ChromaDB.

The embedding model runs locally — no API key required.
Default model: sentence-transformers/all-MiniLM-L6-v2

For better quality (at the cost of higher compute):
    EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

Usage:
    embedder = Embedder()
    embeddings = embedder.embed_texts(["text1", "text2"])
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Wrapper around sentence-transformers for generating dense embeddings.

    Lazy-loads the model on first use to avoid slow startup times
    when the model is not needed (e.g., during ingestion status checks).

    Args:
        model_name: Hugging Face model name or path.
        device: "cpu" or "cuda" (auto-detected if not set).
        batch_size: Number of texts to embed per batch.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        device: str = settings.EMBEDDING_DEVICE,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None  # Lazy loaded

    def _load_model(self):
        """Load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Embedding model loaded. Dim={self._model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            show_progress: Show tqdm progress bar for large batches.

        Returns:
            List of embedding vectors (as Python float lists).
        """
        if not texts:
            return []

        self._load_model()
        logger.debug(f"Embedding {len(texts)} texts (batch_size={self.batch_size})")

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress and len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Cosine similarity via dot product
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        
        Args:
            query: The search query string.

        Returns:
            Embedding vector.
        """
        self._load_model()
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
