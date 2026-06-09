"""
indexing/embedder.py
====================
Dense embedding generation for the Industrial Energy Efficiency Copilot.

Uses Mistral API to generate embeddings for all document chunks. 
These are stored in ChromaDB.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Optional

from app.config.settings import settings
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger
from app.generation.mistral_client import mistral_client

logger = get_logger(__name__)

class Embedder:
    """
    Wrapper around Mistral Embeddings API for generating dense embeddings.
    """
    def __init__(self):
        pass

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using Mistral API.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} texts via Mistral API")
        
        results = []
        batch_size = 32 # Mistral API batch limit
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                future = asyncio.run_coroutine_threadsafe(mistral_client.embed_batch(batch), loop)
                embeddings = future.result()
            else:
                embeddings = loop.run_until_complete(mistral_client.embed_batch(batch))
            results.extend(embeddings)

        return results

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            future = asyncio.run_coroutine_threadsafe(mistral_client.embed_batch([query]), loop)
            embeddings = future.result()
        else:
            embeddings = loop.run_until_complete(mistral_client.embed_batch([query]))
            
        if embeddings:
            return embeddings[0]
        return []
