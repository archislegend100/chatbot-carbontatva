import os
import hashlib
import asyncio
from typing import List
from diskcache import Cache
from app.config.settings import settings
from app.generation.mistral_client import mistral_client

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.batch_size = settings.EMBED_BATCH_SIZE
        
        # Disk cache for embeddings to avoid recomputing
        cache_dir = "data/cache/embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)

    def _generate_cache_key(self, text: str) -> str:
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self.model_name}_{text_hash}"

    async def embed_text(self, text: str) -> List[float]:
        """Asynchronously embed a single text chunk via Mistral API."""
        if not text:
            return []
            
        cache_key = self._generate_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Call Mistral API for a single text
        embeddings = await mistral_client.embed_batch([text])
        if embeddings and len(embeddings) > 0:
            self.cache[cache_key] = embeddings[0]
            return embeddings[0]
        return []

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronously embed a batch of texts via Mistral API (used by ingestion scripts)."""
        if not texts:
            return []
            
        results = []
        texts_to_compute = []
        indices_to_compute = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                results.append(None) # Placeholder
                texts_to_compute.append(text)
                indices_to_compute.append(i)

        # Compute missing embeddings
        if texts_to_compute:
            # We must run the async mistral client within a synchronous event loop context
            # Since this is strictly for the ingest script, we can safely use asyncio.run
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            for i in range(0, len(texts_to_compute), self.batch_size):
                batch_texts = texts_to_compute[i:i + self.batch_size]
                batch_indices = indices_to_compute[i:i + self.batch_size]
                
                # Execute API call
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    # In a running loop (like an API endpoint calling a sync function, not recommended but safe here)
                    future = asyncio.run_coroutine_threadsafe(mistral_client.embed_batch(batch_texts), loop)
                    embeddings = future.result()
                else:
                    embeddings = loop.run_until_complete(mistral_client.embed_batch(batch_texts))
                
                for j, emb in enumerate(embeddings):
                    original_idx = batch_indices[j]
                    text = texts[original_idx]
                    results[original_idx] = emb
                    self.cache[self._generate_cache_key(text)] = emb

        return results

embedding_service = EmbeddingService()
