import os
import hashlib
from typing import List
from diskcache import Cache
from sentence_transformers import SentenceTransformer
from app.config.settings import settings

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.batch_size = settings.EMBED_BATCH_SIZE
        self._model = None
        
        # Disk cache for embeddings to avoid recomputing
        cache_dir = "data/cache/embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _generate_cache_key(self, text: str) -> str:
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self.model_name}_{text_hash}"

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
            
        cache_key = self._generate_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True).tolist()
        self.cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
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
            model = self._get_model()
            # process in batches
            for i in range(0, len(texts_to_compute), self.batch_size):
                batch_texts = texts_to_compute[i:i + self.batch_size]
                batch_indices = indices_to_compute[i:i + self.batch_size]
                
                embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
                
                for j, emb in enumerate(embeddings):
                    original_idx = batch_indices[j]
                    text = texts[original_idx]
                    results[original_idx] = emb
                    self.cache[self._generate_cache_key(text)] = emb

        return results

embedding_service = EmbeddingService()
