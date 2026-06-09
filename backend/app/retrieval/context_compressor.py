import re
import nltk
from typing import List, Dict, Any
from app.config.settings import settings

# Download punkt lazily
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ContextCompressor:
    def __init__(self):
        # We reuse the dense embedding model or a lighter one for sentence-level similarity
        # Here we just use the existing one to avoid loading another model
        self.model_name = settings.EMBEDDING_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compress(self, query: str, candidates: List[Dict[str, Any]], similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Extractive compression: Keeps only sentences directly relevant to the query.
        Preserves formulas, tables, numbers, and units exactly.
        """
        if not candidates:
            return []
            
        model = self._get_model()
        query_embedding = model.encode([query])[0]
        
        compressed_candidates = []
        
        for doc in candidates:
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            # Skip compression for table chunks, keep them whole
            if metadata.get("chunk_type") == "table_chunk" or metadata.get("has_table"):
                compressed_candidates.append(doc)
                continue
                
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                continue
                
            # Embed all sentences
            sentence_embeddings = model.encode(sentences)
            
            # Calculate similarity
            # Since embeddings are normalized, dot product is cosine similarity
            similarities = [float(query_embedding @ sent_emb.T) for sent_emb in sentence_embeddings]
            
            kept_sentences = []
            for i, sent in enumerate(sentences):
                # Keep sentence if highly similar
                if similarities[i] > similarity_threshold:
                    kept_sentences.append(sent)
                    continue
                    
                # Keep sentence if it contains a formula or critical numeric data/units
                if re.search(r'(\b\w+\s*=\s*[^a-zA-Z]|\bη\b|\b%\b|\bkW\b|\bkWh\b|\b°C\b)', sent):
                    kept_sentences.append(sent)
                    continue
            
            # If compression drops everything but we had a high rerank score, fall back to keeping original
            if not kept_sentences:
                compressed_candidates.append(doc)
            else:
                doc_copy = doc.copy()
                doc_copy["text"] = " ".join(kept_sentences)
                doc_copy["compression_ratio"] = len(doc_copy["text"]) / max(1, len(text))
                compressed_candidates.append(doc_copy)
                
        return compressed_candidates

context_compressor = ContextCompressor()
