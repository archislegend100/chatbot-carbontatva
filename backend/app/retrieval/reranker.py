from typing import List, Dict, Any
from app.config.settings import settings

class Reranker:
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self._model = None

    def _get_model(self):
        # Lazy load reranker to save memory if not immediately needed
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        if not settings.ENABLE_RERANKER:
            # Bypass reranking and just return top_k candidates, preserving their existing scores
            for doc in candidates:
                if "rerank_score" not in doc:
                    doc["rerank_score"] = doc.get("score", 0.0)
            return candidates[:top_k]
            
        model = self._get_model()
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc["text"]] for doc in candidates]
        
        scores = model.predict(pairs)
        
        # Assign scores and sort
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
            
        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]

reranker = Reranker()
