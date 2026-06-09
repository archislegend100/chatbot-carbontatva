from typing import List, Dict, Any
from app.indexing.embedding_service import embedding_service
from app.indexing.vector_store import VectorStore
from app.routing.query_router import QueryProfile

class DenseRetriever:
    def __init__(self):
        self.store = VectorStore()

    async def search(self, query: str, top_k: int = 15, query_profile: QueryProfile = None) -> List[Dict[str, Any]]:
        query_embedding = await embedding_service.embed_text(query)
        
        results = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        candidates = []
        for r in results:
            score_boost = 0.0
            if query_profile:
                chunk_type = r["metadata"].get("chunk_type", "")
                if query_profile.needs_table and chunk_type == "table_chunk":
                    score_boost += 0.2
                if query_profile.needs_formula and chunk_type == "formula_chunk":
                    score_boost += 0.2
                    
            candidates.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": r["score"] + score_boost,
                "source": "dense"
            })
            
        return candidates

dense_retriever = DenseRetriever()
