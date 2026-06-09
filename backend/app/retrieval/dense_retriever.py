from typing import List, Dict, Any
import chromadb
from app.indexing.embedding_service import embedding_service
from app.routing.query_router import QueryProfile

class DenseRetriever:
    def __init__(self, persist_directory="data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Using a unified collection for both domains
        self.collection = self.client.get_or_create_collection(name="carbontatva_docs")

    async def search(self, query: str, top_k: int = 15, query_profile: QueryProfile = None) -> List[Dict[str, Any]]:
        query_embedding = embedding_service.embed_text(query)
        
        where_filter = {}
        # Optional: Apply domain filter if domain is strictly known and not mixed
        # if query_profile and query_profile.utility_domain in ["thermal", "electrical"]:
        #    where_filter["utility_domain"] = query_profile.utility_domain
            
        # Optional: Boost certain chunks based on intent via post-processing or hybrid weighing
        # Chromadb doesn't natively support dynamic boosting on boolean metadata, so we retrieve more and rerank.

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )

        candidates = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                
                # Boost specific chunk types if requested
                score_boost = 0.0
                if query_profile:
                    chunk_type = metadata.get("chunk_type", "")
                    if query_profile.needs_table and chunk_type == "table_chunk":
                        score_boost += 0.2
                    if query_profile.needs_formula and chunk_type == "formula_chunk":
                        score_boost += 0.2

                candidates.append({
                    "chunk_id": doc_id,
                    "text": doc_text,
                    "metadata": metadata,
                    "score": (1.0 / (1.0 + distance)) + score_boost, # Convert distance to similarity-like score
                    "source": "dense"
                })

        return candidates

dense_retriever = DenseRetriever()
