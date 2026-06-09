from typing import List, Dict, Any
import logging
from app.retrieval.dense_retriever import dense_retriever

logger = logging.getLogger(__name__)

class ParentExpansion:
    def expand(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replaces child chunks with their expanded parent context while preserving child citations.
        """
        expanded_candidates = []
        parent_ids_to_fetch = set()
        chunk_to_parent_map = {}
        
        for doc in candidates:
            metadata = doc.get("metadata", {})
            parent_id = metadata.get("parent_id")
            
            if parent_id and parent_id != doc["chunk_id"]:
                parent_ids_to_fetch.add(parent_id)
                chunk_to_parent_map[doc["chunk_id"]] = parent_id
            else:
                # Already a parent chunk or no parent exists
                expanded_candidates.append(doc)
                
        if parent_ids_to_fetch:
            try:
                # Fetch parent chunks from ChromaDB
                # This requires fetching by ID.
                results = dense_retriever.collection.get(ids=list(parent_ids_to_fetch))
                
                parent_map = {}
                if results and results["ids"]:
                    for i, doc_id in enumerate(results["ids"]):
                        parent_map[doc_id] = {
                            "chunk_id": doc_id,
                            "text": results["documents"][i],
                            "metadata": results["metadatas"][i]
                        }
                        
                for doc in candidates:
                    chunk_id = doc["chunk_id"]
                    if chunk_id in chunk_to_parent_map:
                        p_id = chunk_to_parent_map[chunk_id]
                        if p_id in parent_map:
                            parent_doc = parent_map[p_id].copy()
                            
                            # Preserve original child info for accurate citation
                            parent_doc["evidence_chunk_id"] = chunk_id
                            parent_doc["rerank_score"] = doc.get("rerank_score", 0.0)
                            parent_doc["rrf_score"] = doc.get("rrf_score", 0.0)
                            parent_doc["source"] = "parent_expanded"
                            
                            expanded_candidates.append(parent_doc)
                        else:
                            # Fallback to original child if parent not found
                            expanded_candidates.append(doc)
            except Exception as e:
                logger.error(f"Error during parent expansion: {e}")
                # Fallback to original candidates
                return candidates
                
        # Deduplicate
        seen = set()
        final_candidates = []
        for doc in expanded_candidates:
            cid = doc.get("evidence_chunk_id", doc["chunk_id"])
            if cid not in seen:
                seen.add(cid)
                final_candidates.append(doc)
                
        return final_candidates

parent_expansion = ParentExpansion()
