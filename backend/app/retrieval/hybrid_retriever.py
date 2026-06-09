from typing import List, Dict, Any

def reciprocal_rank_fusion(results_lists: List[List[Dict[str, Any]]], k=60) -> List[Dict[str, Any]]:
    """
    Fuses multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Deduplicates by chunk_id.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    source_map: Dict[str, List[str]] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            chunk_id = doc["chunk_id"]
            
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = doc
                source_map[chunk_id] = []
                
            source_map[chunk_id].append(doc["source"])
                
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            
            # RRF Formula: 1 / (k + rank)
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for chunk_id, score in sorted_chunks:
        doc = chunk_map[chunk_id]
        doc["rrf_score"] = score
        doc["sources"] = source_map[chunk_id]
        fused_results.append(doc)

    return fused_results
