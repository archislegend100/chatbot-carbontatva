import os
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.routing.query_router import QueryProfile

class SparseRetriever:
    def __init__(self, index_path="data/indexes/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_data = [] # Stores dict with id, text, metadata
        self._load_index()
        
        self.domain_synonyms = {
            "vfd": ["variable frequency drive", "variable speed drive", "vsd"],
            "pf": ["power factor"],
            "whr": ["waste heat recovery"],
            "sec": ["specific energy consumption", "energy intensity"],
            "boiler": ["steam generator"],
            "flue gas": ["stack gas"],
            "condensate": ["condensate recovery", "return condensate"],
            "compressed air": ["air compressor", "pneumatic system"],
            "motor": ["electric motor", "induction motor"],
            "pump": ["pumping system"],
            "fan": ["blower", "draught fan"],
            "lighting": ["illumination"],
            "transformer": ["distribution transformer"]
        }

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data.get('bm25')
                self.corpus_data = data.get('corpus_data', [])

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Called during ingestion to build BM25 index."""
        self.corpus_data = chunks
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'corpus_data': self.corpus_data}, f)

    def _expand_query(self, query: str) -> str:
        q_lower = query.lower()
        expanded = [q_lower]
        for term, synonyms in self.domain_synonyms.items():
            if term in q_lower:
                expanded.extend(synonyms)
            # also check if synonym is in query to add base term
            for syn in synonyms:
                if syn in q_lower and term not in expanded:
                    expanded.append(term)
        return " ".join(set(expanded))

    async def search(self, query: str, top_k: int = 15, query_profile: QueryProfile = None) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus_data:
            return []
            
        expanded_query = self._expand_query(query)
        tokenized_query = expanded_query.split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        candidates = []
        for idx in top_n:
            if scores[idx] <= 0:
                continue
                
            chunk = self.corpus_data[idx]
            
            # Boost specific chunk types if requested
            score_boost = 0.0
            if query_profile:
                chunk_type = chunk.get("metadata", {}).get("chunk_type", "")
                if query_profile.needs_table and chunk_type == "table_chunk":
                    score_boost += 5.0 # BM25 scores are scale-dependent, boost accordingly
                if query_profile.needs_formula and chunk_type == "formula_chunk":
                    score_boost += 5.0
                    
            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": float(scores[idx]) + score_boost,
                "source": "sparse"
            })
            
        return candidates

sparse_retriever = SparseRetriever()
