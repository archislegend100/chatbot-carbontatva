import asyncio
from typing import List, Dict, Any
from app.config.settings import settings
from app.routing.query_router import QueryProfile
from app.generation.mistral_client import mistral_client

class MultiQueryRetriever:
    def __init__(self):
        self.enabled = settings.ENABLE_MULTI_QUERY
        self.max_variants = settings.MAX_QUERY_VARIANTS

    async def generate_variants(self, query: str, query_profile: QueryProfile) -> List[str]:
        if not self.enabled:
            return [query]
            
        # Only use multi-query for vague/hard queries
        if query_profile.intent in ["formula", "table_lookup", "navigation"]:
            return [query]
            
        prompt = f"""You are an AI assistant for industrial energy efficiency.
Generate up to {self.max_variants} different search queries to find relevant information for this user question.
Output only the queries, one per line. Do not number them. Do not add quotes.

User question: {query}
"""
        try:
            response = await mistral_client.generate(prompt)
            variants = [q.strip() for q in response.split("\n") if q.strip()]
            
            # Add original query
            if query not in variants:
                variants.insert(0, query)
                
            return variants[:self.max_variants + 1]
        except Exception:
            return [query]

class HydeRetriever:
    def __init__(self):
        self.enabled = settings.ENABLE_HYDE

    async def generate_hypothetical_document(self, query: str, query_profile: QueryProfile) -> str:
        if not self.enabled or not query_profile.needs_hyde:
            return query
            
        prompt = f"""You are an industrial engineering expert.
Write a short, factual, hypothetical paragraph that answers this question. 
Do not explain, just provide the expected technical text that might be found in a BEE Utility manual.

Question: {query}
"""
        try:
            response = await mistral_client.generate(prompt)
            return response.strip()
        except Exception:
            return query

multi_query = MultiQueryRetriever()
hyde_retriever = HydeRetriever()
