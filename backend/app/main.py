import os
import time
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from app.config.settings import settings
from app.routing.query_router import query_router
from app.routing.retrieval_planner import retrieval_planner
from app.models.chat import ChatRequest, QueryResponse, DebugMetadata, TitleRequest, TitleResponse

from app.retrieval.dense_retriever import dense_retriever
from app.retrieval.sparse_retriever import sparse_retriever
from app.retrieval.colbert_retriever import colbert_retriever
from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.retrieval.reranker import reranker
from app.retrieval.parent_expansion import parent_expansion
from app.retrieval.context_compressor import context_compressor
from app.retrieval.multi_query import multi_query, hyde_retriever

from app.generation.mistral_client import mistral_client
from app.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.generation.verifier import verifier

app = FastAPI(title="CarbonTatva API", version="2.0.1")

# Configure CORS dynamically for Render + Vercel
allowed_origins = ["http://localhost:5173", "http://localhost:3000"]
if settings.FRONTEND_ORIGIN:
    allowed_origins.append(settings.FRONTEND_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.1"}

@app.post("/chat/title", response_model=TitleResponse)
async def generate_title_endpoint(request: TitleRequest):
    prompt = f"Generate a short 3-5 word title for a chat session that starts with this user query: '{request.query}'. Do not use quotes, punctuation, or prefixes."
    try:
        title = await mistral_client.generate(prompt, "You are a title generator.")
        return TitleResponse(title=title.strip(' ".'))
    except Exception as e:
        return TitleResponse(title=request.query[:30])

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    node_latency = {}
    debug_warnings = []
    
    # 1. Routing & Planning
    t0 = time.time()
    profile = query_router.route_query(request.query)
    plan = retrieval_planner.create_plan(
        query=request.query, 
        profile=profile, 
        mode=request.retrieval_mode, 
        options=request.advanced_options
    )
    node_latency["routing"] = (time.time() - t0) * 1000

    # Handle graceful fallback warnings
    if request.advanced_options.force_colbert and not settings.ENABLE_COLBERT:
        debug_warnings.append("Precision Retrieval (ColBERT) unavailable: index not found/disabled.")
    if request.advanced_options.force_hyde and not settings.ENABLE_HYDE:
        debug_warnings.append("Semantic Expansion (HyDE) unavailable: disabled in config.")
    if request.advanced_options.force_multi_query and not settings.ENABLE_MULTI_QUERY:
        debug_warnings.append("Query Expansion (Multi-query) unavailable: disabled in config.")

    # 2. Query Expansion
    t0 = time.time()
    search_queries = [request.query]
    
    if plan.use_multi_query:
        search_queries = await multi_query.generate_variants(request.query, profile)
        
    if plan.use_hyde:
        hyde_doc = await hyde_retriever.generate_hypothetical_document(request.query, profile)
        if hyde_doc and hyde_doc != request.query:
            search_queries.append(hyde_doc)
            
    node_latency["query_expansion"] = (time.time() - t0) * 1000

    # 3. Parallel Retrieval
    t0 = time.time()
    tasks = []
    
    for sq in search_queries:
        if plan.use_dense:
            tasks.append(dense_retriever.search(sq, top_k=plan.top_k_dense, query_profile=profile))
        if plan.use_sparse:
            tasks.append(sparse_retriever.search(sq, top_k=plan.top_k_sparse, query_profile=profile))
        if plan.use_colbert:
            tasks.append(colbert_retriever.search(sq, top_k=plan.top_k_colbert, query_profile=profile))

    if tasks:
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results_lists if isinstance(r, list)]
    else:
        valid_results = []
    node_latency["retrieval"] = (time.time() - t0) * 1000

    # 4. Fusion
    t0 = time.time()
    fused_candidates = reciprocal_rank_fusion(valid_results)
    node_latency["fusion"] = (time.time() - t0) * 1000

    # 5. Reranking
    t0 = time.time()
    if plan.use_reranking and fused_candidates:
        reranked_candidates = reranker.rerank(request.query, fused_candidates, top_k=plan.rerank_top_k)
    else:
        reranked_candidates = fused_candidates[:plan.rerank_top_k]
    node_latency["reranking"] = (time.time() - t0) * 1000

    # Store scores for debug if requested
    debug_scores = []
    if plan.return_debug and request.advanced_options.show_scores:
        for c in reranked_candidates:
            debug_scores.append({
                "chunk_id": c.get("chunk_id"),
                "rrf_score": c.get("rrf_score"),
                "rerank_score": c.get("rerank_score"),
                "sources": c.get("sources")
            })

    # 6. Advanced Context Processing
    final_candidates = reranked_candidates
    if plan.use_parent_expansion and final_candidates:
        t0 = time.time()
        final_candidates = parent_expansion.expand(final_candidates)
        node_latency["parent_expansion"] = (time.time() - t0) * 1000
        
    if plan.use_context_compression and final_candidates:
        t0 = time.time()
        final_candidates = context_compressor.compress(request.query, final_candidates)
        node_latency["context_compression"] = (time.time() - t0) * 1000

    # Limit max context chunks
    final_candidates = final_candidates[:plan.max_context_chunks]

    # Build Context String & Citations
    context_blocks = ""
    citations = []
    for doc in final_candidates:
        meta = doc.get("metadata", {})
        chunk_id = doc.get("evidence_chunk_id", doc.get("chunk_id", "unknown"))
        book = meta.get("book_name", "Unknown Book")
        ch = meta.get("chapter_title", "")
        sec = meta.get("section_title", "")
        p_start = meta.get("page_start", 0)
        
        cit_str = f"[source: {book}, {ch}, {sec}, page {p_start}]"
        context_blocks += f"{cit_str}\n{doc['text']}\n\n"
        
        citations.append({
            "chunk_id": chunk_id,
            "book": book,
            "chapter": ch,
            "section": sec,
            "page": p_start,
            "text_snippet": doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
        })

    # 7. Generation
    t0 = time.time()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=request.query,
        intent=profile.intent,
        utility_domain=profile.utility_domain,
        context_blocks=context_blocks
    )
    
    try:
        proposed_answer = await mistral_client.generate(user_prompt, SYSTEM_PROMPT)
    except Exception as e:
        proposed_answer = "I could not retrieve enough supported context for this question, or the API request failed."
        debug_warnings.append(str(e))
    node_latency["generation"] = (time.time() - t0) * 1000

    # 8. Verification
    if plan.use_verification:
        t0 = time.time()
        proposed_answer = await verifier.verify(request.query, proposed_answer, context_blocks, profile)
        node_latency["verification"] = (time.time() - t0) * 1000

    total_latency_ms = (time.time() - start_time) * 1000
    node_latency["total"] = total_latency_ms

    # Construct Response
    debug_metadata = DebugMetadata()
    if plan.return_debug:
        debug_metadata.warnings = debug_warnings
        if request.advanced_options.show_latency or request.retrieval_mode == "research":
            debug_metadata.latency_ms = node_latency
        if request.advanced_options.show_scores or request.retrieval_mode == "research":
            debug_metadata.scores = debug_scores
        if request.advanced_options.show_chunks or request.retrieval_mode == "research":
            debug_metadata.retrieved_chunks = [
                {"chunk_id": c.get("chunk_id"), "text": c.get("text")} 
                for c in final_candidates
            ]

    return QueryResponse(
        answer=proposed_answer,
        citations=citations,
        retrieval_profile=profile.model_dump(),
        retrieval_plan=plan.model_dump(),
        debug=debug_metadata
    )
