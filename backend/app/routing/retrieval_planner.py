from app.routing.query_router import QueryProfile
from app.models.chat import AdvancedRetrievalOptions, RetrievalPlanOutput
from app.config.settings import settings

class RetrievalPlanner:
    def create_plan(
        self, 
        query: str, 
        profile: QueryProfile, 
        mode: str, 
        options: AdvancedRetrievalOptions
    ) -> RetrievalPlanOutput:
        
        mode = mode.lower()
        if mode not in ["auto", "fast", "deep", "research"]:
            mode = settings.DEFAULT_RETRIEVAL_MODE

        plan = RetrievalPlanOutput()

        # BASE CONFIG FOR ALL MODES
        plan.use_dense = True
        plan.use_sparse = True
        plan.use_reranking = True

        if mode == "fast":
            # FAST MODE
            plan.use_colbert = False
            plan.use_hyde = False
            plan.use_multi_query = False
            plan.use_verification = False
            plan.use_parent_expansion = False
            plan.use_context_compression = False
            
            plan.top_k_dense = settings.FAST_TOP_K_DENSE
            plan.top_k_sparse = settings.FAST_TOP_K_SPARSE
            plan.top_k_colbert = 0
            plan.rerank_top_k = settings.FAST_RERANK_TOP_K
            plan.max_context_chunks = settings.FAST_MAX_CONTEXT_CHUNKS
            plan.return_debug = False

        elif mode == "deep" or mode == "research":
            # DEEP / RESEARCH MODE
            plan.use_parent_expansion = True
            plan.use_context_compression = True
            
            # Allow colbert, multi_query, hyde if useful
            plan.use_colbert = profile.needs_colbert and settings.ENABLE_COLBERT
            plan.use_multi_query = settings.ENABLE_MULTI_QUERY and profile.intent in ["explainer", "troubleshoot", "comparison", "summarize"]
            plan.use_hyde = profile.needs_hyde and settings.ENABLE_HYDE
            plan.use_verification = profile.needs_verification and settings.ENABLE_VERIFICATION
            
            if mode == "deep":
                plan.top_k_dense = settings.DEEP_TOP_K_DENSE
                plan.top_k_sparse = settings.DEEP_TOP_K_SPARSE
                plan.top_k_colbert = settings.DEEP_TOP_K_COLBERT
                plan.rerank_top_k = settings.DEEP_RERANK_TOP_K
                plan.max_context_chunks = settings.DEEP_MAX_CONTEXT_CHUNKS
                plan.return_debug = False
            else:
                plan.top_k_dense = settings.RESEARCH_TOP_K_DENSE
                plan.top_k_sparse = settings.RESEARCH_TOP_K_SPARSE
                plan.top_k_colbert = settings.RESEARCH_TOP_K_COLBERT
                plan.rerank_top_k = settings.RESEARCH_RERANK_TOP_K
                plan.max_context_chunks = settings.RESEARCH_MAX_CONTEXT_CHUNKS
                plan.return_debug = True

            # Apply manual overrides ONLY in Research Mode
            if mode == "research":
                if options.force_colbert:
                    plan.use_colbert = True
                if options.force_hyde:
                    plan.use_hyde = True
                if options.force_multi_query:
                    plan.use_multi_query = True
                    
        else:
            # AUTO MODE
            plan.use_parent_expansion = profile.needs_parent_expansion
            plan.use_context_compression = True # Typically safe for expanded text
            plan.use_colbert = profile.needs_colbert and settings.ENABLE_COLBERT
            plan.use_hyde = profile.needs_hyde and settings.ENABLE_HYDE
            plan.use_multi_query = settings.ENABLE_MULTI_QUERY and profile.intent in ["explainer", "troubleshoot", "comparison", "summarize"]
            plan.use_verification = profile.needs_verification and settings.ENABLE_VERIFICATION
            
            # Balances latency and recall
            plan.top_k_dense = settings.DEEP_TOP_K_DENSE
            plan.top_k_sparse = settings.DEEP_TOP_K_SPARSE
            plan.top_k_colbert = settings.DEEP_TOP_K_COLBERT
            plan.rerank_top_k = settings.DEEP_RERANK_TOP_K
            plan.max_context_chunks = settings.DEEP_MAX_CONTEXT_CHUNKS
            plan.return_debug = False

        # Additional safeguards: Check global enables
        if not settings.ENABLE_COLBERT:
            plan.use_colbert = False
        if not settings.ENABLE_HYDE:
            plan.use_hyde = False
        if not settings.ENABLE_MULTI_QUERY:
            plan.use_multi_query = False

        # Apply output debug formatting from options
        if options.show_chunks or options.show_scores or options.show_latency:
            plan.return_debug = True

        return plan

retrieval_planner = RetrievalPlanner()
