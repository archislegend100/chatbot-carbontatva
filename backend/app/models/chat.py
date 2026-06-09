from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List

class AdvancedRetrievalOptions(BaseModel):
    force_colbert: bool = False
    force_hyde: bool = False
    force_multi_query: bool = False
    show_chunks: bool = False
    show_scores: bool = False
    show_latency: bool = False

class TitleRequest(BaseModel):
    query: str

class TitleResponse(BaseModel):
    title: str

class ChatRequest(BaseModel):
    query: str
    retrieval_mode: Literal["auto", "fast", "deep", "research"] = "auto"
    advanced_options: AdvancedRetrievalOptions = Field(default_factory=AdvancedRetrievalOptions)

class RetrievalPlanOutput(BaseModel):
    use_dense: bool = True
    use_sparse: bool = True
    use_reranking: bool = True
    use_colbert: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    use_parent_expansion: bool = False
    use_context_compression: bool = False
    use_verification: bool = False
    top_k_dense: int = 15
    top_k_sparse: int = 15
    top_k_colbert: int = 0
    rerank_top_k: int = 5
    max_context_chunks: int = 4
    return_debug: bool = False

class DebugMetadata(BaseModel):
    warnings: List[str] = []
    retrieved_chunks: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    latency_ms: Dict[str, float] = {}
    
class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_profile: Dict[str, Any] = {}
    retrieval_plan: Dict[str, Any] = {}
    debug: DebugMetadata = Field(default_factory=DebugMetadata)
