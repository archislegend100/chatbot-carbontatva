"""
app/services/__init__.py
=========================
Singleton service managing all RAG components.

v2 additions:
- LangGraph compiled graph
- Ollama as primary LLM (with fallback chain)
- Query result cache
- Ollama availability check at startup
"""

from __future__ import annotations

from typing import Optional, Any

from app.indexing.embedder import Embedder
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.router.query_classifier import QueryClassifier
from app.router.tool_router import ToolRouter
from app.generation.llm_client import get_llm_client
from app.cache.query_cache import QueryCache
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class IndexService:
    """
    Application-level service managing all RAG components.
    Initialized once at startup; shared across all requests.
    """

    def __init__(self):
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.bm25_index: Optional[BM25Index] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        self.sparse_retriever: Optional[SparseRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[Reranker] = None
        self.classifier: Optional[QueryClassifier] = None
        self.tool_router: Optional[ToolRouter] = None
        self.llm_client = None
        self.langgraph = None          # Compiled LangGraph
        self.query_cache: Optional[QueryCache] = None
        self._loaded = False
        self._ollama_available = False

    @property
    def is_loaded(self) -> bool:
        # If already loaded, return cached value
        if self._loaded:
            return True
        # Lazily re-check if indexes have become available since startup
        if self.bm25_index and self.vector_store:
            bm25_ok = self.bm25_index.is_loaded or self.bm25_index.load()
            chroma_ok = self.vector_store.count() > 0
            if bm25_ok and chroma_ok:
                self._loaded = True
                logger.info(
                    f"Indexes became available (lazy check): "
                    f"{self.vector_store.count()} chunks"
                )
        return self._loaded

    @property
    def chunk_count(self) -> int:
        return self.vector_store.count() if self.vector_store else 0

    def reload(self) -> bool:
        """Force a full re-initialization of all components. Used after ingestion."""
        logger.info("Force-reloading IndexService...")
        self._loaded = False
        return self.load()

    def load(self) -> bool:
        """Initialize all components. Returns True if indexes are ready."""
        logger.info("Initializing IndexService v2 (LangGraph + Ollama)...")

        # ---- Core components ----
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.reranker = Reranker()
        self.query_cache = QueryCache()

        # ---- LLM Client: Ollama first, fallback chain ----
        self.llm_client = get_llm_client()
        self._check_ollama_sync()

        # ---- Check indexes ----
        bm25_loaded = self.bm25_index.load()
        chroma_count = self.vector_store.count()

        if not bm25_loaded or chroma_count == 0:
            logger.warning(
                "Indexes not found or empty. Run: python scripts/ingest.py"
            )
            self._loaded = False
        else:
            logger.info(
                f"Indexes ready: {chroma_count} chunks (ChromaDB) + "
                f"{self.bm25_index.chunk_count} (BM25)"
            )
            self._loaded = True

        # ---- Retrieval pipeline ----
        self.dense_retriever = DenseRetriever(self.embedder, self.vector_store)
        self.sparse_retriever = SparseRetriever(self.bm25_index)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever,
            self.sparse_retriever,
        )

        # ---- Legacy v1 router (kept as fallback for /api/query) ----
        self.classifier = QueryClassifier(self.llm_client)
        self.tool_router = ToolRouter(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            classifier=self.classifier,
            llm_client=self.llm_client,
            vector_store=self.vector_store,
        )

        # ---- LangGraph pipeline ----
        self._build_langgraph()

        return self._loaded

    def _check_ollama_sync(self):
        """Quick synchronous Ollama availability check."""
        try:
            import httpx
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            self._ollama_available = resp.status_code == 200
            if self._ollama_available:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.info(
                    f"Ollama available: {len(models)} model(s) loaded — "
                    f"{', '.join(models[:3])}"
                )
            else:
                logger.warning("Ollama running but returned non-200")
        except Exception:
            self._ollama_available = False
            logger.warning(
                "Ollama not reachable at localhost:11434. "
                "Falling back to configured LLM provider. "
                "Start Ollama: ollama serve"
            )

    def _build_langgraph(self):
        """Compile the LangGraph pipeline."""
        try:
            from app.graph.graph import build_graph
            self.langgraph = build_graph(
                llm_client=self.llm_client,
                hybrid_retriever=self.hybrid_retriever,
                reranker=self.reranker,
                vector_store=self.vector_store,
            )
        except Exception as e:
            logger.error(f"LangGraph compilation failed: {e}")
            self.langgraph = None


# ---- Singleton ----
_service: Optional[IndexService] = None


def get_service() -> IndexService:
    global _service
    if _service is None:
        _service = IndexService()
    return _service
