from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Base LLM Settings
    MISTRAL_API_KEY: str = "VHs0tggpboiuLlgGQWQewtA6OnO9IKWl"
    MISTRAL_MODEL: str = "mistral-small-latest"
    MISTRAL_TEMPERATURE: float = 0.2
    MISTRAL_MAX_TOKENS: int = 1200
    
    # Embedding Models
    EMBEDDING_MODEL: str = "mistral-embed"
    EMBED_BATCH_SIZE: int = 32
    
    # Reranking Models
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = False  # Disabled for Render Free Tier
    
    # Feature Toggles
    ENABLE_COLBERT: bool = False
    COLBERT_INDEX_PATH: str = "data/indexes/colbert"
    CHROMA_DIR_ABS: str = "data/indexes/chroma"
    BM25_INDEX_PATH: str = "data/indexes/bm25_index.pkl"
    CHUNK_DATA_PATH: str = "data/indexes/chunks.jsonl"
    
    ENABLE_HYDE: bool = False
    ENABLE_MULTI_QUERY: bool = True
    MAX_QUERY_VARIANTS: int = 3
    ENABLE_VERIFICATION: bool = True
    
    # Modes Configuration
    DEFAULT_RETRIEVAL_MODE: str = "auto"
    ALLOW_RESEARCH_MODE: bool = True
    SHOW_DEBUG_BY_DEFAULT: bool = False

    # Top-K limits for modes
    FAST_TOP_K_DENSE: int = 15
    FAST_TOP_K_SPARSE: int = 15
    FAST_RERANK_TOP_K: int = 5
    FAST_MAX_CONTEXT_CHUNKS: int = 4

    DEEP_TOP_K_DENSE: int = 30
    DEEP_TOP_K_SPARSE: int = 30
    DEEP_TOP_K_COLBERT: int = 30
    DEEP_RERANK_TOP_K: int = 8
    DEEP_MAX_CONTEXT_CHUNKS: int = 8

    RESEARCH_TOP_K_DENSE: int = 40
    RESEARCH_TOP_K_SPARSE: int = 40
    RESEARCH_TOP_K_COLBERT: int = 40
    RESEARCH_RERANK_TOP_K: int = 10
    RESEARCH_MAX_CONTEXT_CHUNKS: int = 10

    # CORS
    FRONTEND_ORIGIN: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
