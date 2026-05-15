from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    LLM_PROVIDER: Literal["mistral"] = "mistral"

    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 64

    PDF_THERMAL: str = "bee guide - thermal utility.pdf"
    PDF_ELECTRICAL: str = "bee guide - electrical utilities.pdf"

    @property
    def PDF_THERMAL_PATH(self) -> Path:
        return PROJECT_ROOT / self.PDF_THERMAL

    @property
    def PDF_ELECTRICAL_PATH(self) -> Path:
        return PROJECT_ROOT / self.PDF_ELECTRICAL

    DATA_DIR: str = "data"
    INDEX_DIR: str = "data/indexes"
    CHROMA_DIR: str = "data/indexes/chroma"
    BM25_INDEX_PATH: str = "data/indexes/bm25_index.pkl"
    CHUNKS_CACHE_PATH: str = "data/indexes/chunks.jsonl"

    @property
    def CHROMA_DIR_ABS(self) -> Path:
        return PROJECT_ROOT / self.CHROMA_DIR

    @property
    def BM25_INDEX_PATH_ABS(self) -> Path:
        return PROJECT_ROOT / self.BM25_INDEX_PATH

    @property
    def CHUNKS_CACHE_PATH_ABS(self) -> Path:
        return PROJECT_ROOT / self.CHUNKS_CACHE_PATH

    @property
    def BASE_DIR(self) -> Path:
        return PROJECT_ROOT

    OCR_DPI: int = 200
    OCR_MIN_CONFIDENCE: float = 30.0
    FORCE_OCR: bool = False

    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    RERANK_TOP_K: int = 5
    HYBRID_ALPHA: float = Field(default=0.6, ge=0.0, le=1.0)

    SEMANTIC_CHUNK_SIZE: int = 400
    SEMANTIC_CHUNK_OVERLAP: int = 50
    SECTION_CHUNK_SIZE: int = 1200
    TABLE_CHUNK_MAX_SIZE: int = 600

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_RELOAD: bool = True
    LOG_LEVEL: str = "INFO"

    NEXT_PUBLIC_API_URL: str = "http://localhost:8000"

    # Accepts a JSON list or comma-separated string; "*" allows all origins for public deployment
    CORS_ORIGINS: list[str] = ["*"]

    ENABLE_RERANKING: bool = True
    MAX_CONTEXT_CHUNKS: int = 5
    MAX_GENERATION_TOKENS: int = 1200


settings = Settings()
