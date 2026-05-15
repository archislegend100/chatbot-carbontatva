"""
cache/query_cache.py
=====================
LRU query-level response cache using diskcache.

Caches the full JSON response for identical queries.
This avoids repeated LLM calls for the same question in a session.

Cache key: SHA-256(query + tool_mode + domain_filter + explanation_level)
TTL: 1 hour by default (configurable)

Storage: data/query_cache/ (disk-backed, survives server restarts)

Usage:
    cache = QueryCache()
    
    result = cache.get("What is boiler efficiency?", "qa", "thermal", "engineer")
    if result:
        return result  # Instant cache hit
    
    # ... run pipeline ...
    
    cache.set("What is boiler efficiency?", "qa", "thermal", "engineer", response)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = settings.BASE_DIR / "data" / "query_cache"

TTL_SECONDS = 3600  # 1 hour


class QueryCache:
    """
    Disk-backed LRU query cache.
    
    Falls back to an in-memory dict if diskcache is unavailable.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = TTL_SECONDS):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._mem: dict[str, Any] = {}
        self._disk_available = False
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import diskcache
            self._disk = diskcache.Cache(str(self.cache_dir))
            self._disk_available = True
            logger.info(f"QueryCache initialized (disk): {self.cache_dir}")
        except ImportError:
            logger.warning("diskcache not installed — using in-memory cache only")

    def _key(
        self,
        query: str,
        tool_mode: str = "auto",
        domain: Optional[str] = None,
        level: str = "engineer",
    ) -> str:
        raw = f"{query.strip().lower()}|{tool_mode}|{domain or ''}|{level}"
        return "q:" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        tool_mode: str = "auto",
        domain: Optional[str] = None,
        level: str = "engineer",
    ) -> Optional[dict]:
        """Return cached response or None."""
        key = self._key(query, tool_mode, domain, level)
        
        if self._disk_available:
            value = self._disk.get(key)
            if value is not None:
                logger.info(f"Query cache HIT: {key[:12]}...")
                return value
        else:
            return self._mem.get(key)
        
        return None

    def set(
        self,
        query: str,
        tool_mode: str = "auto",
        domain: Optional[str] = None,
        level: str = "engineer",
        response: dict = None,
    ) -> None:
        """Store a response in cache."""
        if response is None:
            return
        key = self._key(query, tool_mode, domain, level)
        
        if self._disk_available:
            self._disk.set(key, response, expire=self.ttl)
        else:
            self._mem[key] = response
        
        logger.debug(f"Query cache SET: {key[:12]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._disk_available:
            self._disk.clear()
        self._mem.clear()
        logger.info("Query cache cleared")

    def stats(self) -> dict:
        if self._disk_available:
            return {
                "backend": "disk",
                "size": len(self._disk),
                "directory": str(self.cache_dir),
            }
        return {"backend": "memory", "size": len(self._mem)}
