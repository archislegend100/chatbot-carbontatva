"""
generation/ollama_client.py
============================
Ollama LLM client with streaming, JSON mode, and GPU support.

Ollama exposes a local REST API (default: http://localhost:11434).
This client wraps it with:
  - Async generation for FastAPI compatibility
  - Streaming token-by-token output via async generator
  - JSON mode (format=json) for structured planner output
  - Model availability checking
  - Configurable timeout and retry

GPU usage:
  Ollama automatically uses the RTX 4050 GPU if available.
  Verify with: `ollama run qwen2.5:7b "hello"` and check GPU in nvidia-smi.

Usage:
    from app.generation.ollama_client import OllamaClient
    client = OllamaClient(model="qwen2.5:7b")
    
    # Standard generation
    answer = await client.generate(system_prompt, user_prompt)
    
    # Streaming
    async for token in client.stream(system_prompt, user_prompt):
        print(token, end="", flush=True)
    
    # JSON mode (for planner)
    plan = await client.generate(..., json_mode=True)
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Optional

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0  # seconds


class OllamaClient:
    """
    Async Ollama LLM client.
    
    Args:
        model: Ollama model name (e.g. "qwen2.5:7b")
        base_url: Ollama API base URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.info(f"OllamaClient initialized: model={model} url={base_url}")

    async def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check if our model (or variant) is available
                base_name = self.model.split(":")[0]
                return any(base_name in m for m in models) or self.model in models
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return list of available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.15,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a full response (non-streaming).
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            max_tokens: Max tokens to generate (Ollama: num_predict)
            temperature: Sampling temperature (0.0 = deterministic)
            json_mode: If True, set format="json" for guaranteed JSON output
        
        Returns:
            Generated text string
        """
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,   # Context window
                "top_p": 0.9,
            },
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        
        except httpx.TimeoutException:
            logger.error(f"Ollama timeout after {self.timeout}s")
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama — is it running?")
            raise RuntimeError(
                "Cannot connect to Ollama at http://localhost:11434. "
                "Start it with: ollama serve"
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.15,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens as they are generated.
        
        Yields:
            Individual tokens / text chunks
        
        Usage:
            async for token in client.stream(system, user):
                yield token  # forward to SSE stream
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
                "top_p": 0.9,
            },
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        
        except httpx.ConnectError:
            yield "\n\n[Error: Cannot connect to Ollama. Run `ollama serve` first.]\n"
        except Exception as e:
            yield f"\n\n[Generation error: {e}]\n"

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> str:
        """Synchronous generation wrapper for the classifier fallback."""
        import asyncio
        return asyncio.run(
            self.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
