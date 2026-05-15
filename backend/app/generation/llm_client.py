from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> str: ...

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> AsyncGenerator[str, None]:
        # Default: fall back to full generate and yield as a single chunk
        answer = await self.generate(system_prompt, user_prompt, max_tokens, temperature)
        yield answer

    def generate_sync(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        return asyncio.run(
            self.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )


class MistralClient(BaseLLMClient):

    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        from app.generation.mistral_client import MistralClient as _Mistral
        self._client = _Mistral(api_key=api_key, model=model)
        self.model_name = model
        logger.info(f"MistralClient ready: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
        json_mode: bool = False,
    ) -> str:
        return await self._client.generate(
            system_prompt, user_prompt, max_tokens, temperature, json_mode=json_mode
        )

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> AsyncGenerator[str, None]:
        async for token in self._client.stream(system_prompt, user_prompt, max_tokens, temperature):
            yield token




class MockLLMClient(BaseLLMClient):

    def __init__(self):
        logger.warning("MockLLMClient active — set LLM_PROVIDER and API key in .env")

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return (
            "[Demo Mode] LLM not configured. "
            "Set LLM_PROVIDER and the corresponding API key in .env."
        )


def get_llm_client() -> BaseLLMClient:
    if not settings.MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY not set — falling back to MockLLMClient")
        return MockLLMClient()
    return MistralClient(api_key=settings.MISTRAL_API_KEY, model=settings.MISTRAL_MODEL)
