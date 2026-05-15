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


class GeminiClient(BaseLLMClient):

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model_name = model
        logger.info(f"GeminiClient ready: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> str:
        import google.generativeai as genai
        config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(full_prompt, generation_config=config),
        )
        return response.text


class OpenAIClient(BaseLLMClient):

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self.model_name = model
        logger.info(f"OpenAIClient ready: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class OllamaClient(BaseLLMClient):

    def __init__(self, base_url: str, model: str = "qwen2.5:7b"):
        from app.generation.ollama_client import OllamaClient as _Ollama
        self._ollama = _Ollama(model=model, base_url=base_url)
        self.model_name = model
        logger.info(f"OllamaClient ready: {model} @ {base_url}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
        json_mode: bool = False,
    ) -> str:
        return await self._ollama.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.MAX_GENERATION_TOKENS,
        temperature: float = 0.15,
    ) -> AsyncGenerator[str, None]:
        async for token in self._ollama.stream(system_prompt, user_prompt, max_tokens, temperature):
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
    provider = settings.LLM_PROVIDER.lower()

    if provider == "mistral":
        if not settings.MISTRAL_API_KEY:
            logger.warning("MISTRAL_API_KEY not set — falling back to MockLLMClient")
            return MockLLMClient()
        return MistralClient(api_key=settings.MISTRAL_API_KEY, model=settings.MISTRAL_MODEL)

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — falling back to MockLLMClient")
            return MockLLMClient()
        return GeminiClient(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set — falling back to MockLLMClient")
            return MockLLMClient()
        return OpenAIClient(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)

    if provider == "ollama":
        return OllamaClient(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)

    logger.warning(f"Unknown LLM_PROVIDER='{provider}' — using MockLLMClient")
    return MockLLMClient()
