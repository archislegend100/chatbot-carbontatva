import httpx
import logging
from typing import Optional, AsyncGenerator, List
from app.config.settings import settings

logger = logging.getLogger(__name__)

class MistralAPIError(Exception):
    pass

class MistralClient:
    def __init__(self):
        self.api_key = settings.MISTRAL_API_KEY
        self.model = settings.MISTRAL_MODEL
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Non-streaming text generation."""
        if not self.api_key:
            raise MistralAPIError("MISTRAL_API_KEY is not set.")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.MISTRAL_TEMPERATURE,
            "max_tokens": settings.MISTRAL_MAX_TOKENS,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                logger.error(f"Mistral API Status Error: {e.response.status_code}")
                raise MistralAPIError(f"Mistral API Error. Please check your configuration.")
            except httpx.RequestError as e:
                logger.error(f"Mistral API Request Error: {str(e)}")
                raise MistralAPIError("Failed to communicate with Mistral API.")

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streaming text generation."""
        if not self.api_key:
            raise MistralAPIError("MISTRAL_API_KEY is not set.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.MISTRAL_TEMPERATURE,
            "max_tokens": settings.MISTRAL_MAX_TOKENS,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk.startswith("data: "):
                            data = chunk[6:]
                            if data == "[DONE]":
                                break
                            import json
                            try:
                                parsed = json.loads(data)
                                content = parsed["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Mistral Streaming Error: {str(e)}")
                yield " [Error: Communication with Mistral API failed.]"

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation."""
        if not self.api_key:
            raise MistralAPIError("MISTRAL_API_KEY is not set.")
            
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                # Sort by index to ensure order matches input
                embeddings = []
                # Mistral returns data array where each element has 'embedding' and 'index'
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                for item in sorted_data:
                    embeddings.append(item["embedding"])
                return embeddings
            except httpx.HTTPStatusError as e:
                logger.error(f"Mistral API Status Error during embedding: {e.response.status_code}")
                raise MistralAPIError("Failed to generate embeddings. Please check your configuration.")
            except httpx.RequestError as e:
                logger.error(f"Mistral API Request Error during embedding: {str(e)}")
                raise MistralAPIError("Failed to communicate with Mistral API.")

mistral_client = MistralClient()
