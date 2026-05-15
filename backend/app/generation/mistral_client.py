from __future__ import annotations

import json
from typing import AsyncGenerator, Optional

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_TIMEOUT = 60.0


class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        self.api_key = api_key
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"MistralClient initialized: model={model}")

    def _payload(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        json_mode: bool = False,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.15,
        json_mode: bool = False,
    ) -> str:
        payload = self._payload(
            system_prompt, user_prompt, max_tokens, temperature, json_mode=json_mode
        )
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(MISTRAL_API_URL, json=payload, headers=self._headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Mistral API error {e.response.status_code}: {e.response.text}")
            raise RuntimeError(f"Mistral API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Mistral request timed out")
            raise RuntimeError("Mistral API request timed out.")

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.15,
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(
            system_prompt, user_prompt, max_tokens, temperature, stream=True
        )
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                async with client.stream(
                    "POST", MISTRAL_API_URL, json=payload, headers=self._headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError):
                            continue
        except httpx.HTTPStatusError as e:
            yield f"\n\n[Error: Mistral API {e.response.status_code}]"
        except httpx.ConnectError:
            yield "\n\n[Error: Cannot reach Mistral API. Check connectivity and MISTRAL_API_KEY.]"
        except Exception as e:
            yield f"\n\n[Generation error: {e}]"

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ) -> str:
        return await self.generate(
            system_prompt, user_prompt, max_tokens, temperature, json_mode=True
        )
