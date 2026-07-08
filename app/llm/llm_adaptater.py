
import os

import httpx

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> str: ...

class LocalAdapter(LLMInterface):
    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://llm:11434")
        self.model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            )
            response.raise_for_status()
            return response.json()["response"]


class OnlineAdapter(LLMInterface):
    """OpenAI-compatible adapter — works with OpenAI, Azure OpenAI, Groq, Together.ai, etc."""

    def __init__(self):
        self.api_key = os.getenv("ONLINE_LLM_API_KEY", "")
        self.base_url = os.getenv("ONLINE_LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("ONLINE_LLM_MODEL", "undefined")
        self.timeout = float(os.getenv("ONLINE_LLM_TIMEOUT", "30"))

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
