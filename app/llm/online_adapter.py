import os

import httpx

from llm.interface import LLMInterface


class OnlineAdapter(LLMInterface):
    """OpenAI-compatible adapter — works with OpenAI, Azure OpenAI, Groq, Together.ai, etc."""

    def __init__(self):
        self.api_key = os.getenv("ONLINE_LLM_API_KEY", "")
        self.base_url = os.getenv("ONLINE_LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("ONLINE_LLM_MODEL", "gpt-4o-mini")

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
