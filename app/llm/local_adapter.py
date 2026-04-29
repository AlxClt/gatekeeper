import os

import httpx

from llm.interface import LLMInterface


class LocalAdapter(LLMInterface):
    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://llm:11434")
        self.model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            )
            response.raise_for_status()
            return response.json()["response"]
