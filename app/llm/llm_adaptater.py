
import os
from typing import Dict

import httpx

from abc import ABC, abstractmethod

USER_INPUT_TEMPLATE = """
<text_to_analyze>
{{input}}
</text_to_analyze>
"""

class LLMInterface(ABC):

    @staticmethod
    def format_user_input_text(user_input: str) -> str:
        return USER_INPUT_TEMPLATE.replace("{{input}}", user_input)

    @abstractmethod
    async def complete(self, system_prompt: str, user_input:str) -> str: ...

    @abstractmethod
    async def aclose(self) -> None: ...

class LocalAdapter(LLMInterface):
    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://llm:11434")
        self.model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
        self.num_ctx = int(os.getenv("LOCAL_LLM_NUM_CTX", "4096"))
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def _make_payload(self, system_prompt: str, user_input:str) -> Dict[str, str]:
        payload = {
            "model": self.model, 
            "prompt": system_prompt + "\n" + self.format_user_input_text(user_input=user_input),
            # TODO: uncomment and replace "prompt": with the following to implement KV caching. 
            # It is considerably faster but small llama models are too sensitive to the authority of system message and classify everything as a threat
            #"messages": [
            #        {"role": "system", "content": system_prompt},  # stable prefix → cached
            #        {"role": "user", "content": self.format_user_input_text(user_input)}, 
            #    ],
                "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 5,
                #"num_ctx": self.num_ctx #must be fixed for KV caching
                }
            }
        return payload

    async def complete(self, system_prompt: str, user_input:str) -> str:
        payload = await self._make_payload(system_prompt, user_input)
        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["response"]
    
    async def aclose(self):
        await self._client.aclose()

class OnlineAdapter(LLMInterface):
    """OpenAI-compatible adapter — works with OpenAI, Azure OpenAI, Groq, Together.ai, etc."""

    def __init__(self):
        self.api_key = os.getenv("ONLINE_LLM_API_KEY", "")
        self.base_url = os.getenv("ONLINE_LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("ONLINE_LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.getenv("ONLINE_LLM_TIMEOUT", "30.0"))
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def _make_payload(self, system_prompt: str, user_input:str) -> Dict[str, str]:
        payload = {
                    "model": self.model,
                    "messages": [
                    {"role": "system", "content": system_prompt},  
                    {"role": "user", "content": self.format_user_input_text(user_input)},       
                         ],
                    "max_tokens": 5,
                    "temperature": 0,
                }
        return payload

    async def complete(self, system_prompt: str, user_input:str) -> str:
        payload = await self._make_payload(system_prompt, user_input)
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def aclose(self):
        await self._client.aclose()