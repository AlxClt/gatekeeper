
import logging
import os

import httpx

from abc import ABC, abstractmethod

logger = logging.getLogger("uvicorn.error")


def _warn_if_gpt_oss(model: str, think: bool) -> None:
    """gpt-oss models require "low"/"medium"/"high" reasoning effort, not a boolean —
    a plain on/off think value is silently ignored for this family."""
    if "gpt-oss" in model.lower():
        logger.warning(
            f"{model!r} is a gpt-oss model — it requires reasoning effort to be "
            f'"low", "medium", or "high" rather than a boolean; LLM_THINK={think} will be ignored.'
        )


class LLMInterface(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> str: ...

class LocalAdapter(LLMInterface):
    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://llm:11434")
        self.model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
        self.think = os.getenv("LLM_THINK", "false").lower() == "true"
        _warn_if_gpt_oss(self.model, self.think)

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "think": self.think,
                    "options": {"temperature": 0},
                },
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
        self.think = os.getenv("LLM_THINK", "false").lower() == "true"
        _warn_if_gpt_oss(self.model, self.think)

    async def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
        # OpenAI-compatible endpoints (incl. Ollama's) control reasoning via "reasoning_effort"
        # rather than a "think" boolean. Only override it to disable thinking — when LLM_THINK is
        # true, leave the field unset so each backend's own default reasoning level applies.
        if not self.think:
            body["reasoning_effort"] = "none"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
