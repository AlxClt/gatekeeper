import os

from app.llm.llm_adaptater import LLMInterface, LocalAdapter, OnlineAdapter

def create_llm() -> LLMInterface:
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "local":
        return LocalAdapter()
    if backend == "online":
        return OnlineAdapter()
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r}. Use 'local' or 'online'.")
