from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> str: ...
