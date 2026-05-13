import logging
from pathlib import Path

import yaml

from db.logger import DBLogger
from app.llm.llm_adaptater import LLMInterface
from verification.preprocessing import preprocess

logger = logging.getLogger("uvicorn.error")

_PROMPTS_PATH = Path(__file__).parent / "prompts" / "default.yaml"


class Verifier:
    def __init__(self, llm: LLMInterface, db_logger: DBLogger):
        self.llm = llm
        self.db_logger = db_logger

    async def verify(self, text: str) -> int:
        """Two-pass: classify raw text, then preprocessed. Returns 1 if either pass detects a threat."""
        result_raw = await self._classify(text)
        preprocessed = preprocess(text).text
        result_clean = await self._classify(preprocessed)
        result = 1 if (result_raw or result_clean) else 0
        await self.db_logger.log(text, result)
        return result

    async def verify_one_pass(self, text: str) -> tuple[int, str]:
        """Single-pass: preprocess then classify. Returns (result, preprocessed_text)."""
        preprocessed = preprocess(text).text
        result = await self._classify(preprocessed)
        await self.db_logger.log(preprocessed, result)
        return result, preprocessed

    async def _classify(self, text: str) -> int:
        prompts = yaml.safe_load(_PROMPTS_PATH.read_text())
        prompt = prompts["system"].replace("{{input}}", text)
        response = await self.llm.complete(prompt)
        return self._parse_result(response)

    @staticmethod
    def _parse_result(response: str) -> int:
        stripped = response.strip()
        if stripped not in ("0", "1"):
            logger.warning(f"Unexpected LLM response: {stripped!r} — defaulting to 1")
        return 0 if stripped == "0" else 1
