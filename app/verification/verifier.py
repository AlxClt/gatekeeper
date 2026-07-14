import os
import time
import logging
from pathlib import Path

import httpx
import yaml

from db.logger import DBLogger
from llm.llm_adaptater import LLMInterface
from verification.preprocessing import preprocess

logger = logging.getLogger("uvicorn.error")

_PROMPTS_PATH = Path(__file__).parent / "prompts" / f"{os.getenv('PROMPT_NAME', 'default')}.yaml"
MAX_RETRIES = int(os.getenv('LLM_RETRY_CALLS', '0'))

class MalformedLLMOutputException(Exception):

    def __init__(self, LLMoutput:str, message=None):
        if message is None:
            message = f"Malformed LLM output: {LLMoutput}"
        super().__init__(message)

class Verifier:
    def __init__(self, llm: LLMInterface, db_logger: DBLogger):
        self.llm = llm
        self.db_logger = db_logger

    async def _preprocess(self, text: str) -> str:
        """Preprocesses the candidate prompt, returns cleaned prompt (not threatless, but )"""
        preprocessed = preprocess(text).text
        return preprocessed
    
    async def _classify(self, text: str) -> int:
        # Note: yaml kept for future extensibility
        n_retries = 0
        template = yaml.safe_load(_PROMPTS_PATH.read_text())
        prompt = template.replace("{{input}}", text)
        while n_retries<=MAX_RETRIES:
            try:
                response = await self.llm.complete(prompt)
                classification_result = self._parse_result(response)
            except httpx.HTTPError as exc:
                n_retries+=1
                time.sleep(0.5)
                if n_retries>=MAX_RETRIES: #> for the case where MAX_RETRIES=0
                    raise exc
                else:
                    logger.warning(f"LLM Server error {str(exc)} - retrying ({n_retries}/{MAX_RETRIES})...")
            except MalformedLLMOutputException:
                n_retries+=1
                if n_retries>=MAX_RETRIES:
                    logger.warning("Malformed LLM output detected after maximum retries reached - defaulting to 1")
                    classification_result = 1
                else:
                    logger.warning(f"Malformed LLM output returned - retrying ({n_retries}/{MAX_RETRIES})...")
            else:
                break

        return classification_result    
                
       
    async def verify(self, text: str) -> tuple[int, str]:
        """Single-pass: preprocess then classify. Returns (result, preprocessed_text)."""
        preprocessed = await self._preprocess(text)
        result = await self._classify(preprocessed)
        await self.db_logger.log(preprocessed, result)
        return result, preprocessed

    async def verify_raw(self, text: str) -> int:
        """Two-pass: classify raw text, then preprocessed. Returns 1 if either pass detects a threat."""
        result_raw = await self._classify(text)
        preprocessed = await self._preprocess(text)
        result_clean = await self._classify(preprocessed)
        result = max(result_raw, result_clean)
        await self.db_logger.log(text, result)
        return result
    
    @staticmethod
    def _parse_result(response: str) -> int:
        stripped = response.strip()
        if stripped not in ("0", "1"):
            raise MalformedLLMOutputException(LLMoutput=stripped)
        return 0 if stripped == "0" else 1


