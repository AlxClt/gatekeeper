import re
from pathlib import Path

import yaml

from db.logger import DBLogger
from llm.interface import LLMInterface

_RULES_PATH = Path(__file__).parent / "rules" / "default.yaml"
_PROMPTS_PATH = Path(__file__).parent / "prompts" / "default.yaml"


class Verifier:
    def __init__(self, llm: LLMInterface, db_logger: DBLogger):
        self.llm = llm
        self.db_logger = db_logger
        self._rules = yaml.safe_load(_RULES_PATH.read_text())
        self._prompts = yaml.safe_load(_PROMPTS_PATH.read_text())

    async def verify(self, text: str) -> int:
        rules_text = "\n".join(f"- {r}" for r in self._rules["rules"])
        prompt = (
            f"{self._prompts['system']}\n\n"
            f"Rules:\n{rules_text}\n\n"
            f"Text to analyze:\n{text}\n\n"
            "Respond with only the digit 0 or 1."
        )

        response = await self.llm.complete(prompt)
        result = self._parse_result(response)
        await self.db_logger.log(text, result)
        return result

    @staticmethod
    def _parse_result(response: str) -> int:
        match = re.search(r"\b([01])\b", response.strip())
        return int(match.group(1)) if match else 0
