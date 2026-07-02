import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI

from api.routes import router
from db.logger import DBLogger
from llm.factory import create_llm
from verification.verifier import Verifier

_PROMPTS_PATH = Path(__file__).parent / "verification" / "prompts" / f"{os.getenv('PROMPT_NAME', 'default')}.yaml"

logger = logging.getLogger("uvicorn.error")


async def _wait_for_local_model():
    url = os.getenv("LOCAL_LLM_URL", "http://llm:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
    logger.info(f"Waiting for model '{model}' to be available in Ollama...")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{url}/api/tags", timeout=5.0)
                if resp.status_code == 200:
                    names = [m["name"] for m in resp.json().get("models", [])]
                    if any(model in name for name in names):
                        break
            except Exception:
                pass
            await asyncio.sleep(10)

        logger.info(f"Loading model '{model}' into memory...")

        system_prompt = yaml.safe_load(_PROMPTS_PATH.read_text())
        warmup_prompt = system_prompt.replace("{{input}}", "hello")
        timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
        await client.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": warmup_prompt, "stream": False, "options": {"temperature": 0}},
            timeout=timeout,
        )
        logger.info(f"Model '{model}' ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LLM_BACKEND", "online") == "local":
        await _wait_for_local_model()
    llm = create_llm()
    db_logger = DBLogger(enabled=os.getenv("LOG_TO_DB", "false").lower() == "true")
    await db_logger.connect()
    app.state.verifier = Verifier(llm=llm, db_logger=db_logger)
    logger.info("Gatekeeper ready — POST /verify to classify prompts")
    yield
    await db_logger.close()


app = FastAPI(title="Gatekeeper", lifespan=lifespan)
app.include_router(router)
