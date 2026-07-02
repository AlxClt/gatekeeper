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
from llm.llm_adaptater import OnlineAdapter
from verification.verifier import Verifier

_PROMPTS_PATH = Path(__file__).parent / "verification" / "prompts" / f"{os.getenv('PROMPT_NAME', 'default')}.yaml"

logger = logging.getLogger("uvicorn.error")


def _warmup_prompt() -> str:
    system_prompt = yaml.safe_load(_PROMPTS_PATH.read_text())
    return system_prompt.replace("{{input}}", "hello")


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

        timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
        await client.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": _warmup_prompt(), "stream": False, "options": {"temperature": 0}},
            timeout=timeout,
        )
        logger.info(f"Model '{model}' ready.")


async def _wait_for_online_model():
    """Opt-in warmup for OpenAI-compatible backends with local cold-start behavior
    (e.g. an Ollama instance behind a compatibility shim). Real hosted APIs don't
    need this — they're already warm — so it's gated behind ONLINE_LLM_WARMUP."""
    adapter = OnlineAdapter()
    retries = int(os.getenv("ONLINE_LLM_WARMUP_RETRIES", "3"))
    logger.info(f"Warming up online LLM backend (model '{adapter.model}')...")
    for attempt in range(1, retries + 1):
        try:
            await adapter.complete(_warmup_prompt())
            logger.info("Online LLM backend ready.")
            return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Online LLM backend has no model '{adapter.model}' at "
                    f"{adapter.base_url!r} (404 Not Found) — check ONLINE_LLM_MODEL "
                    "for a typo or a mismatch with what the backend actually serves."
                ) from exc
            logger.warning(f"Online LLM warmup attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            await asyncio.sleep(10)
        except Exception as exc:
            logger.warning(f"Online LLM warmup attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LLM_BACKEND", "online") == "local":
        await _wait_for_local_model()
    elif os.getenv("ONLINE_LLM_WARMUP", "false").lower() == "true":
        await _wait_for_online_model()
    llm = create_llm()
    db_logger = DBLogger(enabled=os.getenv("LOG_TO_DB", "false").lower() == "true")
    await db_logger.connect()
    app.state.verifier = Verifier(llm=llm, db_logger=db_logger)
    logger.info("Gatekeeper ready — POST /verify to classify prompts")
    yield
    await db_logger.close()


app = FastAPI(title="Gatekeeper", lifespan=lifespan)
app.include_router(router)
