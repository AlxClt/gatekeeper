import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


class VerifyRequest(BaseModel):
    prompt: str


class VerifyResponse(BaseModel):
    result: int
    preprocessed_prompt: str


class VerifyRawResponse(BaseModel):
    result: int


@router.post("/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest, request: Request):
    try:
        result, preprocessed_prompt = await request.app.state.verifier.verify(body.prompt)
    except httpx.HTTPError as exc:
        logger.error(f"LLM backend unavailable after retries: {exc}")
        raise HTTPException(status_code=502, detail="LLM backend unavailable") from exc
    return VerifyResponse(result=result, preprocessed_prompt=preprocessed_prompt)


@router.post("/verify-raw", response_model=VerifyRawResponse)
async def verify_raw(body: VerifyRequest, request: Request):
    try:
        result = await request.app.state.verifier.verify_raw(body.prompt)
    except httpx.HTTPError as exc:
        logger.error(f"LLM backend unavailable after retries: {exc}")
        raise HTTPException(status_code=502, detail="LLM backend unavailable") from exc
    return VerifyRawResponse(result=result)
