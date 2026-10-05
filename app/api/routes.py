import logging
import os
import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

API_TOKEN = os.getenv("GATEKEEPER_API_TOKEN", "")
_bearer = HTTPBearer(auto_error=False)


def require_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    # Fail closed: an unconfigured token must never mean "open to everyone"
    if not API_TOKEN:
        logger.error("GATEKEEPER_API_TOKEN is not set — rejecting request")
        raise HTTPException(status_code=503, detail="Authentication not configured")
    if creds is None or not secrets.compare_digest(creds.credentials.encode(), API_TOKEN.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(dependencies=[Depends(require_token)])


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
